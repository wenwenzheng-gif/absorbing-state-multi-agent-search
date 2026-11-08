"""Corrected additional seeds must preserve the valid baseline and wire settings."""
import asyncio,copy,json,pathlib,sys,tempfile,unittest
from unittest.mock import patch
import httpx
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/luna'))
from common import REF,SEEDS,TASKS,POLICIES,config,literal_seed,expected,verify_sources
from src.agent_system.schemas import LLMConfig,load_run_config
from src.agent_system.policies.openai_policy import LLMBackend

class CorrectedConfigurationTests(unittest.TestCase):
    def test_unproven_transport_fragment_requires_rejection_and_audited_repair(self):
        from audit import audit_transport_fragment
        fragment=dict(cache_key='fragment',episode='d0',agent=1,round=2,phase='branch',batch=0,attempt=0,content='{"updates":',finish_reason='stop',usage={'total_tokens':0})
        repair=dict(fragment,cache_key='repair',attempt=1,content='{}',sampling_audit={'upstream_response_id':'resp_test','actual_model':'gpt-6-luna','effective_temperature':1.0,'effective_reasoning_effort':'none','reasoning_tokens':0})
        requests=[{'cache_key':'repair','messages':[{'content':'Your previous answer broke a hard constraint: invalid JSON'}]}]
        self.assertEqual(audit_transport_fragment(fragment,requests,[fragment,repair])['repair_cache_key'],'repair')
        with self.assertRaises(AssertionError):audit_transport_fragment(dict(fragment,content='{}'),requests,[fragment,repair])
        with self.assertRaises(AssertionError):audit_transport_fragment(fragment,requests,[fragment])
        with self.assertRaises(AssertionError):audit_transport_fragment(fragment,[],[fragment,repair])

    def test_exactly_three_unused_seed_batches_and_original_grids(self):
        self.assertEqual(SEEDS,(7,8,9))
        for task in TASKS:
            seeds=[literal_seed(task,s) for s in SEEDS]
            self.assertEqual(len(set(seeds)),3)
            self.assertFalse(set(seeds)&set(REF['used_task_seeds'][task]))
            for policy in POLICIES:
                c=config(task,7,policy);base=copy.deepcopy(REF['cells'][f'{task}_{policy}'])
                self.assertEqual(load_run_config(c).as_dict(),c)
                for obj in [c,base]:
                    for k in ['name','task_seeds','output_root']:obj.pop(k)
                    for k in ['temperature','max_wall_seconds']:obj['policy']['llm'].pop(k)
                self.assertEqual(c,base)
        self.assertEqual(sum(3*expected(t) for t in TASKS),171)
        verify_sources()

    def test_required_runtime_sampling_and_no_wall_cap(self):
        for task in TASKS:
            for seed in SEEDS:
                for policy in POLICIES:
                    llm=config(task,seed,policy)['policy']['llm']
                    self.assertEqual((llm['model'],llm['temperature'],llm['reasoning_effort'],llm['enable_thinking']),('gpt-6-luna',1.0,'none',False))
                    self.assertIsNone(llm['max_wall_seconds'])
                    self.assertEqual(llm['concurrency'],12)

    def test_official_api_is_blocked_before_any_request_or_scientific_job(self):
        import run as corrected_runner
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(corrected_runner,'OUT',pathlib.Path(tmp)), patch.object(corrected_runner,'dotenv_values',return_value={'OPENAI_BASE_URL':'https://api.openai.com/v1','OPENAI_API_KEY':'test-only'}):
                self.assertFalse(asyncio.run(corrected_runner.preflight(pathlib.Path('/unused.env'))))
                record=json.loads((pathlib.Path(tmp)/'preflight.json').read_text())
                self.assertEqual(record['status'],'BLOCKED_PREFLIGHT')
                self.assertEqual(record['scientific_workers_launched'],0)

    def test_initial_pool_starts_all_nine_task_seed_jobs(self):
        import run as corrected_runner
        with patch.object(corrected_runner.pathlib.Path,'iterdir',return_value=[]):
            queue=corrected_runner.schedule()
        self.assertEqual(len(queue),27)
        self.assertEqual(len(set(queue)),27)
        self.assertEqual({(task,seed) for task,seed,policy in queue[:9]},{(task,seed) for task in TASKS for seed in SEEDS})
        self.assertTrue(all(policy=='C0' for task,seed,policy in queue[:9]))

    def test_small_stream_error_clusters_do_not_reset_capacity_observation(self):
        import run as corrected_runner
        self.assertFalse(corrected_runner.capacity_unstable({'HTTP 502':6},{},123))
        self.assertTrue(corrected_runner.capacity_unstable({'ReadTimeout':6},{},123))
        self.assertTrue(corrected_runner.capacity_unstable({}, {'upstream_429':3},1000))
        self.assertTrue(corrected_runner.capacity_unstable({'HTTP 502':20},{},50))

    def test_upload_audit_rejects_incorrect_sampling(self):
        from audit import require_settings
        require_settings({'model':'gpt-6-luna','temperature':1.0,'reasoning_effort':'none','enable_thinking':False})
        for key,value in [('temperature',None),('reasoning_effort','medium'),('enable_thinking',True)]:
            bad={'model':'gpt-6-luna','temperature':1.0,'reasoning_effort':'none','enable_thinking':False};bad[key]=value
            with self.assertRaises(AssertionError):require_settings(bad)

    def test_actual_http_body_contains_temperature_none_effort_and_seed(self):
        async def check():
            recorded=[];captured=[]
            def handle(request):
                body=json.loads(request.content);captured.append(body)
                return httpx.Response(200,json={'model':'gpt-6-luna','choices':[{'finish_reason':'stop','message':{'content':'{}'}}],'usage':{'completion_tokens_details':{'reasoning_tokens':0}}})
            cfg=LLMConfig(model='gpt-6-luna',base_url='https://unit.invalid/v1',api_style='openai',temperature=1.0,reasoning_effort='none',enable_thinking=False)
            backend=LLMBackend(cfg,'not-a-real-credential',request_sink=recorded.append)
            await backend._client.aclose();backend._client=httpx.AsyncClient(base_url=cfg.base_url,transport=httpx.MockTransport(handle))
            await backend.complete(messages=[{'role':'user','content':'{}'}],schema={'type':'object'},schema_name='test',cache_key='new-only',metadata={},extra_body={'seed':123})
            await backend.aclose()
            self.assertEqual(captured[0]['temperature'],1.0);self.assertEqual(captured[0]['reasoning_effort'],'none');self.assertEqual(captured[0]['seed'],123)
            self.assertNotIn('chat_template_kwargs',captured[0]);self.assertEqual(recorded[0]['request_parameters']['temperature'],1.0)
        asyncio.run(check())

if __name__=='__main__':unittest.main()
