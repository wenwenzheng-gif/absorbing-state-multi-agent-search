"""Strict corrected-data validation, immutable staging, analysis and append-only HF publication."""
import collections, hashlib, json, pathlib, shutil, subprocess, time, zipfile
from common import ROOT, OUT, REF, SEEDS, TASKS, POLICIES, SETTINGS, DATASET, batch, cell_dir, completed, config, rows, verify_sources, write

def require_settings(mapping):
    for key in ['model','temperature','reasoning_effort','enable_thinking']:
        assert mapping.get(key)==SETTINGS[key],f'Invalid {key}: {mapping.get(key)!r}'

def audit_cell(task,seed,policy):
    d=cell_dir(task,seed,policy);manifest=json.loads((d/'manifest.json').read_text());c=manifest['config'];want=config(task,seed,policy)
    assert c==want,f'Configuration differs from pinned baseline: {d.name}'
    require_settings(c['policy']['llm'])
    assert c['task_seeds'][0] not in REF['used_task_seeds'][task]
    assert 'medium' not in d.parts and 'smoke' not in d.name
    episodes=rows(d/'episodes.jsonl');good=[r for r in episodes if r['status']=='ok']
    target={f'd{degree}_t{c["task_seeds"][0]}_r0' for degree in c['degrees']}
    assert len(good)==len(target) and {r['episode_id'] for r in good}==target,'Missing or duplicated successful episodes'
    requests=rows(d/'requests.jsonl');responses=rows(d/'responses.jsonl');assert requests and responses
    for req in requests:
        params=req.get('request_parameters',{})
        assert params.get('model')=='gpt-6-luna' and params.get('temperature')==1.0
        assert params.get('reasoning_effort')=='none' and isinstance(params.get('seed'),int)
        assert 'chat_template_kwargs' not in params and req.get('request_body_sha256')
        assert req['episode'] in target
    for response in responses:
        assert response.get('model','').startswith('gpt-6-luna'),'Backend model mismatch'
        assert response['episode'] in target
        details=(response.get('usage') or {}).get('completion_tokens_details') or {}
        assert details.get('reasoning_tokens')==0,'Reasoning tokens not proven to be zero'
        assert not response.get('reasoning'),'Thinking content returned'
        actual=response.get('sampling_audit') or {}
        assert actual.get('actual_model')==response['model'] and actual.get('upstream_response_id')
        assert actual.get('effective_temperature')==1.0 and actual.get('effective_reasoning_effort')=='none' and actual.get('reasoning_tokens')==0
        assert actual.get('temperature_forwarded') is False and actual.get('seed_forwarded') is False
    summary=json.loads((d/'run_summary.json').read_text());require_settings(summary)
    assert summary['planned_episodes']==len(target)
    return {'cell':d.name,'task':task,'seed':seed,'policy':policy,'task_seed':c['task_seeds'][0],'episodes':len(good),'requests':len(requests),'responses':len(responses),'api_requests':len(rows(d/'api_attempts.jsonl')),'transport_errors':dict(collections.Counter(r['kind'] for r in rows(d/'transport_errors.jsonl'))),'actual_models':sorted({r['model'] for r in responses})}

def aggregate():
    codes={}
    for task in TASKS:
        for policy in POLICIES:
            d=OUT/'aggregate'/f'{task}_{policy}';d.mkdir(parents=True,exist_ok=True)
            c=config(task,7,policy);c['task_seeds']=[config(task,s,policy)['task_seeds'][0] for s in SEEDS]
            write(d/'manifest.json',{'config':c,'run_id':f'gpt-6-luna_seed7-9_{task}_{policy}',**SETTINGS,'seed_ids':list(SEEDS)})
            for ledger in ['episodes','events']:
                with (d/f'{ledger}.jsonl').open('w') as f:
                    for seed in SEEDS:
                        for row in rows(cell_dir(task,seed,policy)/f'{ledger}.jsonl'):
                            if ledger=='episodes' and row.get('status')!='ok':continue
                            f.write(json.dumps(row)+'\n')
            for module in ['summarize','critical_degree']:
                with (d/f'{module}.log').open('wb') as log:
                    result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m',f'src.agent_system.analysis.{module}','--run-dir',str(d)],cwd=ROOT,stdout=log,stderr=log)
                codes[f'{task}_{policy}/{module}']=result.returncode
                assert result.returncode==0,f'Analysis failed: {task}/{policy}/{module}'
    return codes

def stage(cells,analysis_codes):
    target=OUT/'validated_upload';target.mkdir(exist_ok=True)
    # Completed writers have exited. Copy only explicitly validated scientific files.
    allowed=['config.json','manifest.json','requests.jsonl','responses.jsonl','events.jsonl','api_attempts.jsonl','run_summary.json','checkpoint.json']
    for task in TASKS:
        for seed in SEEDS:
            for policy in POLICIES:
                source=cell_dir(task,seed,policy);dest=target/source.relative_to(OUT);dest.mkdir(parents=True,exist_ok=True)
                for name in allowed:
                    p=source/name
                    if p.exists():shutil.copy2(p,dest/name)
                with (dest/'episodes.jsonl').open('w') as f:
                    for row in rows(source/'episodes.jsonl'):
                        if row['status']=='ok':f.write(json.dumps(row)+'\n')
    for source in (OUT/'aggregate').rglob('*'):
        if source.is_file() and source.name not in ['episodes.jsonl','events.jsonl'] and source.suffix in ['.json','.csv','.md','.png']:
            dest=target/'pooled_analysis'/source.relative_to(OUT/'aggregate');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    report={**json.loads((OUT/'batch.json').read_text()),'status':'VALIDATED','cells':cells,'analysis_exit_codes':analysis_codes,'completed_episodes_by_task':dict(collections.Counter({t:sum(c['episodes'] for c in cells if c['task']==t) for t in TASKS})),'api_requests':sum(c['api_requests'] for c in cells),'responses':sum(c['responses'] for c in cells),'actual_models':sorted({m for c in cells for m in c['actual_models']}),'scientific_workers_peak':9,'wall_seconds':time.time()-json.loads((OUT/'batch.json').read_text())['created_at'],'source_archive_retained':REF['source_archive'],'baseline_endpoint':'https://api.openai.com/v1','new_endpoint':'local codex_api','same_proxy_as_baseline':False,'temperature_forwarded':False,'effective_temperature':1.0,'sampling_seed_forwarded':False,'endpoint_migration_recorded':True,'no_cancelled_data':not (ROOT/'outputs/gpt-6-luna').exists(),'no_smoke_data':True,'no_failed_episode':True,'no_partial_seed':True,'no_seed_overlap':True,'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    require_settings(report);assert report['completed_episodes_by_task']=={'synthetic':45,'tcas':63,'physics':63}
    capacities=rows(OUT/'capacity.jsonl');report['peak_stable_global_limit']=max([r['upstream_limit'] for r in capacities if not r['window_errors']] or [24])
    report['peak_stable_upstream_concurrency']=max([r.get('peak_active_upstream_calls',0) for r in capacities if not r['window_errors']] or [0])
    errors=collections.Counter()
    for cell in cells:errors.update(cell['transport_errors'])
    import urllib.request
    proxy_totals=collections.Counter()
    for phase in rows(OUT/'operations/proxy_phases.jsonl'):proxy_totals.update(phase['metrics'])
    live=json.load(urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=5));proxy_totals.update(live['metrics'])
    client_retries=sum(sum(r.get('transport_attempt',0)>0 for r in rows(cell_dir(t,s,p)/'api_attempts.jsonl')) for t in TASKS for s in SEEDS for p in POLICIES)
    client_5xx=sum(n for k,n in errors.items() if k.startswith('HTTP 5'))
    report.update(effective_temperature=1.0,thinking='disabled',api_requests_scientific_client_attempts=report['api_requests'],
                  total_api_requests_including_proxy_retries_smoke_and_preflight=proxy_totals['upstream_attempts'],
                  transport_retries=client_retries+proxy_totals['upstream_retries'],client_transport_retries=client_retries,
                  proxy_rate_limit_retries=proxy_totals['upstream_retries'],http_429=proxy_totals['upstream_429'],
                  http_5xx=max(client_5xx,proxy_totals['upstream_5xx']),client_http_5xx=client_5xx,
                  upstream_http_5xx=proxy_totals['upstream_5xx'],timeouts=sum(n for k,n in errors.items() if 'Timeout' in k)+proxy_totals['upstream_timeouts'],proxy_metrics=dict(proxy_totals))
    write(target/'metadata.json',report)
    preflight=json.loads((OUT/'preflight.json').read_text());assert preflight['status']=='PASS' and preflight['effective_temperature']==1.0 and preflight['reasoning_tokens']==0;write(target/'request_path_preflight.json',preflight)
    readme=f'''# GPT-6-Luna: additional seed batches 7, 8, 9\n\nExtends `{REF['source_archive']}` without replacing its six valid seed batches.\nModel gpt-6-luna; requested and effective temperature 1.0; reasoning effort none; thinking disabled.\nTask seeds: {json.dumps(report['task_seed_ids'])}.\nThe original C0/C1/I1 scientific configurations are unchanged. The valid baseline used the official API; this authorized extension uses local codex_api. The proxy omits temperature (raw upstream response proves the default is 1.0) and does not forward per-request sampling seeds. This transport difference is explicitly recorded; local task RNG seeds and request-seed derivation are unchanged. Nine independent task/seed jobs cover 27 cells and 171 successful episodes.\nSmoke tests, cancelled outputs, failed episodes, credentials, logs and process state are excluded. See metadata.json and request_path_preflight.json for audit evidence.\n'''
    (target/'README.md').write_text(readme)
    from dotenv import dotenv_values
    from huggingface_hub import get_token
    env=dotenv_values(preflight['env_file']);secrets=[v.encode() for k,v in env.items() if v and ('KEY' in k or 'TOKEN' in k) and len(v)>8]
    if get_token():secrets.append(get_token().encode())
    for p in target.rglob('*'):
        if p.is_file():
            assert p.suffix in ['.json','.jsonl','.csv','.md','.png'],f'Unexpected staging artifact: {p}'
            data=p.read_bytes();assert not any(secret in data for secret in secrets),f'Secret in staged file: {p.name}'
            if p.name=='episodes.jsonl':assert all(r['status']=='ok' for r in rows(p))
    archive=OUT/'gpt-6-luna_seed7-9_runs.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(target)))
    report['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();write(OUT/'validation_report.json',report)
    return archive,report

def upload(archive,report):
    from huggingface_hub import HfApi,CommitOperationAdd,get_token
    from dotenv import dotenv_values
    def token():
        path=json.loads((OUT/'preflight.json').read_text())['env_file']
        return dotenv_values(path).get('HF_TOKEN') or get_token()
    while not token():
        write(OUT/'upload_status.json',{**SETTINGS,'status':'AUTHENTICATION_REQUIRED','dataset':DATASET,'validated_archive':archive.name})
        time.sleep(60)
    api=HfApi(token=token());api.whoami()
    files=api.list_repo_files(DATASET,repo_type='dataset')
    paths={'archive':'gpt-6-luna/'+archive.name,'metadata':'gpt-6-luna/gpt-6-luna_seed7-9_metadata.json','readme':'gpt-6-luna/gpt-6-luna_seed7-9_README.md'}
    assert all(p not in files for p in paths.values()),'Destination exists; refusing to replace existing valid data.'
    before={p.rfilename:(getattr(p.lfs,'sha256',None) or (p.lfs.get('sha256') if isinstance(p.lfs,dict) else None)) if p.lfs else p.blob_id for p in api.repo_info(DATASET,repo_type='dataset',files_metadata=True).siblings if p.rfilename.startswith('gpt-6-luna/')}
    ops=[CommitOperationAdd(path_in_repo=paths['archive'],path_or_fileobj=str(archive)),CommitOperationAdd(path_in_repo=paths['metadata'],path_or_fileobj=str(OUT/'validation_report.json')),CommitOperationAdd(path_in_repo=paths['readme'],path_or_fileobj=str(OUT/'validated_upload/README.md'))]
    commit=api.create_commit(DATASET,repo_type='dataset',operations=ops,commit_message='Add three validated temperature-one, reasoning-none GPT-6-Luna seed batches (7-9)')
    after={p.rfilename:(getattr(p.lfs,'sha256',None) or (p.lfs.get('sha256') if isinstance(p.lfs,dict) else None)) if p.lfs else p.blob_id for p in api.repo_info(DATASET,repo_type='dataset',files_metadata=True).siblings if p.rfilename.startswith('gpt-6-luna/')}
    assert all(after.get(p)==digest for p,digest in before.items()),'Existing HF data changed'
    write(OUT/'upload_status.json',{**SETTINGS,'status':'UPLOADED','dataset':DATASET,'paths':paths,'commit':commit.oid,'prior_files_unchanged':True})
    return paths

def finish():
    verify_sources();assert not (ROOT/'outputs/gpt-6-luna').exists()
    cells=[audit_cell(t,s,p) for t in TASKS for s in SEEDS for p in POLICIES]
    for task in TASKS:
        for seed in SEEDS:assert json.loads((OUT/'jobs'/f'{task}_seed{seed}'/'status.json').read_text())['status']=='COMPLETE'
    codes=aggregate();archive,report=stage(cells,codes)
    write(OUT/'combined_summary.json',report)
    paths=upload(archive,report)
    report.update(status='COMPLETE',hugging_face_paths=paths,total_wall_seconds=time.time()-json.loads((OUT/'batch.json').read_text())['created_at'])
    compact=ROOT/'results/gpt-6-luna-t1-none-new3';write(compact/'final_report.json',report);write(OUT/'final_report.json',report)
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip();assert branch.startswith('experiment/') and branch!='main'
    subprocess.run(['git','add','scripts/luna','src/agent_system/policies/openai_policy.py','src/agent_system/driver.py','tests/agent_system/test_luna_corrected.py',str(compact.relative_to(ROOT))],cwd=ROOT,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Finalize and audit three corrected GPT-6-Luna seed batches'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin',branch],cwd=ROOT,check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();write(OUT/'publication.json',{'commit':sha,'branch':branch,'status':'PUSHED',**SETTINGS})
    write(OUT/'final_status.json',{'status':'COMPLETE','hugging_face_paths':paths,'git_commit':sha,**SETTINGS})
