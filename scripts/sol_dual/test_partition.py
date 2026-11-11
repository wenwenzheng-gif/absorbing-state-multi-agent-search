import contextlib,fcntl,importlib.util,io,json,pathlib,tempfile,threading,time,unittest
spec=importlib.util.spec_from_file_location('partition',pathlib.Path(__file__).with_name('partition.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class PartitionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name);self.out=self.root/'out';self.out.mkdir();self.dest=self.root/'published';self.names=[f'synthetic_C0_s{i}_d39' for i in range(684)]
  (self.out/'batch.json').write_text(json.dumps({'jobs':self.names,'prompt_hashes':{},'reference_commit':'test'}));(self.out/'allocation.json').write_text(json.dumps({'owners':{n:'A' for n in self.names},'b_enabled':False}))
  for n in self.names:
   d=self.out/'jobs'/n;d.mkdir(parents=True);(d/'config.json').write_text(json.dumps({'degrees':[39]}));(d/'channel_provenance.json').write_text('{}')
  self.pre=self.root/'preflight.json';self.pre.write_text(json.dumps({'passed':True,'actual_model':'gpt-6-sol','http_status':200,'estimated_cost_usd':.03}))
 def tearDown(self):self.tmp.cleanup()
 def reserve(self):
  with contextlib.redirect_stdout(io.StringIO()):mod.reserve(self.out,self.dest,self.pre)
 def test_failed_preflight_never_partitions(self):
  self.pre.write_text(json.dumps({'passed':False}))
  with self.assertRaises(AssertionError):self.reserve()
  self.assertFalse(json.loads((self.out/'allocation.json').read_text())['b_enabled'])
 def test_disjoint_budgeted_assignment_preserves_running(self):
  running=self.names[-1];(self.out/'jobs'/running/'status.json').write_text('{"state":"RUNNING"}')
  self.reserve();a=json.loads((self.dest/'allocation.json').read_text());self.assertEqual(len(a['jobs']),24);self.assertNotIn(running,a['jobs']);self.assertLessEqual(len(a['jobs'])*4+.03,100);self.assertEqual(len(a['owners']),684)
  with self.assertRaises(AssertionError):self.reserve()
 def test_atomic_lock_serializes_claim_and_partition(self):
  ready=threading.Event();done=threading.Event();name=self.names[-1]
  def claim():
   with (self.out/'allocation.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX);ready.set();time.sleep(.05);(self.out/'jobs'/name/'status.json').write_text('{"state":"RUNNING"}')
   done.set()
  t=threading.Thread(target=claim);t.start();ready.wait();self.reserve();t.join();self.assertTrue(done.is_set());self.assertNotIn(name,json.loads((self.dest/'allocation.json').read_text())['jobs'])
if __name__=='__main__':unittest.main()
