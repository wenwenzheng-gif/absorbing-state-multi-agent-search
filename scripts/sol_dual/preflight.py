"""One authorized API compatibility call. Never prints or persists the secret."""
import os,json,time,pathlib,urllib.request,urllib.error,hashlib,re
OUT=pathlib.Path('artifacts/channel-b-preflight');OUT.mkdir(parents=True,exist_ok=True)
key=os.environ.get('OPENAI_API_KEY',''); BASE='https://api.openai.com/v1'
def scrub(text):
 if key:text=text.replace(key,'<redacted>')
 return re.sub(r'sk-[A-Za-z0-9_-]+','<redacted>',text)
def write(name,data):
 p=OUT/name;p.write_text(scrub(json.dumps(data,indent=2)))
def call(path,body=None):
 req=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 started=time.time()
 try:
  with urllib.request.urlopen(req,timeout=300) as r:status=r.status;headers=r.headers;raw=r.read().decode()
 except urllib.error.HTTPError as e:status=e.code;headers=e.headers;raw=e.read().decode()
 except Exception as e:return {'state':'AMBIGUOUS','error_type':type(e).__name__,'elapsed_s':time.time()-started}
 try:payload=json.loads(scrub(raw))
 except ValueError:payload={'non_json_response':True}
 allowed={k:v for k,v in headers.items() if k.lower().startswith('x-ratelimit-') or k.lower() in ('x-request-id','retry-after')}
 return {'status':status,'headers':allowed,'response':payload,'elapsed_s':time.time()-started}
report={'channel':'B','base_url':BASE,'requested_model':'gpt-6-sol','reasoning_effort':'medium','secret_present':bool(key),'run_id':os.environ.get('GITHUB_RUN_ID'),'run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT'),'passed':False}
if not key:report['blocker']='SECRET_NOT_INJECTED'
elif os.environ.get('GITHUB_RUN_ATTEMPT','1')!='1':report['blocker']='RERUN_DENIED_TO_PREVENT_DUPLICATE_CALL'
else:
 metadata=call('/models/gpt-6-sol');write('model_access.json',metadata)
 if metadata.get('status')!=200:report['blocker']='MODEL_ACCESS_FAILED';report['model_access_status']=metadata.get('status');report['error']=metadata.get('response',{}).get('error')
 else:
  body=json.loads(pathlib.Path('scripts/sol_dual/preflight_request.json').read_text());assert body['model']=='gpt-6-sol' and body['reasoning_effort']=='medium' and 'temperature' not in body
  write('request.json',body);write('request_intent.json',{'state':'INFLIGHT','time':time.time(),'retries':0})
  result=call('/chat/completions',body);write('raw_response.json',result)
  payload=result.get('response',{});report.update(http_status=result.get('status'),actual_model=payload.get('model'),usage=payload.get('usage'),rate_headers=result.get('headers'),latency_s=result.get('elapsed_s'))
  if result.get('status')==200:
   try:
    choice=payload['choices'][0]; data=json.loads(choice['message']['content']);report['json_valid']=True;report['finish_reason']=choice['finish_reason'];report['passed']=payload['model']=='gpt-6-sol' and choice['finish_reason']=='stop'
    usage=payload.get('usage',{});cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0);report['estimated_cost_usd']=((usage.get('prompt_tokens',0)-cached)*2+cached*.2+usage.get('completion_tokens',0)*10)/1e6
   except (KeyError,ValueError,IndexError):report['blocker']='INVALID_OUTPUT'
  else:report['blocker']='API_CALL_FAILED';report['error']=payload.get('error',{'type':result.get('error_type')})
write('summary.json',report)
print(scrub(json.dumps(report,indent=2)))
raise SystemExit(0 if report['passed'] else 1)
