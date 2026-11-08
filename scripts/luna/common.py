"""Pinned scientific protocol and isolated corrected-batch paths."""
import copy, hashlib, json, os, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/gpt-6-luna-t1-none-new3'
REF = json.loads((ROOT / 'scripts/luna/reference_protocol.json').read_text())
SEEDS = (7, 8, 9)
TASKS = ('synthetic', 'tcas', 'physics')
POLICIES = ('C0', 'C1', 'I1')
SETTINGS = dict(model='gpt-6-luna', temperature=1.0, reasoning_effort='none', enable_thinking=False, thinking='disabled')
DATASET = 'TobyYang7/Branching-Search-runs'
ENV_DEFAULT = '/data4/toby/codex_api/.env'

def write(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)

def rows(path):
    if not pathlib.Path(path).exists():
        return []
    with pathlib.Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def literal_seed(task, index):
    assert index in SEEDS
    return max(REF['used_task_seeds'][task]) + index - max(REF['valid_seed_indices'])

def cell_dir(task, index, policy):
    tag = f'gpt-6-luna_seed{index}'
    return OUT / 'runs/agents' / tag / f'{tag}_{task}_{policy}'

def config(task, index, policy, smoke=False):
    c = copy.deepcopy(REF['cells'][f'{task}_{policy}'])
    c.update(name=f'gpt-6-luna_seed{index}_{task}_{policy}', task_seeds=[literal_seed(task,index)], output_root=str(cell_dir(task,index,policy).parent))
    c['policy']['llm'].update(model='gpt-6-luna', api_style='openai', reasoning_effort='none', temperature=1.0, enable_thinking=False, transport_retries=8, max_wall_seconds=None)
    if smoke:
        c.update(name=c['name']+'_smoke', agents=2, initial_width=1, degrees=[1])
    return c

def verify_sources():
    for name, digest in REF['protocol_source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Scientific source changed: {name}')

def completed(path):
    return {r['episode_id'] for r in rows(path / 'episodes.jsonl') if r.get('status') == 'ok'}

def expected(task):
    return sum(len(REF['cells'][f'{task}_{policy}']['degrees']) for policy in POLICIES)

def batch():
    return {**SETTINGS, 'batch':'gpt-6-luna-t1-none-new3', 'seed_ids':list(SEEDS), 'task_seed_ids':{t:{str(s):literal_seed(t,s) for s in SEEDS} for t in TASKS}, 'valid_seed_ids':REF['valid_seed_indices'], 'valid_task_seeds':REF['used_task_seeds'], 'policies':list(POLICIES), 'expected_episodes':{t:3*expected(t) for t in TASKS}, 'total_expected_episodes':sum(3*expected(t) for t in TASKS), 'reference_archive_sha256':REF['source_archive_sha256'], 'cancelled_output_root_absent':not (ROOT/'outputs/gpt-6-luna').exists(), 'created_at':time.time()}
