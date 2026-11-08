#!/usr/bin/env python3
"""Prepare isolated configs; never launch or overwrite a seed."""
import copy
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TASKS = {'synthetic': ('llm_toy', 9), 'tcas': ('llm_tcas', 12), 'ode': ('llm_physics', 11)}

def main():
    records = []
    for size, port in [('4B', 8114), ('9B', 8119)]:
        model = f'qwen3.5-{size.lower()}'
        for task, (source, target) in TASKS.items():
            reference = ROOT / f'configs/agents/{source}.json'
            original = json.loads(reference.read_text())
            assert len(original['degrees']) * original['repeats'] == target
            for seed in range(6):
                c = copy.deepcopy(original)
                c.update(name=f'{model}_{task}_seed_{seed}', task_seeds=[seed], output_root=f'outputs/{model}/{task}')
                c['policy']['llm'].update(model=f'Qwen/Qwen3.5-{size}', base_url=f'http://127.0.0.1:{port}/v1')
                # All scientific fields and native Qwen transport defaults are retained.
                path = ROOT / f'configs/qwen35-six-seed/{model}/{task}/seed_{seed}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                encoded = json.dumps(c, indent=2) + '\n'
                if path.exists() and path.read_text() != encoded:
                    raise RuntimeError(f'Refusing to replace {path}')
                path.write_text(encoded)
                records.append(dict(model=model, task=task, seed=seed, target_episodes=target,
                                    status='BLOCKED_GPU_CAPACITY', completed_episodes=0,
                                    config=str(path.relative_to(ROOT)), reference=str(reference.relative_to(ROOT)),
                                    reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest()))
    assert len(records) == 36 and sum(r['target_episodes'] for r in records) == 384
    (ROOT / 'results/qwen35/prepared_runs.json').write_text(json.dumps(records, indent=2)+'\n')
    print('Prepared 36 configs / 384 target episodes. No experiments launched.')

if __name__ == '__main__':
    main()
