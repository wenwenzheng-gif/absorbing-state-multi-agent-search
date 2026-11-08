#!/usr/bin/env python3
"""Remove ONLY GitHub Actions artifacts from the four proven incorrect Nano runs."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO="wenwenzheng-gif/absorbing-state-multi-agent-search"
WRONG_RUNS=(42501273757,42501522916,42501667950,42502083463)
WRONG_BRANCH="experiment/gpt54-nano-3tasks-6seeds-parallel"
EXPECTED={
    42501273757:{"GPT-5.4 Nano exact-model preflight"},
    42501522916:{"GPT-5.4 Nano three tasks six seeds"},
    42501667950:{"GPT-5.4 Nano three tasks six seeds"},
    42502083463:{"GPT-5.4 Nano three tasks six seeds"},
}
VALID_NAME=re.compile(r"^(?:nano-(?:model-preflight|smoke-(?:synthetic|tcas|ode)|(?:synthetic|tcas|ode)-seed-[0-5])|gpt54-nano-preflight-\\d+)$")
OUT=Path("outputs/gpt-5.4-nano-t1-none-new3/incorrect_run_cleanup.json")

def gh(path,delete=False):
    cmd=["gh","api"]
    if delete: cmd.extend(["-X","DELETE"])
    cmd.append(f"repos/{REPO}/{path}")
    c=subprocess.run(cmd,capture_output=True,text=True)
    if c.returncode:
        raise RuntimeError("GitHub Actions API rejected "+("DELETE" if delete else "GET")+" request")
    return json.loads(c.stdout) if c.stdout.strip() else {}

def main():
    if not os.getenv("GH_TOKEN"):
        raise RuntimeError("GitHub token was not injected")
    deleted=[]
    report={"wrong_run_ids":list(WRONG_RUNS),"deleted_artifact_ids":deleted,
            "other_artifacts_left_untouched":0,"status":"IN_PROGRESS"}
    try:
        for rid in WRONG_RUNS:
            run=gh(f"actions/runs/{rid}")
            if run.get("head_branch")!=WRONG_BRANCH or run.get("name") not in EXPECTED[rid]:
                raise RuntimeError(f"Refusing deletion: run {rid} provenance mismatched")
            artifacts=gh(f"actions/runs/{rid}/artifacts?per_page=100").get("artifacts",[])
            for a in artifacts:
                name=a["name"]
                if not VALID_NAME.fullmatch(name):
                    report["other_artifacts_left_untouched"]+=1
                    continue
                gh(f"actions/artifacts/{a['id']}",delete=True)
                deleted.append({"run_id":rid,"artifact_id":a["id"],"name":name})
        report["status"]="COMPLETE"
    except Exception as e:
        report.update(status="FAILED",error_type=type(e).__name__)
        raise
    finally:
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Deleted exactly",len(deleted),"artifacts from the four incorrect/cancelled run IDs.")
    print("No valid Nano t1, Luna or Qwen outputs were targeted.")
    return 0

if __name__=="__main__":
    sys.exit(main())
