#!/usr/bin/env bash
# Check that the local vLLM endpoint configured in .env is up, serves the
# expected model, and accepts a real chat request with enable_thinking off.
#
# Usage: scripts/check_vllm.sh [--env-file PATH]
#
# Exit status: 0 if every check passes, 1 otherwise. Prints one pass/fail
# line per check plus a final "OK" / "FAIL" summary line.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE=".env"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "FAIL: env file not found: $ENV_FILE" >&2
  exit 1
fi

# Read KEY=value lines without exporting into the caller's shell.
OPENAI_BASE_URL=$(grep -E '^OPENAI_BASE_URL=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)
OPENAI_MODEL=$(grep -E '^OPENAI_MODEL=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)
OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)

status=0

echo "env file: $ENV_FILE"
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL:-<empty>}"
echo "OPENAI_MODEL=${OPENAI_MODEL:-<empty>}"

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "FAIL: OPENAI_BASE_URL is empty in $ENV_FILE"
  exit 1
fi
if [[ -z "${OPENAI_MODEL:-}" ]]; then
  echo "FAIL: OPENAI_MODEL is empty in $ENV_FILE"
  exit 1
fi

# 1. /health
if curl -sS --max-time 10 -o /dev/null -w '' "${OPENAI_BASE_URL%/v1}/health"; then
  echo "PASS: /health reachable"
else
  echo "FAIL: /health not reachable at ${OPENAI_BASE_URL%/v1}/health"
  status=1
fi

# 2. /v1/models lists OPENAI_MODEL
models_json=$(curl -sS --max-time 10 -H "Authorization: Bearer ${OPENAI_API_KEY:-none}" "${OPENAI_BASE_URL}/models" || true)
if echo "$models_json" | python3 -c "
import json, sys
model = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception as exc:
    print('could not parse /v1/models response:', exc, file=sys.stderr)
    sys.exit(1)
ids = [row.get('id') for row in data.get('data', [])]
sys.exit(0 if model in ids else 1)
" "$OPENAI_MODEL"; then
  echo "PASS: /v1/models lists $OPENAI_MODEL"
else
  echo "FAIL: /v1/models does not list $OPENAI_MODEL"
  echo "  served models: $(echo "$models_json" | python3 -c 'import json,sys;d=json.load(sys.stdin);print([r.get("id") for r in d.get("data",[])])' 2>/dev/null || echo "$models_json")"
  status=1
fi

# 3. one tiny chat request with chat_template_kwargs.enable_thinking=false
#    (the hosted OpenAI API rejects that key and max_tokens, so there it is
#    max_completion_tokens + reasoning_effort=none, as api_style="openai" sends)
request_body=$(python3 -c "
import json, sys
body = {'model': sys.argv[1], 'messages': [{'role': 'user', 'content': 'Reply with exactly one word: pong'}]}
if 'api.openai.com' in sys.argv[2]:
    body.update({'max_completion_tokens': 16, 'reasoning_effort': 'none'})
else:
    body.update({'max_tokens': 8, 'temperature': 0.0, 'chat_template_kwargs': {'enable_thinking': False}})
print(json.dumps(body))
" "$OPENAI_MODEL" "$OPENAI_BASE_URL")

chat_response=$(curl -sS --max-time 60 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-none}" \
  -d "$request_body" \
  "${OPENAI_BASE_URL}/chat/completions" || true)

if echo "$chat_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except Exception as exc:
    print('could not parse chat response:', exc, file=sys.stderr)
    sys.exit(1)
choices = data.get('choices') or []
if not choices:
    print('no choices in response:', data, file=sys.stderr)
    sys.exit(1)
content = (choices[0].get('message') or {}).get('content')
if not content:
    print('empty content in response:', data, file=sys.stderr)
    sys.exit(1)
print('  reply:', content.strip()[:80])
"; then
  echo "PASS: chat completion returned content"
else
  echo "FAIL: chat completion did not return usable content"
  status=1
fi

if [[ $status -eq 0 ]]; then
  echo "OK: vLLM endpoint is healthy and serving $OPENAI_MODEL"
else
  echo "FAIL: one or more vLLM checks failed"
fi
exit $status
