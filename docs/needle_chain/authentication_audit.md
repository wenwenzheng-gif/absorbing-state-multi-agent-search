# Existing provider setup audit

Completed in the user's requested order, 2026-11-07. No secret value was
printed, copied, committed or modified.

1. Current repository scripts use `--env-file .env`, `load_env_file`,
   `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and the OpenAI-compatible `LLMBackend`.
   Historical configs record the family alias `gpt-5.4-nano`.
2. `git worktree list` identifies one checkout of the canonical repository.
   Two prior scientific checkouts belong to `wenwenzheng-gif/absorb-state-new`.
   None of these checkouts contains a real `.env` file. They are separate
   repositories, with the earlier experiment infrastructure retained in the
   latter.
3. Prior experiment branches/configurations route live work through GitHub
   Actions. Their documentation references the original standard `.env`
   local route and bounded hosted campaigns; their provider wrapper calls
   `https://api.openai.com/v1/chat/completions`.
4. The earlier workflows reference `secrets.OPENAI_API_KEY`, use
   `ubuntu-latest`, and declare no separate GitHub environment. GitHub metadata
   confirms the existing secret name in `absorb-state-new`, last updated
   2026-11-01. Only its name and update timestamp were inspected. The canonical
   public release has no configured Actions secrets.
5. Existing `.zprofile`, `.zshrc`, `.profile` and system zsh configuration
   contain no OpenAI/provider variable bindings or dotenv loader. Existing
   experiment loaders remain the standard `.env` reader and Actions injection.
6. Prior successful live run
   [42325666972](https://github.com/wenwenzheng-gif/absorb-state-new/actions/runs/42325666972)
   launches `scripts/run_phase_confirmation_cell.py`, binds `OPENAI_API_KEY`,
   and records `gpt-5.4-nano-2026-03-17` provider responses. Saved prior pilot
   inventories independently contain that snapshot identifier. Only filtered
   command/model references were emitted from the run log.

Resolution: reuse the existing hosted Actions environment and secret in place.
The dedicated runner branch uses the exact experimental branch name and
checks out the canonical frozen commit. No secrets are migrated; code and
reviewed results remain on the canonical experimental branch. The full run
is excluded from the hosted entry point.
