# AGENTS.md — Castor

> **Canonical machine-readable operating contract** for any AI agent or human
> reading, editing, testing, or optimizing this repository.
>
> **Precedence:** `AGENTS.md` > `CLAUDE.md` / `GEMINI.md` (persona overlays) > `CONTRIBUTING.md` > `README.md`.
> If two files disagree, `AGENTS.md` wins and the discrepancy is a bug to file.

---

## 0. What This Repository Is

**Castor** is a local-first, **$0-token-cost** agent microkernel written in Rust that lets a cloud *Lead Architect* drive a locally-served peer programmer across any OpenAI-compatible serving engine (`llama.cpp`, `vLLM`, `SGLang`, `Ollama`, `LM Studio`, etc. — see [`docs/engine-profiles.md`](docs/engine-profiles.md)) inside a zero-trust sandbox.

```
Lead Architect (cloud)
        |  MCP over stdio - 3 consolidated tools (castor_coworker, castor_task, castor_server)
        v
castor  (Rust MCP server + Castor microkernel; :18021 wait, :18022 SSE proxy)
        v
Local Serving Engine (:18020)  ->  Pluggable Engine Profile
```

The crate root lives at `Cargo.toml` (dual-target lib + bin). A Node.js cross-platform shim (`bin/castor.js`) supports npm and `npx` client integration.

---

## 1. Agent Personas

| Persona | Runtime | Mandate | Hard Limits |
|---|---|---|---|
| **Lead Architect** | Cloud orchestrator | Architecture, task decomposition, manifest authorship, supervision, and final synthesis. Parallelize independent single-concern slices per workspace so they share one warm KV prefix. | **Never** bulk-read source into cloud context; **never** author code; no manual concurrency sizing (slot semaphore governs admission dynamically); **never** fall back to direct shell/file execution when the coworker is offline. |
| **Autonomous Execution Coworker** | Local peer inside the sandboxed Castor runtime | Hands-on execution: code exploration, file editing, git and Python via `bash`, structural AST surgery (`ast`), visual/PDF inspection (`read_file`), and web/academic research. Lean 8-tool core (`read_file`, `write_file`, `edit_file`, `list_dir`, `search_code`, `bash`, `web_search`, `web_fetch`), feature-gated `ast_search`/`ast_replace` and `paper_lookup`, plus the `finish` control-flow termination tool. | **Never** authors high-level plans or roadmaps; **never** escapes the sandbox root. |
| **Autonomous Optimizer (Evo)** | Local offline batch runner | Closed-loop mutation: propose -> evaluate -> select/revert against a deterministic fitness metric. | **Snapshot before mutating**; **revert on regression**; never edits files outside the candidate's snapshot list. |

**Honest attribution:** Never claim "we" or coworker collaboration unless a `castor_coworker` MCP call was genuinely dispatched and its output incorporated.

---

## 2. Setup, Build & Core Configuration

- **Prerequisites:** Rust >= 1.85 (Cargo, 2024 edition), Node.js >= 22 (optional, for `bin/castor.js`), Git (leave `core.autocrlf` at default — `.gitattributes` pins LF). **No GPU is required for the offline test gate.**
- **Build:** `cargo build`
- **Run MCP server:** `cargo run -- mcp` (or `./target/debug/castor mcp` or `node bin/castor.js mcp`)
- **Register with a client:** `./target/debug/castor install --client all` (or `node bin/castor.js install --client all`)

Configuration precedence: `CASTOR_*` env vars > `~/.castor/config.json` (`$CASTOR_STATE_DIR/config.json`) > built-in defaults. One key per setting (no aliases). Full inventory: `src/config.rs` and [`docs/engine-profiles.md`](docs/engine-profiles.md).

| Variable | Default | Meaning |
|---|---|---|
| `CASTOR_MODEL` | *(auto)* | Target model ID; auto-discovered from `/v1/models` when unset. |
| `CASTOR_BASE_URL` | `http://127.0.0.1:18020/v1` | OpenAI-compatible serving endpoint. |
| `CASTOR_API_KEY` | `local-token` | API key sent to the serving engine. |
| `CASTOR_ENGINE_TYPE` | *(unset)* | Serving backend: `llama.cpp` \| `vllm` \| `openai`. |
| `CASTOR_PROFILE` | *(unset)* | Named profile defined in `config.json` `profiles` (none compiled-in). |
| `CASTOR_MAX_CONTEXT` | *(auto)* | Max context tokens; auto-adopted from `max_model_len` when unset. |
| `CASTOR_LAUNCH_COMMAND` / `CASTOR_STOP_COMMAND` | *(profile)* | Optional custom engine launch/stop command overrides. |
| `CASTOR_BOOT_TIMEOUT_SECS` | `180` | Serving engine boot timeout in seconds before abort. |
| `CASTOR_PORT_ENGINE` / `STATUS` / `PROXY` | `18020` / `18021` / `18022` | Serving engine, long-poll wait, and loopback SSE proxy ports. |
| `CASTOR_STATE_DIR` | `~/.castor` | Root for task JSON, slot leases, session logs, and Evo lineage. |
| `CASTOR_MAX_CONCURRENT_TASKS` | `4` | Cross-process execution slots (token-budgeted against `max_context`). |
| `CASTOR_TOOL_PREFIX` | `castor` | Namespace prefix for MCP tools (`castor_coworker`, etc.). |
| `CASTOR_WORKSPACE_SKELETON` | `true` | Prepend deterministic workspace skeleton to the system prompt. |
| `CASTOR_TOOL_SPILL_BYTES` | `8192` | Max single tool output bytes before spilling to disk. |
| `CASTOR_DGI_REJECT_MARGIN` / `BYPASS_WARM_SESSION` | `0.0` / `true` | DGI admission log-odds reject margin and warm-session bypass. |
| `CASTOR_TURN_TIMEOUT_SECS` / `MAX_TURN_REASONING_TOKENS` | `600` / `16384` | Per-turn watchdog wall-clock and reasoning-token budgets (`0` disables). |
| `CASTOR_SEARXNG_URL` / `CASTOR_BRAVE_API_KEY` | `http://127.0.0.1:8888/` / *(unset)* | Primary SearXNG endpoint and Brave fallback for `web_search`. |
| `CASTOR_OPENALEX_EMAIL` / `CASTOR_OPENALEX_API_KEY` | *(unset)* | Optional polite-pool credentials for `paper_lookup`. |

---

## 3. Code Conventions, Architecture & Invariants

### 3.1 Formatting, Line Endings & Feature Flags
- **Rust:** 4-space indent, 2024 edition; enforced by `cargo fmt` and `cargo clippy --all-targets --all-features -- -D warnings`.
- **Line endings:** `.gitattributes` pins `* text=auto eol=lf`; `*.bat` and `*.cmd` are the only CRLF files.
- **Feature flags (`Cargo.toml`):** `default = ["mcp", "ast"]`; `mcp`, `evals`, `evo` (`= evals`), `ast` (structural AST surgery & syntax gate), `paper` (OpenAlex `paper_lookup`), `full` (`mcp` + `evo` + `evals` + `ast` + `paper`).

### 3.2 Structural AST Surgery & Edit Primitives
- Prefer `ast_search` and `ast_replace` (native `ast-grep` / tree-sitter) over regex for syntactic discovery and refactoring; `ast_replace` rolls back per file on syntax error.
- `edit_file`: 4-tier resilient matcher (exact, CRLF/LF normalized, trimmed, relative-indent) with uniqueness guard and line-ending preservation. Under `ast`, enforces a two-tier syntax gate: clean files must stay error-free; already-broken files cannot gain error nodes.
- `write_file`: atomic temp-file swap. Under `ast`, enforces a strict gate: zero `ERROR`/`UNDEFINED`/`MISSING` nodes regardless of prior state (non-code extensions and empty content pass).

### 3.3 Warm KV Prefix & Token-Budgeted Slot Semaphore
- `build_worker_system_prompt` (`src/mcp/worker.rs`) builds a byte-identical, workspace-deterministic warm KV prefix across all prompts and resumed sessions in the same workspace; prompt-specific CPG/PPR retrieved code attaches to the user turn (`build_user_turn_augmentation`).
- Concurrency is governed dynamically by the file-backed slot semaphore (`src/task/semaphore/`): `token_budget_admits` checks aggregate `per_slot_state_tokens + context_tokens` against `max_context`, FIFO-deferring on saturation and ranking KV-pool backpressure by `context_tokens` descending (`wait_backpressure`).
- Stall/timeout retries are zero-invalidation: preserve the message prefix and original reasoning effort (mutating `chat_template_kwargs` invalidates the KV cache) and append a nudge at the tail.

### 3.4 Present-State Truth (Rustdoc & Comments)
- Document current architecture, behavior, and invariants strictly in present-state truth.
- Never write chronological narratives ("Raised from X", "Formerly"), calendar dates, version snapshots, milestone/ticket tags (`M6a:`, `Issue #51:`), or lab diaries.

---

## 4. Invariant Boundaries

- **ALWAYS:** Run the verification gates before committing; keep paths relative or env-driven (no hardcoded drive letters, home dirs, or usernames); surface unadulterated errors (fail-fast, zero-masking); keep `CLAUDE.md` and `GEMINI.md` byte-identical; store temporary scripts/artifacts in `<workspace>/.scratch/`; keep `Cargo.toml` and `package.json` versions byte-identical before tagging; monitor CI/CD (`gh run watch`) to completion on release.
- **ASK FIRST:** Before weakening sandbox containment (`src/tools/sandbox/`), weakening shell scanner rules (`src/tools/shell/scanner.rs`), changing default external ports (`18020`/`18021`/`18022`), or pushing release tags.
- **NEVER:** Commit `.evo/`, `.scratch/`, or `*.log`; let file or shell ops escape the workspace root; dump raw binary bytes into text buffers (`BinaryFileError`; PDFs are parsed in-memory via `lopdf`); silently suppress errors or HTTP status codes; weaken, skip, or delete failing tests; interrupt, probe, or reboot a running serving engine during default tests; substitute cloud shell/file tools when the coworker is offline.

---

## 5. Verification Gates

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

**Zero Engine Interruption Invariant:** `cargo test --all-features` runs fully offline against deterministic mock/replay harnesses. Live GPU suites (`tests/llama_cpp.rs`, `tests/bonsai_deep_eval.rs`) are `#[ignore]`-gated and require `ALLOW_ENGINE_INTERRUPT=1`.

---

## 6. Security & Ground-Truth Hierarchy

- **Security boundaries:** (1) 5-layer zero-trust path containment (`src/tools/sandbox/`, regression suite `src/tools/sandbox/tests.rs`), (2) lexical shell command gate (`src/tools/shell/scanner.rs`), (3) `ast` in-memory syntax validation across 27 grammars. Report vulnerabilities privately per [`SECURITY.md`](SECURITY.md).
- **Ground-truth hierarchy:** (1) Active source code, `Cargo.toml`, and build artifacts; (2) `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `CONTRIBUTING.md`; (3) secondary docs and notes as non-authoritative reference.

---

## 7. Release & Packaging Contract

- **Triple-Lock Version Invariant:** `Cargo.toml` (`version = "X.Y.Z"`), `package.json` (`"version": "X.Y.Z"`), and git tag `vX.Y.Z` must remain byte-identical on every release.
- **npm OIDC Trusted Publishing:** `.github/workflows/release.yml` publishes with `npm publish --provenance` via GitHub OIDC (`id-token: write`, Sigstore/Rekor) with no static `NPM_TOKEN`. Registry versions are immutable (`403` on collision).
- **Atomic release checklist:** Run `cargo test --all-features` and `cargo clippy --all-targets --all-features -- -D warnings` -> bump `Cargo.toml` and `package.json` -> commit and push `main` -> tag `vX.Y.Z` and push tag -> run `gh run watch` to completion (inspect `gh run view --log-failed` on failure).
