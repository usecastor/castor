//! Background worker process implementation for Castor coworker tasks.
//!
//! When a task is dispatched, the MCP server spawns a detached worker process:
//!
//! ```bash
//! castor __worker <path-to-job-spec.json>
//! ```
//!
//! **Prompt composition (KV prefix stability).** The system prompt is a pure
//! function of the *workspace* (static persona + `[Workspace]` root + the
//! deterministic `[Workspace Skeleton]` + the active-skills index) and is
//! **byte-identical across every prompt and every resumed session** in the same
//! workspace — that is the warm KV prefix that makes resumed continuations
//! cache-hit from token 0. The only variable context (the CPG/PPR
//! retrieved-code block) is built separately and attached to the **user
//! message**, never the system prompt, so a new prompt never perturbs the
//! cached prefix.
//!
//! The worker:
//! 1. Loads the [`JobSpec`].
//! 2. Runs the DGI gatekeeper classification; a rejected dispatch fails
//!    fast before any lease is held.
//! 3. Acquires a token-budgeted execution slot from [`TaskSemaphore`].
//! 4. Transitions the task status to `Executing` in [`TaskRegistry`] and records its PID.
//! 5. Sets up the session [`EventLogger`], indexes relevant skills, and prepares [`CompositeExecutor`].
//! 6. Drives [`crate::runner::run_session`].
//! 7. Mirrors terminal status (`Completed` or `Failed`) to [`TaskRegistry`].
//! 8. Releases the slot lease on both success and failure.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::Config;
use crate::engine::EngineClient;
use crate::runner::events::EventLogger;
use crate::runner::{self, ChatEngine};
use crate::skills;
use crate::state::StateDir;
use crate::task::registry::{TaskRegistry, TaskStatus};
use crate::task::semaphore::TaskSemaphore;
use crate::tools::CompositeExecutor;
use crate::tools::extensions::ExtensionBridge;

/// A serialized job specification written by the MCP spawner for the detached worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSpec {
    pub task_id: String,
    pub prompt: String,
    pub cwd: String,
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns_budget: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

fn now_epoch_ms() -> u64 {
    crate::platform::now_epoch_ms()
}

/// Probe the workspace repository's HEAD commit and dirty state without blocking
/// worker launch. Returns `(None, None)` immediately when `.git` is absent, or
/// runs `rev-parse HEAD` and `status --porcelain --untracked-files=no`
/// concurrently with a 1500 ms wall-clock budget.
pub async fn probe_git_provenance(
    workspace_root: &std::path::Path,
) -> (Option<String>, Option<bool>) {
    if !workspace_root.join(".git").exists() {
        return (None, None);
    }

    let head_fut = async {
        tokio::process::Command::new("git")
            .arg("-C")
            .arg(workspace_root)
            .args(["rev-parse", "HEAD"])
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .output()
            .await
    };
    let status_fut = async {
        tokio::process::Command::new("git")
            .arg("-C")
            .arg(workspace_root)
            .args(["status", "--porcelain", "--untracked-files=no"])
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .output()
            .await
    };

    let Ok((head_res, status_res)) =
        tokio::time::timeout(std::time::Duration::from_millis(1500), async {
            tokio::join!(head_fut, status_fut)
        })
        .await
    else {
        return (None, None);
    };

    let base_commit = match head_res {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if s.is_empty() { None } else { Some(s) }
        }
        _ => None,
    };
    let git_dirty = match status_res {
        Ok(out) if out.status.success() => {
            Some(!String::from_utf8_lossy(&out.stdout).trim().is_empty())
        }
        _ => None,
    };

    (base_commit, git_dirty)
}

/// Canonicalize the workspace root by walking up from the provided path
/// to find the repository boundary (.git, Cargo.toml, package.json).
/// Falls back to the provided path if no root marker is found.
pub fn find_workspace_root(p: &std::path::Path) -> std::path::PathBuf {
    let mut cur = if p.is_file() {
        p.parent().unwrap_or(p).to_path_buf()
    } else {
        p.to_path_buf()
    };
    loop {
        if cur.join(".git").exists()
            || cur.join("Cargo.toml").exists()
            || cur.join("package.json").exists()
        {
            return cur;
        }
        if let Some(parent) = cur.parent() {
            if parent == cur {
                break;
            }
            cur = parent.to_path_buf();
        } else {
            break;
        }
    }
    p.to_path_buf()
}

/// Build the worker system prompt in strict low-churn → high-churn KV prefix
/// order so the Turn-1 prefix is byte-for-byte stable across sessions in the
/// same workspace:
///   (a) Static persona, invariants, and tool-routing hierarchy (most stable)
///   (b) `[Workspace]` root, then the deterministic `[Workspace Skeleton]`
///   (c) The active-skills index (highest churn — skills can vary per dispatch)
///
/// **Byte-identical, workspace-deterministic.** This is a pure function of the
/// workspace's stable shape and the build — *not* of the dispatch prompt or any
/// per-run context. The CPG/PPR retrieved-code block is **deliberately excluded
/// here**; it attaches to the user message (see
/// [`build_user_turn_augmentation`]) so the system prompt stays a 100%
/// deterministic warm KV prefix that resumed sessions reuse byte-for-byte.
///
/// Extracted from [`run_job_with_engine`] as a pure function of its inputs so
/// the prefix ordering can be regression-tested without spawning a job.
pub fn build_worker_system_prompt(
    workspace_root: &std::path::Path,
    skeleton_enabled: bool,
    skeleton: &str,
    skills_index: &str,
) -> String {
    let mut system_prompt = String::from(
        "You are Castor, an autonomous senior peer engineer pairing with the Lead Architect.\n\n\
Peer Engineering Invariants:\n\
- Senior Peer Mandate: Reason, decide, explore, and execute autonomously. Challenge flawed assumptions, cite conflicting code coordinates, and propose cleaner architectural alternatives rather than looping on broken premises.\n\
- Ground-Truth Hierarchy: Active source code and compiler/test diagnostics are ground truth. Documentation and notes are reference ledgers. Write present-state truth only—never historical changelogs, ticket tags, or defensive lore.\n\
- Tool Routing Hierarchy (prefer dedicated native tools; use `bash` for Git, Python, and builds):\n\
",
    );

    // Build the tool-routing bullets in a fixed, feature-aware order so the
    // Turn-1 prefix stays byte-for-byte stable for a given build. The default
    // surface is a lean 8-tool set (file I/O, search, shell, web); `ast` adds
    // structural AST surgery and `paper` adds OpenAlex `paper_lookup`. Git and
    // Python have no dedicated tool anymore and are driven through `bash`.
    let mut routing: Vec<&'static str> = Vec::new();
    routing.push(if cfg!(feature = "ast") {
        "  * Code Discovery & Refactoring: Use `ast_search` and `ast_replace` for structural syntax queries and multi-file rewrites (prefer over regex); use `search_code`, `list_dir`, `read_file`, `edit_file`, and `write_file` for file operations.\n"
    } else {
        "  * File Operations & Code Discovery: Use `search_code` for literal pattern search, `read_file` for exact regions, and `list_dir`, `edit_file`, and `write_file` for file authoring.\n"
    });
    routing.push(
        "  * Version Control: Use `bash` for git operations (`git status`, `git diff`, `git log`, `git commit`).\n",
    );
    routing.push(
        "  * Data Inspection & Calculation: Use `bash` for Python execution (`python3`) and other computation/data tasks.\n",
    );
    routing.push(
        "  * Visual & Document Inspection: Use `read_file` directly on local images (`.png`, `.jpg`, `.webp`, `.gif`, `.bmp`) and PDFs.\n",
    );
    routing.push(if cfg!(feature = "paper") {
        "  * Research: Use `paper_lookup` (OpenAlex) for peer-reviewed literature, plus `web_search` (SearXNG/Brave) and `web_fetch` (HTML/PDF → Markdown).\n"
    } else {
        "  * Research: Use `web_search` (SearXNG/Brave) and `web_fetch` (HTML/PDF → Markdown).\n"
    });
    routing.push(
        "  * Shell (`bash`): Also use for builds (`cargo build`), test/lint gates (`cargo test`, `cargo clippy`), and process/CI execution.\n",
    );
    let routing = routing.concat();
    system_prompt.push_str(&routing);
    system_prompt.push_str(
        "- Verification Discipline: Never claim completion or close a task without running the verification gate and confirming clean output.\n\
- High-Throughput Batching: Always emit ALL independent tool calls in parallel within a single turn. When exploring, diagnosing, or refactoring, batch your searches and reads together (e.g. issue multiple `read_file` or `search_code` calls in one turn). Never ping-pong single reads or searches sequentially when they can be dispatched concurrently.\n\
- Security & Authority: Tool refusals tagged `Access denied`, `Execution blocked`, or `binary file` are binding policy decisions, not transient failures or permission requests. Never route around one \u{2014} no `bash`/`cat`/`cp`/`mv`/`tee` substitute, alternative path, symlink, or privilege escalation. If the blocked file or operation is essential to the task, stop that line of work and report the constraint in your final report, naming the exact target and the allowed root or permission that would unblock it, so the Lead Architect / user can adjust the workspace policy deliberately. (If the denial was your own typo \u{2014} a malformed or mistyped in-workspace path \u{2014} correct it and retry; that is fixing your error, not bypassing policy.)",
    );

    system_prompt.push_str(&format!(
        "\n\n[Workspace]\nRoot: {}",
        workspace_root.display()
    ));
    if skeleton_enabled && !skeleton.is_empty() {
        system_prompt.push('\n');
        system_prompt.push_str(skeleton);
    }
    if !skills_index.is_empty() {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(skills_index);
    }
    system_prompt
}

/// Build the **variable** Turn-1 augmentation block that is attached to the
/// **user message** (never the system prompt), so the system prompt stays a
/// byte-identical warm KV prefix.
///
/// The block contains the CPG/PPR retrieved-code context — prompt-specific, so
/// it must never land in the system prompt. It is computed only when the
/// workspace-skeleton prefix is enabled; `build_prompt_retrieved_context`
/// yields an empty string when the `ast` feature is disabled, so this is a
/// single call path regardless of feature configuration.
///
/// Returns an empty string when the CPG context is absent, in which case the
/// caller uses the raw dispatch prompt untouched (preserving the fresh
/// `[system, user]` byte shape for zero-invalidation retries).
///
/// The DGI gatekeeper is strictly binary ADMIT/REJECT and never injects any
/// text into the worker prompt: a rejected dispatch fails fast before the
/// session starts, and an admitted dispatch proceeds with the raw prompt.
pub fn build_user_turn_augmentation(
    workspace_root: &std::path::Path,
    skeleton_enabled: bool,
    prompt: &str,
) -> String {
    let mut aug = String::new();

    // CPG/PPR retrieved-code context (prompt-specific).
    if skeleton_enabled {
        let ctx = runner::skeleton::build_prompt_retrieved_context(workspace_root, prompt);
        if !ctx.is_empty() {
            aug.push_str(&ctx);
        }
    }

    aug
}

/// Execute a job specification against the real local engine.
pub async fn run_job(spec: &JobSpec, state: &StateDir, config: &Config) -> Result<String, String> {
    let registry = TaskRegistry::new(state);
    let now = now_epoch_ms();
    let _ = registry
        .update(&spec.task_id, |r| {
            r.pid = Some(std::process::id());
            r.heartbeat = now;
        })
        .await;

    let lc = crate::engine::EngineLifecycle::new(config, state);
    if let Err(e) = lc.ensure_running().await {
        let err_msg = format!("Engine auto-boot failed: {e}");
        let _ = registry
            .transition(&spec.task_id, TaskStatus::Failed, Some(err_msg.clone()))
            .await;
        return Err(err_msg);
    }

    // Resolve the effective configuration dynamically: when `model`,
    // `max_context`, or `engine_type` are not pinned, adopt the values
    // discovered from the live engine profile.
    let mut effective_config = config.clone();
    let budget = if let Ok(profile) = lc.profile().await {
        if effective_config.model.is_none() {
            effective_config.model = Some(profile.model_id.clone());
        }
        if effective_config.max_context.is_none() {
            effective_config.max_context = profile.max_context;
        }
        if effective_config.engine_type.is_none() {
            effective_config.engine_type = Some(profile.backend.as_str().to_string());
        }
        let budget = profile.kv_pool_budget(&effective_config);
        if budget.pool_capacity_tokens > 0
            && effective_config
                .max_context
                .is_some_and(|mc| mc > budget.pool_capacity_tokens)
        {
            effective_config.max_context = Some(budget.pool_capacity_tokens);
        }
        Some(budget)
    } else {
        None
    };

    let engine = match EngineClient::from_config(&effective_config) {
        Ok(eng) => eng,
        Err(e) => {
            let err_msg = e.to_string();
            let _ = registry
                .transition(&spec.task_id, TaskStatus::Failed, Some(err_msg.clone()))
                .await;
            return Err(err_msg);
        }
    };
    run_job_with_engine_budgeted(spec, state, &effective_config, &engine, budget).await
}

/// Execute a job specification with a provided [`ChatEngine`] (supports mock engines in tests).
pub async fn run_job_with_engine(
    spec: &JobSpec,
    state: &StateDir,
    config: &Config,
    engine: &dyn ChatEngine,
) -> Result<String, String> {
    run_job_with_engine_budgeted(spec, state, config, engine, None).await
}

/// Execute a job specification with a provided [`ChatEngine`] and an optional
/// physical KV-pool budget from [`crate::engine::EngineProfile::kv_pool_budget`].
pub async fn run_job_with_engine_budgeted(
    spec: &JobSpec,
    state: &StateDir,
    config: &Config,
    engine: &dyn ChatEngine,
    budget: Option<crate::engine::backend::KvPoolBudget>,
) -> Result<String, String> {
    let registry = TaskRegistry::new(state);
    let max_slots = config.max_concurrent_tasks as usize;
    let max_ctx = config.max_context.unwrap_or(0);

    // 1. DGI gatekeeper: classify the dispatch BEFORE committing to a slot
    //    lease. The gate is strictly binary (ADMIT / REJECT): a rejected or
    //    failed probe never holds a lease and fails fast with the
    //    unadulterated error (no heuristic fallback), and an admitted dispatch
    //    proceeds with the raw prompt — the DGI injects no advisory text into
    //    the worker prompt.
    let mut dgi_telemetry: Option<crate::mcp::dgi::DgiEvaluation> = None;
    if config.base_url.is_some() && config.model.is_some() {
        let dgi = if crate::mcp::dgi::warm_session_bypass(config, state, &spec.session_id) {
            tracing::info!(
                session_id = %spec.session_id,
                "[dgi] warm-session bypass: admitting without probe"
            );
            crate::mcp::dgi::DgiEvaluation {
                verdict: crate::mcp::dgi::DgiVerdict::Admit,
                margin: None,
                source: "warm_session_bypass",
            }
        } else {
            let probe_timeout_secs = std::env::var("CASTOR_DGI_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(30);
            let http = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(probe_timeout_secs))
                .build()
                .unwrap_or_default();
            match crate::mcp::dgi::evaluate_configured_model_probe_with_metrics(
                config,
                &spec.prompt,
                &http,
            )
            .await
            {
                Ok(eval) => {
                    tracing::info!(
                        session_id = %spec.session_id,
                        verdict = if eval.verdict.is_reject() { "REJECT" } else { "ADMIT" },
                        margin = ?eval.margin,
                        reject_margin = config.dgi_reject_margin,
                        "[dgi] probe completed"
                    );
                    eval
                }
                Err(err) => {
                    let err_msg = format!("DgiProbeError: {err}");
                    let _ = registry
                        .transition(&spec.task_id, TaskStatus::Failed, Some(err_msg.clone()))
                        .await;
                    return Err(err_msg);
                }
            }
        };

        if let crate::mcp::dgi::DgiVerdict::Reject(sigs) = &dgi.verdict {
            let body = sigs
                .iter()
                .map(|s| format!("  - {s}"))
                .collect::<Vec<_>>()
                .join("\n");
            let reject_msg = format!(
                "DecompositionGateRejected: dispatch rejected by model decomposition gatekeeper:\n{body}\n\
                 Decompose into a single-concern slice (one subsystem, one verification gate) and re-dispatch."
            );
            let _ = registry
                .transition(
                    &spec.task_id,
                    TaskStatus::RejectedDecomposition,
                    Some(reject_msg.clone()),
                )
                .await;
            return Err(reject_msg);
        }
        dgi_telemetry = Some(dgi);
    }

    // 2. Build the system prompt BEFORE committing to a slot lease, so the
    //    slot's static per-lease token reserve can be computed from the
    //    pre-built prompt's footprint (step 3). This only reads the
    //    filesystem (workspace root, skills, skeleton) — no engine call and
    //    no lease is taken, so it never blocks dispatch.
    let host_raw_path = crate::platform::to_host_path(&spec.cwd);
    let workspace_root = find_workspace_root(&host_raw_path);

    // Discover and index skills (the highest-churn part of the prefix).
    let skill_dirs = [
        workspace_root.join(".agents").join("skills"),
        workspace_root.join("skills"),
        state.root().join("skills"),
    ];
    let all_skills = skills::load_skills(&skill_dirs);
    let active_skills: Vec<_> = if let Some(ref req) = spec.skills {
        all_skills
            .into_iter()
            .filter(|s| req.contains(&s.name))
            .collect()
    } else {
        all_skills
    };
    let skills_index = skills::render_index(&active_skills);

    // Build the system prompt in strict low-churn -> high-churn KV prefix
    // order (delegated to the pure [`build_worker_system_prompt`] helper so the
    // ordering is unit-testable). The skeleton is the bare, prompt-agnostic
    // `[Workspace Skeleton]` — a byte-identical, deterministic warm KV prefix
    // that is a pure function of the workspace's stable top-level shape and the
    // build, so it is identical across every prompt and every resumed session.
    // Prompt-specific context (CPG/PPR retrieval) is NOT part of the system
    // prompt; it attaches to the user message below.
    let skeleton = if config.workspace_skeleton_prefix {
        runner::skeleton::build_workspace_skeleton(&workspace_root)
    } else {
        String::new()
    };
    let system_prompt = build_worker_system_prompt(
        &workspace_root,
        config.workspace_skeleton_prefix,
        &skeleton,
        &skills_index,
    );

    // Build the variable Turn-1 augmentation (CPG/PPR retrieved-code context).
    // It attaches to the user message — never the system prompt — so the prompt
    // above stays a 100% deterministic warm KV prefix (byte-identical across
    // prompts and resumed sessions).
    let user_aug = build_user_turn_augmentation(
        &workspace_root,
        config.workspace_skeleton_prefix,
        &spec.prompt,
    );
    // The user message: the raw prompt when there is no augmentation (preserves
    // the fresh `[system, user]` byte shape for zero-invalidation retries), or
    // the raw prompt preceded by the variable context block (with exactly one
    // blank line separating them) otherwise.
    let user_prompt = if user_aug.is_empty() {
        spec.prompt.clone()
    } else {
        let mut s = String::with_capacity(user_aug.len() + spec.prompt.len() + 2);
        s.push_str(&user_aug);
        if !user_aug.ends_with('\n') {
            s.push('\n');
        }
        s.push('\n');
        s.push_str(&spec.prompt);
        s
    };

    // 3. Acquire execution slot (token-budgeted). The pool's total budget is
    //    `budget.pool_capacity_tokens` (or `max_ctx` when no engine budget is
    //    provided), plus `per_slot_state_tokens` per active slot. Include
    //    built-in tool schemas in the pre-slot `/tokenize` count so Turn-1
    //    admission and resume watermarking match `run_session`.
    let pool_budget = budget.map(|b| b.pool_capacity_tokens).unwrap_or(max_ctx);
    let per_slot_state_tokens = budget.map(|b| b.per_slot_state_tokens).unwrap_or(0);
    let sem = TaskSemaphore::new(state, max_slots)
        .with_token_budget(pool_budget)
        .with_per_slot_state_tokens(per_slot_state_tokens);
    let builtin_tools = crate::tools::builtin_tool_schemas();

    let initial_tokens = {
        let mut init_msgs = vec![runner::messages::msg("system", &system_prompt)];
        if let Some(history) = runner::context::load_messages(state, &spec.session_id)
            && !history.is_empty()
        {
            let mut candidate = Vec::with_capacity(1 + history.len() + 1);
            candidate.push(runner::messages::msg("system", &system_prompt));
            candidate.extend(history.iter().cloned());
            candidate.push(runner::messages::msg("user", &user_prompt));
            let fits_watermark = if max_ctx > 0 {
                if let Some(exact) = engine.count_tokens(&candidate, &builtin_tools).await {
                    exact <= runner::context::watermark_tokens(max_ctx as usize)
                } else {
                    runner::context::should_resume(&history, &user_prompt, max_ctx as usize)
                }
            } else {
                true
            };
            if fits_watermark {
                init_msgs.extend(history);
            }
        }
        init_msgs.push(runner::messages::msg("user", &user_prompt));
        if let Some(exact) = engine.count_tokens(&init_msgs, &builtin_tools).await {
            exact.min(u32::MAX as usize) as u32
        } else {
            let has_history = init_msgs.len() > 2;
            let resumed_tokens = if has_history {
                runner::context::resumed_history_tokens(state, &spec.session_id, Some(max_ctx))
            } else {
                0
            };
            let prompt_tokens = runner::context::uncounted_tail_tokens(&[runner::messages::msg(
                "user",
                &user_prompt,
            )]) as u32;
            prompt_tokens.saturating_add(resumed_tokens)
        }
    };
    let lease = sem.acquire_with_tokens(&spec.task_id, initial_tokens).await;
    let shared_lease = std::sync::Arc::new(std::sync::Mutex::new(lease));

    // 4. Transition status to Executing and stamp worker PID.
    let now = now_epoch_ms();
    let _ = registry
        .update(&spec.task_id, |r| {
            r.status = TaskStatus::Executing;
            r.pid = Some(std::process::id());
            r.started_at = Some(now);
            r.heartbeat = now;
        })
        .await;

    // 5. Initialize EventLogger and record session start (with git provenance).
    let logger = EventLogger::new(state, &spec.session_id);
    let (base_commit, git_dirty) = probe_git_provenance(&workspace_root).await;
    // The effective tier is recorded on session_start (null when unset) so
    // the session ledger is self-explanatory: null = "server default applied".
    let _ = logger.append(json!({
        "type": "session_start",
        "sessionId": spec.session_id,
        "taskId": spec.task_id,
        "cwd": spec.cwd,
        "prompt": spec.prompt,
        "reasoning_effort": spec.reasoning_effort,
        "base_commit": base_commit,
        "git_dirty": git_dirty,
    }));
    if let Some(ref dgi) = dgi_telemetry {
        let _ = logger.append(json!({
            "type": "dgi",
            "verdict": if dgi.verdict.is_reject() { "REJECT" } else { "ADMIT" },
            "margin": dgi.margin,
            "reject_margin": config.dgi_reject_margin,
            "source": dgi.source,
        }));
    }

    // 6. Initialize optional extensions and composite tool executor.
    let ext_bridge = if let Some(ref exts) = spec.extensions {
        ExtensionBridge::spawn(exts).await.ok()
    } else {
        None
    };

    let executor = match CompositeExecutor::with_config(
        &workspace_root,
        config.searxng_url.clone(),
        config.brave_api_key.clone(),
        config.openalex_email.clone(),
        config.openalex_api_key.clone(),
        ext_bridge,
    ) {
        Ok(exec) => {
            // Re-baseline the executor against the task's actual working
            // directory (the dispatch cwd, host-path-normalized and canonical
            // through the OS symlink layer by `with_cwd`). This makes the
            // active-task cwd *live* for every dispatch: relative
            // `read_file`/`write_file`/`edit_file`/`list_dir`/`search_code`
            // paths resolve against it and the `bash` child spawns in it,
            // instead of silently falling back to the workspace root. When the
            // dispatch cwd is the workspace root itself (the common case) this
            // is a no-op relative to the prior behavior.
            exec.with_cwd(host_raw_path.clone())
        }
        Err(e) => {
            let err_msg = format!("failed to initialize executor: {e}");
            // Terminal event guarantee: `session_start` was already logged
            // above, so append a `final` / `failed` event here too — the
            // event ledger must never end without a terminal event on an
            // error exit (the pruner's terminal-set check relies on this).
            let _ = logger.append(json!({
                "type": "final",
                "status": "failed",
                "reason": err_msg.clone(),
            }));
            let _ = registry
                .transition(&spec.task_id, TaskStatus::Failed, Some(err_msg.clone()))
                .await;
            let guard = shared_lease.lock().unwrap();
            let _ = sem.release(&guard);
            // Defensive sweep: reclaim any orphaned slot (our lease is bound
            // to a now-terminal task; the reaper also covers the case where
            // this worker dies before releasing).
            sem.clear_reclaimable();
            return Err(err_msg);
        }
    };

    let tools = executor.tool_schemas();

    // Terminal artifacts (the reasoning-ceiling salvage report) are persisted
    // under the state dir's `.scratch/`. The per-session reasoning-effort
    // tier (validated at dispatch) is threaded into the runner so every
    // engine call in the session carries it.
    let mut options = runner::SessionOptions::with_state(state.clone())
        .with_reasoning_tags(
            config.reasoning_open_tag.clone(),
            config.reasoning_close_tag.clone(),
        )
        .with_turn_watchdog(config.turn_timeout_secs, config.max_turn_reasoning_tokens);
    options.workspace = Some(workspace_root.clone());
    options.reasoning_effort = spec.reasoning_effort.clone();
    // 0 when unset → the session-history watermark is disabled (zero behavior
    // change for callers that don't configure a context budget).
    options.max_context = max_ctx as usize;
    // Thread the operator-configured spill ceiling so oversized tool outputs
    // are spooled to the scratch dir (keeping the in-conversation footprint
    // bounded and the Turn-1 prefix stable).
    options.tool_spill_bytes = Some(config.tool_spill_bytes);
    options = options.with_lease_context(runner::LeaseYieldContext {
        sem: std::sync::Arc::new(sem.clone()),
        lease: shared_lease.clone(),
        task_id: spec.task_id.clone(),
    });

    // 8. Run session loop. The system prompt is the byte-identical warm KV
    //    prefix; the (variable) user message carries the CPG/PPR + DGI
    //    augmentation, so a resumed continuation reuses the exact same system
    //    prompt bytes and its prior conversation KV prefix stays intact.
    let outcome = runner::run_session(
        engine,
        &executor,
        &logger,
        &system_prompt,
        &user_prompt,
        &tools,
        Some(&options),
    )
    .await;

    // 9. Transition terminal status and release slot.
    match outcome {
        Ok(session_res) => {
            let terminal_status = if session_res.status == "completed" {
                TaskStatus::Completed
            } else if session_res.status == "reasoning_budget_exhausted" {
                let salvage_path = state
                    .root()
                    .join(".scratch")
                    .join(format!("salvage_{}.md", spec.session_id));
                let ws_salvage_path = workspace_root
                    .join(".scratch")
                    .join(format!("salvage_{}.md", spec.session_id));
                let has_salvage = (salvage_path.exists()
                    && std::fs::metadata(&salvage_path)
                        .map(|m| m.len() > 0)
                        .unwrap_or(false))
                    || (ws_salvage_path.exists()
                        && std::fs::metadata(&ws_salvage_path)
                            .map(|m| m.len() > 0)
                            .unwrap_or(false));
                if has_salvage {
                    TaskStatus::CompletedWithSalvage
                } else {
                    TaskStatus::Failed
                }
            } else {
                TaskStatus::Failed
            };
            let reason =
                if terminal_status == TaskStatus::Failed && session_res.final_text.is_empty() {
                    format!("[Task ended with status: {}]", session_res.status)
                } else {
                    session_res.final_text.clone()
                };
            let _ = registry
                .transition(&spec.task_id, terminal_status, Some(reason))
                .await;
            let guard = shared_lease.lock().unwrap();
            let _ = sem.release(&guard);
            // Defensive sweep: our lease is bound to a now-terminal task, so
            // the reaper reclaims any orphaned slot this worker left behind.
            sem.clear_reclaimable();
            // Refresh `telemetry/stats.json` from the final session ledger so
            // the stats file stays current after every completed or failed
            // session (best-effort: a write failure never breaks the worker).
            let _ = crate::telemetry::rollup::derive_stats(state.root(), &Default::default());
            Ok(session_res.final_text)
        }
        Err(e) => {
            let err_msg = e.to_string();
            // Terminal event guarantee: a `run_session` error exit returns
            // before its own `final` event is written, so append a
            // `final` / `failed` event here — the event ledger must never end
            // without a terminal event (the pruner's terminal-set check
            // relies on this). Written before `derive_stats` so the stats
            // rollup sees it.
            let _ = logger.append(json!({
                "type": "final",
                "status": "failed",
                "reason": err_msg.clone(),
            }));
            let _ = registry
                .transition(&spec.task_id, TaskStatus::Failed, Some(err_msg.clone()))
                .await;
            let guard = shared_lease.lock().unwrap();
            let _ = sem.release(&guard);
            // Defensive sweep: our lease is bound to a now-terminal task, so
            // the reaper reclaims any orphaned slot this worker left behind.
            sem.clear_reclaimable();
            // Refresh `telemetry/stats.json` from the final session ledger so
            // the stats file stays current after every completed or failed
            // session (best-effort: a write failure never breaks the worker).
            let _ = crate::telemetry::rollup::derive_stats(state.root(), &Default::default());
            Err(err_msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Completion, EngineError, Message, Metrics, ToolCall, ToolSchema};
    use crate::tools::FINISH_TOOL_NAME;
    use async_trait::async_trait;
    use std::collections::VecDeque;
    use std::path::PathBuf;

    struct ScriptedEngine {
        responses: tokio::sync::Mutex<VecDeque<Completion>>,
    }

    #[async_trait]
    impl ChatEngine for ScriptedEngine {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolSchema],
            _stream: bool,
            _reasoning_effort: Option<&str>,
        ) -> Result<Completion, EngineError> {
            self.responses
                .lock()
                .await
                .pop_front()
                .ok_or_else(|| EngineError::Malformed("no scripted response".into()))
        }
    }

    fn tmp_state() -> (StateDir, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "castor_worker_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let state = StateDir::new(&dir);
        let _ = state.ensure();
        (state, dir)
    }

    /// A scripted completion that ends the session via the explicit `finish`
    /// control-flow tool: the `summary` argument becomes the final report and
    /// the session lands in a single turn (no plain-text turn, which would
    /// otherwise be nudged now that the `finish` tool is advertised).
    fn finish_completion(summary: &str) -> Completion {
        Completion {
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![ToolCall {
                index: 0,
                id: "finish_1".to_string(),
                name: FINISH_TOOL_NAME.to_string(),
                arguments: format!(r#"{{"summary":{summary:?}}}"#),
            }],
            finish_reason: Some("tool_calls".to_string()),
            metrics: Metrics {
                ttft_ms: None,
                total_ms: 10.0,
                tokens_per_sec: None,
                prompt_tokens: None,
                cached_tokens: None,
                completion_tokens: None,
                reasoning_tokens: None,
            },
        }
    }

    #[tokio::test]
    async fn run_job_happy_path_transitions_to_completed() {
        let (state, dir) = tmp_state();
        let registry = TaskRegistry::new(&state);

        let task_id = registry
            .create("write a test file", dir.to_str().unwrap(), "sess_1")
            .await
            .unwrap();

        let spec = JobSpec {
            task_id: task_id.clone(),
            prompt: "write a test file".to_string(),
            cwd: dir.to_str().unwrap().to_string(),
            session_id: "sess_1".to_string(),
            reasoning_effort: None,
            extensions: None,
            skills: None,
            test_command: None,
            turns_budget: Some(10),
            timeout_ms: None,
        };

        let engine = ScriptedEngine {
            responses: tokio::sync::Mutex::new(VecDeque::from(vec![finish_completion(
                "Task successfully finished.",
            )])),
        };

        let loaded = crate::config::load().expect("load default config");
        let mut config = loaded.config;
        config.state_dir = dir.clone();

        let res = run_job_with_engine(&spec, &state, &config, &engine)
            .await
            .unwrap();
        assert_eq!(res, "Task successfully finished.");

        // Verify task record is Completed.
        let rec = registry.get(&task_id).await.unwrap();
        assert_eq!(rec.status, TaskStatus::Completed);
        assert_eq!(rec.reason, Some("Task successfully finished.".to_string()));

        // Verify event log was written.
        let logger = EventLogger::new(&state, "sess_1");
        let events = logger.read_all();
        let session_start = events
            .iter()
            .find(|e| e["type"] == "session_start")
            .expect("session_start event present");
        // Unset effort: the field is present and null (honest "server default
        // applied" — not missing, which would be indistinguishable from a
        // pre-override session).
        assert_eq!(
            session_start["reasoning_effort"],
            serde_json::Value::Null,
            "session_start must carry reasoning_effort=null when unset: {session_start}"
        );
        assert!(events.iter().any(|e| e["type"] == "final"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// When a worker session terminates, the telemetry rollup re-derives the
    /// stats and persists `<state_dir>/telemetry/stats.json`, so the stats
    /// file stays current after every completed or failed session.
    #[tokio::test]
    async fn run_job_persists_telemetry_stats_json_on_session_end() {
        let (state, dir) = tmp_state();
        let registry = TaskRegistry::new(&state);

        let task_id = registry
            .create("telemetry stats test", dir.to_str().unwrap(), "sess_tel")
            .await
            .unwrap();

        let spec = JobSpec {
            task_id: task_id.clone(),
            prompt: "telemetry stats test".to_string(),
            cwd: dir.to_str().unwrap().to_string(),
            session_id: "sess_tel".to_string(),
            reasoning_effort: None,
            extensions: None,
            skills: None,
            test_command: None,
            turns_budget: Some(10),
            timeout_ms: None,
        };

        let engine = ScriptedEngine {
            responses: tokio::sync::Mutex::new(VecDeque::from(vec![finish_completion(
                "Task successfully finished.",
            )])),
        };

        let loaded = crate::config::load().expect("load default config");
        let mut config = loaded.config;
        config.state_dir = dir.clone();

        let res = run_job_with_engine(&spec, &state, &config, &engine)
            .await
            .unwrap();
        assert_eq!(res, "Task successfully finished.");

        // The terminal session is projected into `telemetry/stats.json`.
        let stats_path = state.root().join("telemetry/stats.json");
        assert!(
            stats_path.exists(),
            "session termination must persist telemetry/stats.json"
        );
        let stats: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&stats_path).unwrap()).unwrap();
        assert_eq!(stats["total_sessions"], 1);
        assert_eq!(stats["total_tasks_completed"], 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A dispatch carrying a per-session `reasoning_effort` tier records the
    /// tier on the `session_start` event and threads it into
    /// [`runner::SessionOptions`] (which the runner forwards to the engine on
    /// every chat turn).
    #[tokio::test]
    async fn session_start_carries_reasoning_effort() {
        let (state, dir) = tmp_state();
        let registry = TaskRegistry::new(&state);

        let task_id = registry
            .create("effort test", dir.to_str().unwrap(), "sess_effort")
            .await
            .unwrap();

        let spec = JobSpec {
            task_id: task_id.clone(),
            prompt: "effort test".to_string(),
            cwd: dir.to_str().unwrap().to_string(),
            session_id: "sess_effort".to_string(),
            reasoning_effort: Some("xhigh".to_string()),
            extensions: None,
            skills: None,
            test_command: None,
            turns_budget: Some(10),
            timeout_ms: None,
        };

        let engine = ScriptedEngine {
            responses: tokio::sync::Mutex::new(VecDeque::from(vec![finish_completion(
                "done at xhigh",
            )])),
        };

        let loaded = crate::config::load().expect("load default config");
        let mut config = loaded.config;
        config.state_dir = dir.clone();

        let res = run_job_with_engine(&spec, &state, &config, &engine)
            .await
            .unwrap();
        assert_eq!(res, "done at xhigh");

        // The session_start event carries the tier string.
        let logger = EventLogger::new(&state, "sess_effort");
        let events = logger.read_all();
        let session_start = events
            .iter()
            .find(|e| e["type"] == "session_start")
            .expect("session_start event present");
        assert_eq!(
            session_start["reasoning_effort"],
            serde_json::json!("xhigh"),
            "session_start must carry the tier string: {session_start}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn run_job_marks_task_failed_when_auto_boot_fails() {
        let dir = std::env::temp_dir().join(format!("test_boot_fail_{}", now_epoch_ms()));
        let state = StateDir::new(&dir);
        state.ensure().unwrap();

        let registry = TaskRegistry::new(&state);
        let task_id = registry
            .create("failing boot task", "test", "sess_boot_fail")
            .await
            .unwrap();

        let spec = JobSpec {
            task_id: task_id.clone(),
            prompt: "failing boot task".into(),
            cwd: "test".into(),
            session_id: "sess_boot_fail".into(),
            reasoning_effort: None,
            extensions: None,
            skills: None,
            test_command: None,
            turns_budget: None,
            timeout_ms: None,
        };

        let loaded = crate::config::load().expect("load default config");
        let mut config = loaded.config;
        config.state_dir = dir.clone();
        config.launch_command = None;
        config.ports.engine = 0;
        config.base_url = None;

        let res = run_job(&spec, &state, &config).await;
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Engine auto-boot failed"));

        let rec = registry.get(&task_id).await.expect("task record exists");
        assert_eq!(rec.status, TaskStatus::Failed);
        assert!(rec.reason.unwrap().contains("Engine auto-boot failed"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression guard for the KV-prefix ordering invariant: the assembled
    /// worker system prompt MUST place the static persona at byte 0, then the
    /// `[Workspace]` root block, then the deterministic `[Workspace Skeleton]`,
    /// and finally the active-skills index (lowest → highest churn). This is
    /// what keeps the Turn-1 prefix byte-for-byte stable across sessions so the
    /// KV cache hits. Any reordering here silently degrades prefix reuse.
    #[test]
    fn system_prompt_prefix_ordering_is_stable_and_ordered() {
        let root = std::path::PathBuf::from("/work/repo");
        let skeleton = "[Workspace Skeleton]\n- src/\n- Cargo.toml\n- CLAUDE.md\n";

        let skills = vec![
            crate::skills::Skill {
                name: "canary-test-staging".into(),
                description: "Stage changes behind a targeted test file".into(),
                path: std::path::PathBuf::from("/work/repo/skills/canary-test-staging/SKILL.md"),
            },
            crate::skills::Skill {
                name: "hypothesis-generation".into(),
                description: "Structured root-cause hypothesis generation".into(),
                path: std::path::PathBuf::from("/work/repo/skills/hypothesis-generation/SKILL.md"),
            },
        ];
        let skills_index = skills::render_index(&skills);
        let prompt = build_worker_system_prompt(&root, true, skeleton, &skills_index);

        // (a) The static persona is byte-0 and is the FIRST text in the prompt.
        let persona = "You are Castor, an autonomous senior peer engineer";
        assert!(
            prompt.starts_with(persona),
            "persona must be byte 0, got prefix: {prompt:?}"
        );

        // Every section must appear, and each strictly after the last (ordering).
        let workspace = "[Workspace]\nRoot: /work/repo";
        let skeleton_marker = "[Workspace Skeleton]";
        // The skills index block is the `render_index` header (present-state
        // truth is a lowercase, unbracketed "Available skills." line).
        let skills_marker = "Available skills.";

        let p_persona = prompt.find(persona).expect("persona present");
        let p_workspace = prompt.find(workspace).expect("[Workspace] block present");
        let p_skeleton = prompt
            .find(skeleton_marker)
            .expect("[Workspace Skeleton] present");
        let p_skills = prompt.find(skills_marker).expect("skills index present");

        assert_eq!(p_persona, 0, "persona must start at byte 0");
        assert!(
            p_persona < p_workspace,
            "persona must precede [Workspace] (got persona={p_persona}, workspace={p_workspace})"
        );
        assert!(
            p_workspace < p_skeleton,
            "[Workspace] must precede [Workspace Skeleton] (got workspace={p_workspace}, skeleton={p_skeleton})"
        );
        assert!(
            p_skeleton < p_skills,
            "[Workspace Skeleton] must precede the skills index (got skeleton={p_skeleton}, skills={p_skills})"
        );

        // The `[Workspace]` block carries the exact root, and the skeleton
        // header is immediately followed by its deterministic entries.
        assert!(
            prompt.contains("\n\n[Workspace]\nRoot: /work/repo\n[Workspace Skeleton]\n- src/\n- Cargo.toml\n- CLAUDE.md\n"),
            "root and skeleton must be contiguous with a single newline join"
        );

        // The skills index is the terminal block (after the skeleton).
        let skills_suffix = &prompt[p_skills..];
        assert!(
            skills_suffix.contains("- canary-test-staging:")
                && skills_suffix.contains("- hypothesis-generation:"),
            "skills index must list every active skill: {skills_suffix}"
        );
    }

    /// When the workspace-skeleton prefix is disabled, the prompt still opens
    /// with the persona at byte 0, keeps the `[Workspace]` root, and omits the
    /// skeleton block entirely (but still appends the skills index).
    #[test]
    fn system_prompt_without_skeleton_still_ordered() {
        let root = std::path::PathBuf::from("/work/repo");
        let skills_index = skills::render_index(&[crate::skills::Skill {
            name: "hypothesis-generation".into(),
            description: "Structured root-cause hypothesis generation".into(),
            path: std::path::PathBuf::from("/work/repo/skills/hypothesis-generation/SKILL.md"),
        }]);

        // skeleton_enabled=false: even a non-empty skeleton string is ignored.
        let prompt = build_worker_system_prompt(
            &root,
            false,
            "[Workspace Skeleton]\n- src/\n",
            &skills_index,
        );

        assert!(prompt.starts_with("You are Castor, an autonomous senior peer engineer"));
        let p_workspace = prompt
            .find("[Workspace]\nRoot: /work/repo")
            .expect("workspace present");
        let p_skills = prompt
            .find("Available skills.")
            .expect("skills index present");
        assert!(
            p_workspace < p_skills,
            "[Workspace] must precede the skills index when skeleton is disabled"
        );
        assert!(
            !prompt.contains("[Workspace Skeleton]"),
            "skeleton must be omitted when disabled"
        );
    }

    /// The standing `Security & Authority` invariant must be present in the
    /// static (low-churn) persona block: it makes tool denials *binding*,
    /// forbids the `bash`/path bypass, and defines the escalation path to the
    /// Lead Architect / user. This is the normative half of the denial protocol
    /// (the other half is the directive carried by each denial message).
    #[test]
    fn system_prompt_defines_binding_denial_authority() {
        let root = std::path::PathBuf::from("/work/repo");
        let prompt = build_worker_system_prompt(&root, false, "", "");

        assert!(
            prompt.contains("Security & Authority:"),
            "invariant missing: {prompt}"
        );
        for needle in [
            "binding policy decisions",
            "Never route around",
            "`bash`/`cat`/`cp`/`mv`/`tee`",
            "report the constraint",
            "Lead Architect / user",
        ] {
            assert!(
                prompt.contains(needle),
                "invariant missing '{needle}': {prompt}"
            );
        }
        // The invariant sits in the static persona block: before [Workspace]
        // (byte-0 persona) and after the tool-routing list.
        let p_inv = prompt
            .find("Security & Authority:")
            .expect("invariant present");
        let p_workspace = prompt.find("[Workspace]").expect("workspace present");
        assert!(
            p_inv < p_workspace,
            "invariant must sit in the static block, before [Workspace]"
        );
    }

    #[tokio::test]
    async fn status_simplification_exhausted_reasoning_marks_task_completed_with_salvage() {
        let (state, dir) = tmp_state();
        let registry = TaskRegistry::new(&state);
        let task_id = registry
            .create("task reasoning fail", dir.to_str().unwrap(), "sess_fail")
            .await
            .unwrap();

        let spec = JobSpec {
            task_id: task_id.clone(),
            prompt: "solve task".to_string(),
            cwd: dir.to_str().unwrap().to_string(),
            session_id: "sess_fail".to_string(),
            reasoning_effort: None,
            extensions: None,
            skills: None,
            test_command: None,
            turns_budget: None,
            timeout_ms: None,
        };

        let engine = ScriptedEngine {
            responses: tokio::sync::Mutex::new(VecDeque::from(vec![
                Completion {
                    content: "".to_string(),
                    reasoning_content: Some("infinite thoughts...".into()),
                    tool_calls: Vec::new(),
                    finish_reason: Some("length".to_string()),
                    metrics: Metrics {
                        ttft_ms: None,
                        total_ms: 10.0,
                        tokens_per_sec: None,
                        prompt_tokens: None,
                        cached_tokens: None,
                        completion_tokens: None,
                        reasoning_tokens: None,
                    },
                },
                Completion {
                    content: "Salvage summary: reasoning ceiling hit".to_string(),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    finish_reason: Some("stop".to_string()),
                    metrics: Metrics {
                        ttft_ms: None,
                        total_ms: 10.0,
                        tokens_per_sec: None,
                        prompt_tokens: None,
                        cached_tokens: None,
                        completion_tokens: None,
                        reasoning_tokens: None,
                    },
                },
            ])),
        };

        let loaded = crate::config::load().expect("load default config");
        let mut config = loaded.config;
        config.state_dir = dir.clone();

        let res = run_job_with_engine(&spec, &state, &config, &engine)
            .await
            .unwrap();
        assert!(res.contains("Salvage summary: reasoning ceiling hit"));

        let rec = registry.get(&task_id).await.unwrap();
        assert_eq!(rec.status, TaskStatus::CompletedWithSalvage);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn run_job_with_engine_fails_fast_when_dgi_probe_fails() {
        let (state, dir) = tmp_state();
        let registry = TaskRegistry::new(&state);
        let task_id = registry
            .create("dgi fail fast", dir.to_str().unwrap(), "sess_dgi_fail")
            .await
            .unwrap();

        let spec = JobSpec {
            task_id: task_id.clone(),
            prompt: "fix something".to_string(),
            cwd: dir.to_str().unwrap().to_string(),
            session_id: "sess_dgi_fail".to_string(),
            reasoning_effort: None,
            extensions: None,
            skills: None,
            test_command: None,
            turns_budget: None,
            timeout_ms: None,
        };

        // Bind and drop an ephemeral listener so connection refuses immediately.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let closed_port = listener.local_addr().unwrap().port();
        drop(listener);

        let loaded = crate::config::load().expect("load default config");
        let mut config = loaded.config;
        config.state_dir = dir.clone();
        config.base_url = Some(format!("http://127.0.0.1:{closed_port}/v1"));
        config.model = Some("test-model".to_string());
        config.engine_type = Some("vllm".to_string());

        let engine = ScriptedEngine {
            responses: tokio::sync::Mutex::new(VecDeque::new()),
        };

        let err = run_job_with_engine(&spec, &state, &config, &engine)
            .await
            .unwrap_err();
        assert!(
            err.starts_with("DgiProbeError:"),
            "expected DgiProbeError fail-fast, got: {err}"
        );

        let rec = registry.get(&task_id).await.unwrap();
        assert_eq!(rec.status, TaskStatus::Failed);
        assert!(rec.reason.unwrap().starts_with("DgiProbeError:"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The system prompt is a pure function of the workspace's stable shape and
    /// the build — **not** of the dispatch prompt. Two different prompts in the
    /// same workspace must therefore produce **byte-identical** system prompts
    /// (the warm KV prefix that resumed sessions reuse). The prompt-specific
    /// CPG/PPR context must NOT appear in the system prompt; it attaches to the
    /// user message instead.
    #[test]
    fn system_prompt_is_byte_identical_across_prompts_and_cpg_goes_to_user_message() {
        let root = std::path::PathBuf::from("/work/repo");
        let skeleton = "[Workspace Skeleton]\n- src/\n- Cargo.toml\n";
        let skills_index = skills::render_index(&[crate::skills::Skill {
            name: "canary-test-staging".into(),
            description: "Stage changes behind a targeted test file".into(),
            path: std::path::PathBuf::from("/work/repo/skills/canary-test-staging/SKILL.md"),
        }]);

        // Two different prompts in the same workspace.
        let prompt_a = "fix the parser in src/ast.rs";
        let prompt_b = "write tests for the graph pagerank module";

        let sys_a = build_worker_system_prompt(&root, true, skeleton, &skills_index);
        let sys_b = build_worker_system_prompt(&root, true, skeleton, &skills_index);
        assert_eq!(
            sys_a, sys_b,
            "byte-identical system prompt is the warm KV prefix invariant"
        );

        // The system prompt is prompt-agnostic: no prompt-specific CPG/PPR
        // context (those land in the user message).
        assert!(
            !sys_a.contains("Retrieved Code Context"),
            "system prompt must NOT carry CPG/PPR retrieved context"
        );
        assert!(
            !sys_a.contains(prompt_a) && !sys_a.contains(prompt_b),
            "system prompt must not embed the dispatch prompt"
        );

        // With no CPG context (skeleton disabled) and no DGI advisory (the gate
        // is strictly binary and injects none), the augmentation is empty, so
        // the caller uses the raw prompt (preserving the fresh `[system, user]`
        // byte shape).
        let empty_aug = build_user_turn_augmentation(&root, false, prompt_a);
        assert!(
            empty_aug.is_empty(),
            "no CPG context and no DGI advisory -> empty augmentation"
        );
    }

    /// With a real workspace and the workspace-skeleton prefix enabled, the
    /// prompt-specific CPG/PPR retrieved-code context is computed and attached
    /// to the **user-message** augmentation (never the system prompt). The
    /// augmentation carries the `[Retrieved Code Context]` block and names the
    /// relevant symbol, but does not embed the raw prompt text (the prompt is
    /// appended after the block by the caller).
    #[cfg(feature = "ast")]
    #[test]
    fn user_message_augmentation_carries_cpg_retrieved_context() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("calc.rs"),
            "pub fn calculate_sum(a: i32, b: i32) -> i32 { a + b }\npub fn calculate_product(a: i32, b: i32) -> i32 { a * b }\n",
        )
        .unwrap();

        let prompt = "Need the calculate_sum implementation";
        let aug = build_user_turn_augmentation(tmp.path(), true, prompt);

        assert!(
            !aug.is_empty(),
            "augmentation must carry the CPG/PPR retrieved context for this prompt"
        );
        assert!(
            aug.contains("[Retrieved Code Context]"),
            "the retrieved-code block header must land in the user message: {aug}"
        );
        assert!(
            aug.contains("calculate_sum"),
            "the relevant symbol must be named in the user message: {aug}"
        );
        assert!(
            !aug.contains(prompt),
            "the augmentation is a context prefix; the raw prompt is appended separately"
        );
        let _ = tmp;
    }

    #[tokio::test]
    async fn probe_git_provenance_handles_non_git_clean_and_dirty_repos() {
        let non_git = tempfile::tempdir().unwrap();
        assert_eq!(probe_git_provenance(non_git.path()).await, (None, None));

        let repo = tempfile::tempdir().unwrap();
        let run_git = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(repo.path())
                .args(args)
                .status()
                .expect("git command should run");
            assert!(status.success(), "git {args:?} failed");
        };
        run_git(&["init"]);
        run_git(&["config", "user.email", "castor@example.com"]);
        run_git(&["config", "user.name", "Castor Test"]);
        std::fs::write(repo.path().join("tracked.txt"), "v1\n").unwrap();
        run_git(&["add", "tracked.txt"]);
        run_git(&["commit", "-m", "initial"]);

        // Untracked files do not mark git_dirty (--untracked-files=no).
        std::fs::write(repo.path().join("untracked.txt"), "temp\n").unwrap();
        let (head, dirty) = probe_git_provenance(repo.path()).await;
        assert!(
            head.as_ref()
                .is_some_and(|s| s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit())),
            "expected 40-char SHA, got {head:?}"
        );
        assert_eq!(dirty, Some(false));

        // Modifying a tracked file flips git_dirty to Some(true).
        std::fs::write(repo.path().join("tracked.txt"), "v2\n").unwrap();
        let (head2, dirty2) = probe_git_provenance(repo.path()).await;
        assert_eq!(head2, head);
        assert_eq!(dirty2, Some(true));
    }
}
