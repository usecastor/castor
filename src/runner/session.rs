//! Autonomous multi-turn session loop and terminal outcome resolution.

use futures_util::future::join_all;
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use crate::engine::{Message, ToolCall, ToolSchema};
use crate::runner::budget::{LOOP_THRESHOLD, LOOP_WINDOW};
use crate::runner::context::{
    CONTEXT_CEILING_DIRECTIVE, history_tokens, in_flight_ceiling_tokens, in_flight_context_tokens,
    load_messages, prior_session_summary_block, save_messages, should_resume, watermark_tokens,
};
use crate::runner::events::EventLogger;
use crate::runner::loop_detector::{LoopDetector, LoopState};
use crate::runner::messages::{
    completion_text, metrics_json, msg, sanitize_tool_delimiters, strip_reasoning_tags,
};
use crate::runner::salvage::{
    extract_finish_summary, is_reasoning_ceiling, run_degenerate_final_salvage,
    run_reasoning_ceiling_salvage,
};
use crate::runner::skeleton::spill_tool_output;
use crate::runner::watchdog::chat_with_watchdog;
use crate::runner::{
    ChatEngine, RunnerError, SessionOptions, SessionResult, ToolExecutor, ToolOutcome,
};
use crate::tools::FINISH_TOOL_NAME;

/// Maximum consecutive plain-text (no-tool-call) turns the runner will nudge
/// before landing the session, when the session opted into the explicit
/// `finish` control-flow tool. Bounded so a model that neither calls tools nor
/// calls `finish` can never loop the session on repeated nudges.
const MAX_NO_TOOL_NUDGES: u32 = 2;

/// Append one `role: "tool"` result message and ledger event for a Phase-4
/// `finish` call. Returns the (trimmed, non-empty) `summary` carried by the
/// last such call, if any, so the caller can resolve the final report. The
/// result message is appended in the caller's dispatch order, preserving the
/// `assistant(tool_calls)` → `tool(results)` alternation a resumed session
/// requires.
fn append_finish_results(
    messages: &mut Vec<Message>,
    logger: &EventLogger,
    turns: u32,
    finish_calls: &[(usize, &ToolCall)],
    result: &str,
) -> Option<String> {
    let mut last_summary: Option<String> = None;
    for (_idx, tc) in finish_calls {
        last_summary = extract_finish_summary(&tc.arguments);
        messages.push(Message {
            role: "tool".into(),
            content: result.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tc.id.clone()),
        });
        let _ = logger.append(serde_json::json!({
            "type": "tool_result",
            "turn": turns,
            "tool_call_id": tc.id,
            "name": tc.name,
            "is_error": false,
            "output": result,
        }));
    }
    last_summary
}

/// Helper to decide whether a resumed history should be kept: the 90%
/// ceiling-aligned watermark (`in_flight_ceiling_tokens` / `watermark_tokens`)
/// is checked against the history plus the new prompt.
fn check_resume(
    history: &[Message],
    new_user_prompt: &str,
    options: Option<&SessionOptions>,
) -> bool {
    let max_context = options.map(|o| o.max_context).unwrap_or(0);
    should_resume(history, new_user_prompt, max_context)
}

/// Tools executing in Phase 1 (Concurrent read-only exploration).
fn is_phase1_read(name: &str) -> bool {
    matches!(
        name,
        "read_file"
            | "list_dir"
            | "search_code"
            | "ast_search"
            | "web_search"
            | "web_fetch"
            | "paper_lookup"
    )
}

/// Tools executing in Phase 2 (Partitioned file mutations).
fn is_phase2_write(name: &str) -> bool {
    matches!(name, "write_file" | "edit_file" | "ast_replace")
}

/// Resolve the canonical target file key for a Phase 2 mutation call so
/// writes to distinct files execute concurrently while writes to the same
/// file serialize in dispatch order. Returns `None` for directory-wide or
/// unkeyed mutations (such as workspace-wide `ast_replace`), which execute as
/// a sequential barrier.
fn canonical_write_key(tc: &ToolCall, workspace: Option<&Path>) -> Option<PathBuf> {
    let args: serde_json::Value = serde_json::from_str(&tc.arguments).ok()?;
    let raw = args.get("path").and_then(|v| v.as_str())?.trim();
    if raw.is_empty() {
        return None;
    }
    let host_path = crate::platform::to_host_path(raw);
    let joined = if host_path.is_absolute() {
        host_path
    } else if let Some(ws) = workspace {
        ws.join(host_path)
    } else {
        host_path
    };

    let mut norm = PathBuf::new();
    for comp in joined.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                if !norm.pop() {
                    norm.push(comp);
                }
            }
            other => norm.push(other),
        }
    }

    if tc.name == "ast_replace" && norm.is_dir() {
        return None;
    }

    if let Ok(canon) = std::fs::canonicalize(&norm) {
        return Some(canon);
    }

    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = norm.as_path();
    while let Some(name) = cur.file_name() {
        tail.push(name.to_os_string());
        if let Some(parent) = cur.parent() {
            let probe = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
            if let Ok(canon_parent) = std::fs::canonicalize(probe) {
                let mut out = canon_parent;
                for seg in tail.into_iter().rev() {
                    #[cfg(windows)]
                    let seg = std::ffi::OsString::from(seg.to_string_lossy().to_ascii_lowercase());
                    out.push(seg);
                }
                return Some(out);
            }
            cur = parent;
        } else {
            break;
        }
    }

    #[cfg(windows)]
    let norm = PathBuf::from(norm.to_string_lossy().to_ascii_lowercase());
    Some(norm)
}

async fn execute_phase2_writes<'a>(
    executor: &dyn ToolExecutor,
    phase2: Vec<(usize, &'a ToolCall)>,
    workspace: Option<&Path>,
) -> Vec<(usize, &'a ToolCall, ToolOutcome)> {
    async fn run_one<'a>(
        executor: &dyn ToolExecutor,
        idx: usize,
        tc: &'a ToolCall,
    ) -> (usize, &'a ToolCall, ToolOutcome) {
        let outcome = match executor.execute(&tc.name, &tc.arguments).await {
            Ok(o) => o,
            Err(e) => ToolOutcome {
                text: format!("Error: {e}"),
            },
        };
        (idx, tc, outcome)
    }

    async fn flush_wave<'a>(
        executor: &dyn ToolExecutor,
        wave: &mut BTreeMap<PathBuf, Vec<(usize, &'a ToolCall)>>,
        results: &mut Vec<(usize, &'a ToolCall, ToolOutcome)>,
    ) {
        if wave.is_empty() {
            return;
        }
        let groups = std::mem::take(wave);
        let futs = groups.into_values().map(|calls| async move {
            let mut group_out = Vec::with_capacity(calls.len());
            for (idx, tc) in calls {
                group_out.push(run_one(executor, idx, tc).await);
            }
            group_out
        });
        for group_out in join_all(futs).await {
            results.extend(group_out);
        }
    }

    let mut results = Vec::with_capacity(phase2.len());
    let mut wave: BTreeMap<PathBuf, Vec<(usize, &'a ToolCall)>> = BTreeMap::new();
    for (idx, tc) in phase2 {
        match canonical_write_key(tc, workspace) {
            Some(key) => wave.entry(key).or_default().push((idx, tc)),
            None => {
                flush_wave(executor, &mut wave, &mut results).await;
                results.push(run_one(executor, idx, tc).await);
            }
        }
    }
    flush_wave(executor, &mut wave, &mut results).await;
    results
}

/// Idempotent read-only tools whose results may be deduplicated: re-reading
/// the same coordinate without a mutation in between yields identical bytes.
fn is_idempotent_read(name: &str) -> bool {
    matches!(
        name,
        "read_file" | "list_dir" | "search_code" | "ast_search"
    )
}

/// Mutating tools that can change a file's bytes between two reads.
fn is_mutating_tool(name: &str) -> bool {
    matches!(name, "edit_file" | "write_file" | "ast_replace" | "bash")
}

/// The file extensions (ASCII-lowercased) that count as executable / compiled
/// source code: a successful mutation of a file with one of these extensions is
/// a change that a build or test command can verify.
const VERIFIABLE_CODE_EXTENSIONS: [&str; 33] = [
    "rs", "py", "js", "mjs", "cjs", "jsx", "ts", "mts", "cts", "tsx", "go", "c", "cc", "cpp",
    "cxx", "h", "hpp", "java", "cs", "rb", "php", "swift", "kt", "kts", "scala", "dart", "sol",
    "hs", "ex", "exs", "lua", "sh", "bash",
];

/// Pure decision: does a tool call count as a *verifiable code mutation* —
/// a successful `edit_file` / `write_file` / `ast_replace` on a source file?
///
/// Returns `false` when the call errored (`is_error`) or the tool name is not
/// one of the three file-mutation tools. Otherwise it extracts the `"path"`
/// argument from `args_json` (via `serde_json`) and rejects any path that
/// contains a `.scratch` segment. The file extension (ASCII-lowercased) must
/// be one of [`VERIFIABLE_CODE_EXTENSIONS`] for the call to be verifiable.
/// Documentation, data, and other non-source files (`.md`, `.json`, `.toml`,
/// `.yaml`, `.html`, `.css`, `.lock`, extensionless, …) are *not* verifiable:
/// editing them never requires a build/test verification command.
///
/// This is the loop-proof gate for the post-mutation verification nudge.
pub(crate) fn is_verifiable_code_mutation(
    tool_name: &str,
    args_json: &str,
    is_error: bool,
) -> bool {
    if is_error {
        return false;
    }
    if !matches!(tool_name, "edit_file" | "write_file" | "ast_replace") {
        return false;
    }
    let Ok(args) = serde_json::from_str::<serde_json::Value>(args_json) else {
        return false;
    };
    let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
        return false;
    };

    // A `.scratch` segment (the sanctioned scratchpad) is never verifiable
    // code: scratch artifacts are throwaway, not shipped source.
    if path.split(['/', '\\']).any(|seg| seg == ".scratch") {
        return false;
    }

    // The ASCII-lowercased extension must be a known source-code extension.
    let ext = path.rsplit('.').next().unwrap_or("");
    VERIFIABLE_CODE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
}

/// Apply the post-mutation verification guard bookkeeping for one executed
/// tool call: a `bash` call disarms the unverified flag (any bash command
/// counts as the verification command, so we never second-guess what the agent
/// ran); a successful verifiable code mutation arms it. No-op on the ceiling
/// synthesis turn (tool dispatch is disabled there).
fn record_verification_state(
    unverified_code_edit: &mut bool,
    break_after_turn: bool,
    tc: &ToolCall,
    outcome_is_error: bool,
) {
    if break_after_turn {
        return;
    }
    if tc.name == "bash" {
        *unverified_code_edit = false;
    } else if is_verifiable_code_mutation(&tc.name, &tc.arguments, outcome_is_error) {
        *unverified_code_edit = true;
    }
}

/// Tail-only byte-identical tool-result deduplication.
///
/// Called with `messages` already containing the current turn's assistant
/// message (so the current call is part of the history) but *before* the new
/// tail `role: "tool"` message is pushed. Scans the earlier messages backwards
/// for a prior tool message whose content is byte-identical to
/// `sanitized_tool_content` and whose corresponding assistant `ToolCall` has
/// the exact same `(name, arguments)`. Only idempotent read tools are
/// deduplicated; `read_file` additionally requires a byte-identical result of
/// at least 256 bytes and no mutating tool call between the prior read and
/// the current call (a small-file re-read with a possible mutation in between
/// is left untouched). When a match exists the *new tail* message carries
/// only a compact pointer; the earlier message is never mutated, so the KV
/// prefix cache stays 100% intact.
fn dedup_tool_content(messages: &[Message], tc: &ToolCall, sanitized: &str) -> String {
    let is_read = tc.name == "read_file";
    if !is_idempotent_read(&tc.name) {
        return sanitized.to_string();
    }
    if is_read {
        // A byte-identical re-read of a small file carries little KV savings;
        // only compress once the result is substantive.
        if sanitized.len() <= 256 {
            return sanitized.to_string();
        }
    }

    for (i, m) in messages.iter().enumerate().rev() {
        if m.role != "tool" {
            continue;
        }
        if m.content != sanitized {
            continue;
        }
        let prior_id = m.tool_call_id.as_deref();
        // Find the assistant message whose tool call produced this prior
        // result (matched by the result's tool_call_id).
        let Some((ai, am)) = messages.iter().enumerate().find(|(_, a)| {
            a.role == "assistant"
                && a.tool_calls
                    .iter()
                    .any(|c| prior_id.is_some_and(|p| c.id.as_str() == p))
        }) else {
            continue;
        };
        let Some(prior) = am
            .tool_calls
            .iter()
            .find(|c| prior_id.is_some_and(|p| c.id.as_str() == p))
        else {
            continue;
        };
        if prior.name != tc.name || prior.arguments != tc.arguments {
            continue;
        }
        if is_read
            && messages
                .iter()
                .enumerate()
                .filter(|(k, _)| *k > ai && *k < i)
                .filter(|(_, a)| a.role == "assistant")
                .any(|(_, a)| a.tool_calls.iter().any(|c| is_mutating_tool(&c.name)))
        {
            continue;
        }
        return format!(
            "[Unchanged: byte-identical to prior `{}` result earlier in this session. \
             Reuse the content already in context above.]",
            tc.name
        );
    }
    sanitized.to_string()
}

/// Run a multi-turn agent session.
///
/// Builds the message list (tool schemas last, for the vLLM APC contract),
/// calls the engine, executes any tool calls via the [`ToolExecutor`], appends
/// the results as tool messages, and continues until the model produces a
/// final or the turn budget is exhausted.
///
/// `options` supplies the state / workspace context the runner needs for
/// terminal artifacts. It is optional: pass `None` for callers that only
/// care about the in-band [`SessionResult`] (e.g. the offline eval replay).
/// When present and the session terminates at its reasoning ceiling, the
/// salvage pass writes a plain-text report to
/// `<state_or_workspace>/.scratch/salvage_<session_id>.md` (see
/// [`SessionOptions`] and the deliberation-ceiling salvage path below).
#[allow(clippy::too_many_arguments)] // each argument is a distinct session dependency
pub async fn run_session(
    engine: &dyn ChatEngine,
    executor: &dyn ToolExecutor,
    logger: &EventLogger,
    system_prompt: &str,
    user_prompt: &str,
    tools: &[ToolSchema],
    options: Option<&SessionOptions>,
) -> Result<SessionResult, RunnerError> {
    let session_id = logger.session_id();

    // Wall-clock start: the `final` event reports `duration_ms` (the total
    // session runtime) so the telemetry duration series has a source.
    let started = std::time::Instant::now();

    // Named sessions resume their persisted history (see the module docs):
    // the system prompt is re-supplied first, the sanitized non-system history
    // is prepended (if present and within the 90% ceiling-aligned resume
    // watermark — the same fraction as the in-flight ceiling), and the new
    // user prompt is appended last. `None` state or an empty/absent history
    // yields the original fresh `[system, user]` shape.
    let mut messages = vec![msg("system", system_prompt)];
    let user_prompt = {
        if let Some(state) = options.and_then(|o| o.state.as_ref())
            && let Some(history) = load_messages(state, session_id)
            && !history.is_empty()
        {
            let max_context = options.map(|o| o.max_context).unwrap_or(0);
            let fits_watermark = if max_context > 0 {
                let mut candidate = Vec::with_capacity(messages.len() + history.len() + 1);
                candidate.extend(messages.iter().cloned());
                candidate.extend(history.iter().cloned());
                candidate.push(msg("user", user_prompt));
                if let Some(exact) = engine.count_tokens(&candidate, tools).await {
                    exact <= watermark_tokens(max_context)
                } else {
                    check_resume(&history, user_prompt, options)
                }
            } else {
                check_resume(&history, user_prompt, options)
            };
            if fits_watermark {
                messages.extend(history);
                user_prompt.to_string()
            } else {
                tracing::info!(
                    session = %session_id,
                    "persisted history exceeds the 90% max_context resume ceiling; \
                     rolling over to a fresh session with a prior-session summary"
                );
                let _ = logger.append(serde_json::json!({
                    "type": "session_rollover",
                    "session": session_id,
                    "reason": "resume_ceiling",
                }));
                // Carry the prior run's last substantive answer into the new
                // tail user message so the rolled session keeps the prior
                // conclusions instead of starting from amnesia.
                match prior_session_summary_block(&history) {
                    Some(block) => format!("{block}{user_prompt}"),
                    None => user_prompt.to_string(),
                }
            }
        } else {
            user_prompt.to_string()
        }
    };
    messages.push(msg("user", &user_prompt));
    let mut loop_detector = LoopDetector::new(LOOP_WINDOW, LOOP_THRESHOLD);
    let mut turns: u32 = 0;
    let mut final_text;
    let mut tool_activity = false;
    let mut status = "completed".to_string();

    // Per-session reasoning-effort override, applied uniformly to every
    // engine call in the session (main turns, salvage, synthesis) — a
    // mid-session tier change would break the prefix cache, so the tier
    // is session-scoped. `None` leaves the payload byte-identical to the
    // no-override shape (the server-side default applies).
    let effort = options
        .and_then(|o| o.reasoning_effort.as_deref())
        .filter(|e| !e.is_empty());

    // Per-turn stall watchdog. `0` disables each trigger;
    // both default to disabled so callers without a watchdog are unchanged.
    let (wd_timeout, wd_max_reasoning) = match options {
        Some(o) => (
            o.turn_timeout_secs.unwrap_or(0),
            o.max_turn_reasoning_tokens.unwrap_or(0),
        ),
        None => (0, 0),
    };

    // Reasoning-delimiter tag pair for the session (reserved defaults when
    // `SessionOptions` does not override them). Hoisted once so the per-turn
    // assistant strip, the main tool loop, and the severed-stop continuation
    // all share the same pair (DRY) instead of each re-deriving it.
    let open_tag = options
        .map(|o| o.effective_open_tag())
        .unwrap_or_else(|| concat!("<th", "ink>").to_string());
    let close_tag = options
        .map(|o| o.effective_close_tag())
        .unwrap_or_else(|| concat!("</th", "ink>").to_string());

    // In-flight 90% context ceiling (re-checked every turn). When accumulated
    // context reaches 90% of `max_context`, the next turn disables tool
    // dispatch, appends `CONTEXT_CEILING_DIRECTIVE`, and lands the final
    // completion in the remaining 10% headroom. `0` disables the gate.
    let ceiling_tokens = in_flight_ceiling_tokens(options.map(|o| o.max_context).unwrap_or(0));
    let mut ceiling_triggered = false;
    let mut break_after_turn = false;
    // Post-mutation verification guard state (see [`is_verifiable_code_mutation`]).
    // `unverified_code_edit` flips true on a successful source-code mutation and
    // resets on any `bash` call (a verification command). `verification_nudge_sent`
    // is a one-shot latch: the verification advisory is injected at most once per
    // session, so the advisory cannot re-fire on every subsequent unverified edit
    // turn and the session can never loop on it.
    let mut unverified_code_edit = false;
    let mut verification_nudge_sent = false;
    // Explicit `finish` opt-in, detected by *schema name presence*: the session
    // is considered to have advertised the `finish` control-flow tool iff some
    // tool schema in the (unchanged, byte-identical) list is named `finish`
    // (the `function.name` slot in the OpenAI wire shape; the Rust
    // [`ToolSchema`] carries it directly as `.name`). This preserves every
    // existing mock-tool test — they pass plain `tool_schema(name)` fixtures
    // that are never named `finish` — so `explicit_finish_enabled` stays
    // `false` and the new nudge / Phase-4 interception paths remain inert for
    // them. Detection never mutates `tools`, so the KV prefix cache stays
    // intact.
    let explicit_finish_enabled = tools.iter().any(|t| t.name == FINISH_TOOL_NAME);
    // Bounded plain-text nudge counter (explicit `finish` sessions only):
    // reset on any turn that executes at least one tool call; at most
    // `MAX_NO_TOOL_NUDGES` consecutive no-tool turns get a nudge before the
    // session lands, so a model that neither calls tools nor `finish` can
    // never loop the session.
    let mut consecutive_no_tool_nudges: u32 = 0;
    // One-shot latch: an empty / malformed `finish(summary)` is rejected (with
    // a tool result asking for a non-empty summary) at most once per session.
    let mut empty_finish_rejected = false;
    let mut last_prompt_tokens: Option<usize> = if ceiling_tokens > 0 {
        engine.count_tokens(&messages, tools).await
    } else {
        None
    };
    let mut accounted_msgs: usize = if last_prompt_tokens.is_some() {
        messages.len()
    } else {
        0
    };

    loop {
        turns += 1;

        // Pre-turn backpressure: if the aggregate KV pool is under pressure
        // and another heavier or senior active task has priority, pause asynchronously
        // until that task finishes and frees its KV footprint.
        if let Some(ctx) = options.and_then(|o| o.lease_context.as_ref()) {
            let lease = match ctx.lease.lock() {
                Ok(guard) => Some(guard.clone()),
                Err(_) => None,
            };
            if let Some(l) = lease
                && ctx.sem.should_yield(&l)
            {
                let _ = logger.append(serde_json::json!({
                    "type": "cooperative_yield",
                    "turn": turns,
                    "task_id": ctx.task_id,
                    "reason": "pool_backpressure",
                }));
                ctx.sem.wait_backpressure(&l).await;
            }
        }

        if !ceiling_triggered && ceiling_tokens > 0 {
            if accounted_msgs < messages.len()
                && let Some(exact) = engine.count_tokens(&messages, tools).await
            {
                last_prompt_tokens = Some(exact);
                accounted_msgs = messages.len();
            }
            let current_tokens =
                in_flight_context_tokens(&messages, accounted_msgs, last_prompt_tokens);
            if current_tokens >= ceiling_tokens {
                tracing::info!(
                    session = %session_id,
                    context_tokens = current_tokens,
                    ceiling_tokens,
                    "in-flight 90% context ceiling reached; disabling tool dispatch and \
                     landing a final completion in the remaining headroom"
                );
                let _ = logger.append(serde_json::json!({
                    "type": "context_ceiling",
                    "turn": turns,
                    "history_tokens": current_tokens,
                    "max_context": ceiling_tokens,
                }));
                messages.push(msg("user", CONTEXT_CEILING_DIRECTIVE));
                ceiling_triggered = true;
                break_after_turn = true;
            }
        }

        // The tool schema prefix must stay byte-identical across every turn of
        // the session — the vLLM APC prompt keeps it fixed so the warm KV
        // prefix is never invalidated — including the ceiling synthesis turn.
        // On that turn, any tool call the engine still emits is never executed
        // (the loop lands the completion and breaks below).
        let active_tools: &[ToolSchema] = tools;
        accounted_msgs = messages.len();
        let completion = {
            let primary = chat_with_watchdog(
                engine,
                &messages,
                active_tools,
                effort,
                wd_timeout,
                wd_max_reasoning,
                logger,
                turns,
            )
            .await;
            match primary {
                Ok(c) => c,
                Err(crate::engine::EngineError::Sse(err)) => {
                    tracing::warn!(
                        "Transient SSE stream error on turn {turns}: {err}; retrying once"
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    engine.chat(&messages, active_tools, true, effort).await?
                }
                Err(err) => return Err(RunnerError::Engine(err)),
            }
        };

        // Record the dispatch event (with the engine's per-turn token usage,
        // so current Rust runs contribute to the telemetry token totals).
        logger
            .append(serde_json::json!({
                "type": "dispatch",
                "turn": turns,
                "finish_reason": completion.finish_reason,
                "content": completion.content,
                "tool_calls": completion
                    .tool_calls
                    .iter()
                    .map(|tc| serde_json::json!({
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }))
                    .collect::<Vec<_>>(),
                "metrics": metrics_json(&completion.metrics),
            }))
            .map_err(RunnerError::Io)?;

        if let Some(pt) = completion.metrics.prompt_tokens {
            last_prompt_tokens = Some(pt as usize);
        }

        // Strip reasoning traces from the assistant content in-flight, on both
        // tool-call and non-tool-call turns, exactly as `to_persisted` strips
        // them on load — a resumed session then replays byte-identical
        // assistant content, so resuming never invalidates the KV prefix.
        let sanitized_content = strip_reasoning_tags(&completion.content, &open_tag, &close_tag);
        messages.push(Message {
            role: "assistant".into(),
            content: sanitized_content,
            tool_calls: completion.tool_calls.clone(),
            tool_call_id: None,
        });

        if break_after_turn {
            // The in-flight 90% context ceiling: no workspace tool is executed
            // on this turn. If the model still emitted tool calls, close every
            // one with a well-formed `role: "tool"` result so a resumed session
            // never carries a dangling, unanswered `tool_call` (a hard
            // provider rejection on reload). `finish` is acknowledged as
            // completed; any workspace tool is recorded as skipped. The final
            // text prefers an explicit `finish(summary)` argument, else the
            // turn's own synthesis. `status` stays `"completed"` — this path
            // lands the completion in the remaining headroom and never nudges.
            if !completion.tool_calls.is_empty() {
                let finish_summary = completion
                    .tool_calls
                    .iter()
                    .find(|tc| tc.name == FINISH_TOOL_NAME)
                    .and_then(|tc| extract_finish_summary(&tc.arguments));
                for tc in &completion.tool_calls {
                    let response = if tc.name == FINISH_TOOL_NAME {
                        "Task completed.".to_string()
                    } else {
                        "Skipped: 90% working-context ceiling reached.".to_string()
                    };
                    messages.push(Message {
                        role: "tool".into(),
                        content: response.clone(),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(tc.id.clone()),
                    });
                    let _ = logger.append(serde_json::json!({
                        "type": "tool_result",
                        "turn": turns,
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "is_error": false,
                        "output": response,
                    }));
                }
                final_text = finish_summary.unwrap_or_else(|| completion_text(&completion));
            } else {
                final_text = completion_text(&completion);
            }
            break;
        }

        if completion.tool_calls.is_empty() {
            let ceiling_hit = is_reasoning_ceiling(&completion);
            final_text = completion_text(&completion);

            if ceiling_hit {
                // Reasoning ceiling: salvage first, never nudged.
                let outcome = run_reasoning_ceiling_salvage(
                    engine,
                    executor,
                    logger,
                    &mut messages,
                    tools,
                    effort,
                    turns,
                    options,
                    session_id,
                )
                .await;
                final_text = outcome.final_text;
                status = outcome.status;
                break;
            }

            // Degenerate empty final after real tool activity: salvage second,
            // never nudged.
            if final_text.trim().is_empty() && tool_activity {
                let outcome =
                    run_degenerate_final_salvage(engine, logger, &mut messages, effort, turns)
                        .await?;
                final_text = outcome.final_text;
                status = outcome.status;
                break;
            }

            // After both salvage paths: when the session opted into the
            // explicit `finish` tool, nudge a plain-text (no-tool-call) turn at
            // most `MAX_NO_TOOL_NUDGES` times in a row before landing. Zero
            // prose regexes — the decision is purely structural. Any turn that
            // does call tools (below) resets the counter.
            if explicit_finish_enabled && consecutive_no_tool_nudges < MAX_NO_TOOL_NUDGES {
                consecutive_no_tool_nudges += 1;
                let _ = logger.append(serde_json::json!({
                    "type": "no_tool_nudge",
                    "turn": turns,
                    "count": consecutive_no_tool_nudges,
                }));
                messages.push(msg(
                    "user",
                    "[System: No tool call received. Continue calling tools if work remains, or \
                     call `finish(summary)` to return your final response.]",
                ));
                continue;
            }

            break;
        }

        // 3-Phase Concurrent Tool Execution:
        // Phase 1: Read-only tools dispatched concurrently.
        // Phase 2: Mutation tools executed after Phase 1.
        // Phase 3: Shell commands (bash) barrier executed after Phase 2.
        // Phase 4: the explicit `finish` control-flow tool — partitioned out of
        // Phases 1–3 so it is *never* dispatched to the workspace executor;
        // Phases 1 → 2 → 3 always run before `finish` is evaluated.
        let mut phase1 = Vec::new();
        let mut phase2 = Vec::new();
        let mut phase3 = Vec::new();
        let mut phase4_finish = Vec::new();

        for (i, tc) in completion.tool_calls.iter().enumerate() {
            if tc.name == FINISH_TOOL_NAME {
                phase4_finish.push((i, tc));
            } else if is_phase1_read(&tc.name) {
                phase1.push((i, tc));
            } else if is_phase2_write(&tc.name) {
                phase2.push((i, tc));
            } else {
                phase3.push((i, tc));
            }
        }

        // Phase 1 (Concurrent Reads)
        let phase1_futures = phase1.into_iter().map(|(idx, tc)| async move {
            let outcome = match executor.execute(&tc.name, &tc.arguments).await {
                Ok(o) => o,
                Err(e) => ToolOutcome {
                    text: format!("Error: {e}"),
                },
            };
            (idx, tc, outcome)
        });
        let phase1_results = join_all(phase1_futures).await;

        // Phase 2 (File-Partitioned Concurrent Writes - Barrier 1)
        let workspace_dir = options.and_then(|o| o.workspace.as_deref());
        let phase2_results = execute_phase2_writes(executor, phase2, workspace_dir).await;

        // Phase 3 (Bash Barrier - Barrier 2)
        let mut phase3_results = Vec::with_capacity(phase3.len());
        for (idx, tc) in phase3 {
            let outcome = match executor.execute(&tc.name, &tc.arguments).await {
                Ok(o) => o,
                Err(e) => ToolOutcome {
                    text: format!("Error: {e}"),
                },
            };
            phase3_results.push((idx, tc, outcome));
        }

        // Deterministic Reassembly (Index-Sorted for KV Prefix Invariant)
        //
        // Any turn that executed at least one tool call resets the no-tool
        // nudge counter (the counter only counts *consecutive* no-tool turns).
        consecutive_no_tool_nudges = 0;

        let mut all_results = Vec::with_capacity(completion.tool_calls.len());
        all_results.extend(phase1_results);
        all_results.extend(phase2_results);
        all_results.extend(phase3_results);
        all_results.sort_by_key(|(idx, _, _)| *idx);

        let has_bash = completion.tool_calls.iter().any(|tc| tc.name == "bash");
        let mut deferred_advisory: Option<String> = None;
        for (_idx, tc, outcome) in all_results {
            tool_activity = true;
            record_verification_state(
                &mut unverified_code_edit,
                break_after_turn,
                tc,
                outcome.text.starts_with("Error:"),
            );

            let (raw_tool_content, image_attachment) =
                if outcome.text.starts_with("[IMAGE_ATTACHMENT:") {
                    if let Some(end_idx) = outcome.text.find(']') {
                        let header = &outcome.text[..end_idx];
                        let rest = outcome.text[end_idx + 1..].trim_start_matches(['\r', '\n']);
                        let data_url = header.strip_prefix("[IMAGE_ATTACHMENT:").unwrap_or("");
                        let data_url = data_url.split(":path:").next().unwrap_or("");
                        (rest.to_string(), Some(data_url.to_string()))
                    } else {
                        (outcome.text.clone(), None)
                    }
                } else {
                    (outcome.text.clone(), None)
                };

            let scratch_root = options.and_then(|o| o.scratch_root());
            let spill_threshold = options.and_then(|o| o.tool_spill_bytes);
            let (tool_content, spilled) = spill_tool_output(
                &raw_tool_content,
                &tc.id,
                scratch_root.as_deref(),
                spill_threshold,
            );
            if let Some((bytes, path)) = spilled {
                let _ = logger.append(serde_json::json!({
                    "type": "tool_output_spilled",
                    "sessionId": session_id,
                    "toolCallId": tc.id,
                    "toolName": tc.name,
                    "bytes": bytes,
                    "path": path.to_string_lossy(),
                }));
            }

            let sanitized_tool_content =
                sanitize_tool_delimiters(&tool_content, &open_tag, &close_tag);
            let tool_message_content = dedup_tool_content(&messages, tc, &sanitized_tool_content);
            messages.push(Message {
                role: "tool".into(),
                content: tool_message_content,
                tool_call_id: Some(tc.id.clone()),
                tool_calls: Vec::new(),
            });

            if let Some(data_url) = image_attachment {
                for m in &mut messages {
                    if m.role == "user" && m.content.starts_with("[Visual Inspection Attachment:") {
                        m.content =
                            "[Prior visual attachment cleared to stay within the 1-image limit]"
                                .to_string();
                    }
                }
                messages.push(Message {
                    role: "user".into(),
                    content: format!("[Visual Inspection Attachment: {data_url}]"),
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                });
            }

            logger
                .append(serde_json::json!({
                    "type": "tool_result",
                    "turn": turns,
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "is_error": outcome.text.starts_with("Error:"),
                    "output": tool_content,
                }))
                .map_err(RunnerError::Io)?;

            match loop_detector.record(&tc.name, &tc.arguments) {
                LoopState::Ok => {}
                LoopState::Advisory => {
                    deferred_advisory = Some(if tc.name == "read_file" {
                        "[System: Read Advisory] You have repeated the exact same read on this coordinate \
multiple times. If you are facing contradictory requirements across files or an architectural \
impasse, state your findings and ask for alignment rather than continuing to re-read."
                            .to_string()
                    } else {
                        "[System: Loop Advisory] You have repeated the same action sequence multiple times in a \
                         row. Change your approach or synthesize your findings."
                            .to_string()
                    });
                }
                LoopState::LoopDetected => {
                    return Err(RunnerError::LoopDetected {
                        name: tc.name.clone(),
                    });
                }
            }
        }

        // Phase 4: the explicit `finish` control-flow tool. The Phases 1→2→3
        // workspace calls above have all been executed and their `role: "tool"`
        // results appended in dispatch-index order *before* `finish` is
        // evaluated. `finish` itself was never dispatched to the executor
        // (partitioned out above) and never fed to the loop detector. We now
        // resolve it, appending its canned result(s) in dispatch-index order
        // so the turn ends on a clean `assistant(tool_calls)` +
        // `tool(results)` tail with no dangling, unanswered `tool_call`.
        if !phase4_finish.is_empty() {
            // The last `finish` call carries the report; when its `summary`
            // is empty/malformed, fall back to the turn's own synthesis.
            let resolved_summary = phase4_finish
                .last()
                .and_then(|(_, tc)| extract_finish_summary(&tc.arguments))
                .or_else(|| {
                    let ct = completion_text(&completion);
                    if ct.trim().is_empty() { None } else { Some(ct) }
                });

            if unverified_code_edit && !has_bash && !verification_nudge_sent {
                // One-shot verification intercept: verifiable code changed this
                // turn without a `bash` verification command. Defer `finish`
                // with a tool result telling the model to verify first; set the
                // latch so the duplicate end-of-turn advisory (below) is
                // skipped.
                verification_nudge_sent = true;
                let _ = logger.append(serde_json::json!({
                    "type": "finish_deferred",
                    "turn": turns,
                    "reason": "unverified_code_edit",
                }));
                append_finish_results(
                    &mut messages,
                    logger,
                    turns,
                    &phase4_finish,
                    "Finish deferred: verifiable code was modified without running \
                     verification via bash. Run verification now, then call finish(summary).",
                );
                continue;
            }

            if resolved_summary.is_none() && tool_activity && !empty_finish_rejected {
                // One-shot empty/malformed guard: reject the empty `finish`
                // once, asking for a non-empty grounded summary, then let the
                // model retry.
                empty_finish_rejected = true;
                let _ = logger.append(serde_json::json!({
                    "type": "finish_rejected",
                    "turn": turns,
                    "reason": "empty_summary",
                }));
                append_finish_results(
                    &mut messages,
                    logger,
                    turns,
                    &phase4_finish,
                    "Finish rejected: provide a non-empty grounded `summary` argument.",
                );
                continue;
            }

            // Clean terminal completion: acknowledge `finish`, land the report
            // as the final text, and break *before* the deferred loop advisory
            // and any trailing `user` nudge so `messages.json` ends on the
            // clean `assistant(tool_calls)` + `tool(results)` tail.
            append_finish_results(
                &mut messages,
                logger,
                turns,
                &phase4_finish,
                "Task completed.",
            );
            final_text = resolved_summary.unwrap_or_else(|| completion_text(&completion));
            break;
        }

        if let Some(advisory) = deferred_advisory {
            messages.push(msg("user", advisory));
        } else if unverified_code_edit && !has_bash && !verification_nudge_sent {
            // One-shot latch: set the instant the advisory is injected, so it
            // fires at most once per session even if the model keeps making
            // unverified code edits on subsequent turns.
            verification_nudge_sent = true;
            let _ = logger.append(serde_json::json!({
                "type": "verification_advisory",
                "turn": turns,
            }));
            messages.push(msg(
                "user",
                "[System: Code modified without verification command. Verify your changes via `bash` or conclude.]",
            ));
        }

        // End of turn: persist the sanitized history so a later dispatch can
        // resume it (subject to the 90% ceiling-aligned resume watermark on load).
        if let Some(state) = options.and_then(|o| o.state.as_ref()) {
            save_messages(state, session_id, &messages);
        }

        // Per-turn slot bookkeeping: record the turn's token footprint on the held lease
        // so the aggregate KV pool accurately reflects all active sessions.
        if let Some(ctx) = options.and_then(|o| o.lease_context.as_ref()) {
            let prompt_tokens = completion.metrics.prompt_tokens;
            let completion_tokens = completion.metrics.completion_tokens;
            let turn_tokens = prompt_tokens
                .map(|p| p as u32)
                .unwrap_or_else(|| history_tokens(&messages) as u32)
                .saturating_add(completion_tokens.map(|c| c as u32).unwrap_or(0));
            ctx.sem.update_context_tokens(&ctx.task_id, turn_tokens);
        }
    }

    // Record the final event.
    let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
    logger
        .append(serde_json::json!({
            "type": "final",
            "turn": turns,
            "status": status,
            "final_text": final_text,
            "duration_ms": elapsed_ms,
        }))
        .map_err(RunnerError::Io)?;

    // Normal completion: persist the final sanitized history so a later dispatch
    // resumes from here.
    if let Some(state) = options.and_then(|o| o.state.as_ref()) {
        save_messages(state, session_id, &messages);
    }

    Ok(SessionResult {
        final_text,
        turns,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Completion, EngineError, Metrics};
    use crate::runner::ToolError;
    use crate::state::StateDir;
    use async_trait::async_trait;
    use std::collections::VecDeque;

    /// A scripted completion. `None` `prompt_tokens` leaves the in-flight
    /// token tracking on its `chars / 4` approximation.
    fn comp(content: &str, tool_calls: Vec<ToolCall>, prompt_tokens: Option<u64>) -> Completion {
        let finish = if tool_calls.is_empty() {
            "stop"
        } else {
            "tool_calls"
        };
        Completion {
            content: content.into(),
            reasoning_content: None,
            tool_calls,
            finish_reason: Some(finish.into()),
            metrics: Metrics {
                ttft_ms: None,
                total_ms: 1.0,
                tokens_per_sec: None,
                prompt_tokens,
                cached_tokens: None,
                completion_tokens: None,
                reasoning_tokens: None,
            },
        }
    }

    fn tc(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            index: 0,
            id: id.into(),
            name: name.into(),
            arguments: args.into(),
        }
    }

    fn tool_schema(name: &str) -> ToolSchema {
        ToolSchema {
            name: name.into(),
            description: format!("{name} tool"),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    /// One recorded `chat` call: the exact messages and tool schemas it was
    /// invoked with.
    #[derive(Default, Clone, Debug)]
    struct RecordedCall {
        messages: Vec<Message>,
        tools: Vec<ToolSchema>,
    }

    /// A scripted engine that records calls via a shared recorder.
    struct RecordingEngine {
        responses: tokio::sync::Mutex<VecDeque<Completion>>,
        calls: tokio::sync::Mutex<Vec<RecordedCall>>,
    }

    impl RecordingEngine {
        fn new(responses: Vec<Completion>) -> Self {
            Self {
                responses: tokio::sync::Mutex::new(responses.into()),
                calls: tokio::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl ChatEngine for RecordingEngine {
        async fn chat(
            &self,
            messages: &[Message],
            tools: &[ToolSchema],
            _stream: bool,
            _reasoning_effort: Option<&str>,
        ) -> Result<Completion, EngineError> {
            self.calls.lock().await.push(RecordedCall {
                messages: messages.to_vec(),
                tools: tools.to_vec(),
            });
            self.responses
                .lock()
                .await
                .pop_front()
                .ok_or_else(|| EngineError::Malformed("no scripted response".into()))
        }
    }

    /// A mock executor that returns canned results in order.
    struct MockExecutor {
        results: tokio::sync::Mutex<VecDeque<String>>,
    }

    #[async_trait]
    impl ToolExecutor for MockExecutor {
        async fn execute(&self, name: &str, _args: &str) -> Result<ToolOutcome, ToolError> {
            let text = self
                .results
                .lock()
                .await
                .pop_front()
                .ok_or_else(|| ToolError::Execute {
                    name: name.to_string(),
                    message: "no scripted result".into(),
                })?;
            Ok(ToolOutcome { text })
        }
    }

    fn tmp_state() -> StateDir {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "castor_sess_{}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            n
        ));
        let state = StateDir::new(&dir);
        let _ = state.ensure();
        state
    }

    // (a) The ceiling synthesis turn passes the full tool schema list — the
    // tool prefix stays byte-identical to earlier turns (never `&[]`), so the
    // vLLM APC prompt prefix is preserved across the ceiling turn.
    #[tokio::test]
    async fn ceiling_turn_passes_tools_unchanged() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_a");
        let big_prompt = "x".repeat(3_700); // 925 tokens > 900 (90% of 1000)
        let engine = RecordingEngine::new(vec![
            comp(
                "I have the core evidence. Let me run a final verification pass.",
                Vec::new(),
                Some(950),
            ),
            comp("Done.", Vec::new(), None),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(VecDeque::new()),
        };

        let mut options = SessionOptions::with_state(state.clone());
        options.max_context = 1_000;

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            &big_prompt,
            &[tool_schema("bash")],
            Some(&options),
        )
        .await
        .unwrap();
        assert_eq!(res.status, "completed");
        assert_eq!(
            res.final_text,
            "I have the core evidence. Let me run a final verification pass."
        );

        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 1, "ceiling turn passes full tools");
        assert_eq!(calls[0].tools.len(), 1);
        assert_eq!(calls[0].tools[0].name, "bash");
        let events = logger.read_all();
        assert!(
            events.iter().any(|e| e["type"] == "context_ceiling"),
            "a context_ceiling event must be recorded: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (b) An announced-intent completion on a normal (non-ceiling) turn
    // completes cleanly without artificial string regex stalls.
    #[tokio::test]
    async fn announcement_does_not_stall_or_inject_nudges() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_b");
        let text = "I have the core evidence. Now let me run a final verification pass.";
        let engine = RecordingEngine::new(vec![comp(text, Vec::new(), None)]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(VecDeque::new()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("bash")],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.final_text, text);
        assert_eq!(res.turns, 1);

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (c) A duplicate byte-identical `read_file` result compresses the new
    // (tail) tool message into a compact pointer, while the first turn's
    // message content stays untouched.
    #[tokio::test]
    async fn duplicate_byte_identical_read_compresses_tail_only() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_c");
        let big = "x".repeat(300); // > 256 bytes
        let args = r#"{"path":"f.txt"}"#;
        let engine = RecordingEngine::new(vec![
            comp("", vec![tc("c1", "read_file", args)], None),
            comp("", vec![tc("c2", "read_file", args)], None),
            comp("Done.", Vec::new(), None),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(vec![big.clone(), big.clone()].into()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("read_file")],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();
        assert_eq!(res.status, "completed");

        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 3);
        // First read: full content.
        let first_result = calls[1]
            .messages
            .iter()
            .find(|m| m.role == "tool" && m.tool_call_id.as_deref() == Some("c1"))
            .expect("first tool result");
        assert_eq!(
            first_result.content, big,
            "first read must stay full-content"
        );
        // Second (tail) read: compressed to the pointer (the KV prefix keeps
        // the full first result, so only the new tail is deduplicated).
        let second_result = calls[2]
            .messages
            .iter()
            .find(|m| m.role == "tool" && m.tool_call_id.as_deref() == Some("c2"))
            .expect("second tool result");
        assert!(
            second_result
                .content
                .contains("[Unchanged: byte-identical to prior `read_file`"),
            "second identical read must be compressed, got: {}",
            second_result.content
        );
        assert!(
            !second_result.content.contains("bytes)"),
            "dedup pointer must not emit byte-count noise, got: {}",
            second_result.content
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // Pure gate: only a *successful* source-code mutation by the three file
    // tools counts as verifiable. Errors, non-mutating tools, non-source
    // extensions, `.scratch` paths, and malformed args are all rejected.
    #[test]
    fn is_verifiable_code_mutation_gate() {
        // Error result: never verifiable.
        assert!(!is_verifiable_code_mutation(
            "edit_file",
            r#"{"path":"src/lib.rs"}"#,
            true
        ));
        // Non-mutating tool: never verifiable even on a .rs path.
        assert!(!is_verifiable_code_mutation(
            "bash",
            r#"{"cmd":"echo hi"}"#,
            false
        ));
        assert!(!is_verifiable_code_mutation(
            "read_file",
            r#"{"path":"src/lib.rs"}"#,
            false
        ));
        // Source-code mutation: verifiable.
        assert!(is_verifiable_code_mutation(
            "edit_file",
            r#"{"path":"src/lib.rs"}"#,
            false
        ));
        assert!(is_verifiable_code_mutation(
            "write_file",
            r#"{"path":"src/main.py"}"#,
            false
        ));
        assert!(is_verifiable_code_mutation(
            "ast_replace",
            r#"{"path":"pkg/index.ts","pattern":"x"}"#,
            false
        ));
        // Non-source extensions: not verifiable.
        for p in [
            "README.md",
            "notes.txt",
            "data.json",
            "config.toml",
            "flow.yaml",
            "flow.yml",
            "paper.tex",
            "index.html",
            "site.css",
            "Cargo.lock",
        ] {
            assert!(
                !is_verifiable_code_mutation("write_file", &format!("{{\"path\":\"{p}\"}}"), false),
                "{p} must not be verifiable"
            );
        }
        // Extensionless file: not verifiable.
        assert!(!is_verifiable_code_mutation(
            "write_file",
            r#"{"path":"Makefile"}"#,
            false
        ));
        // A `.scratch` segment disqualifies even a source extension.
        assert!(!is_verifiable_code_mutation(
            "write_file",
            r#"{"path":".scratch/foo.rs"}"#,
            false
        ));
        assert!(!is_verifiable_code_mutation(
            "write_file",
            r#"{"path":"deep/.scratch/x.rs"}"#,
            false
        ));
        // Malformed args: not verifiable.
        assert!(!is_verifiable_code_mutation("edit_file", "not-json", false));
        assert!(!is_verifiable_code_mutation(
            "edit_file",
            r#"{"foo":"bar"}"#,
            false
        ));
        // A `.scratch` segment disqualifies even when a source extension also
        // appears in another segment.
        assert!(!is_verifiable_code_mutation(
            "write_file",
            r#"{"path":".scratch/archive/keep.rs"}"#,
            false
        ));
        // A source extension (`.sh`) in a non-`.scratch` path is verifiable.
        assert!(is_verifiable_code_mutation(
            "write_file",
            r#"{"path":"src/verifier.sh"}"#,
            false
        ));
    }

    // (A) Loop-proof refusal: the agent edits source, tries to stop; the
    // verification advisory fires at the end of the edit turn; the agent replies with text
    // and the session terminates immediately on that reply. `turns` is 2 (edit turn + final answer).
    #[tokio::test]
    async fn verification_nudge_fires_once_and_session_terminates() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_va");
        let engine = RecordingEngine::new(vec![
            comp(
                "Editing the module.",
                vec![tc(
                    "e1",
                    "edit_file",
                    r#"{"path":"src/lib.rs","content":"fn x(){}"}"#,
                )],
                None,
            ),
            comp("Applied the change.", Vec::new(), None),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(vec!["ok".to_string()].into()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("edit_file"), tool_schema("bash")],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(
            res.final_text, "Applied the change.",
            "session must terminate on the reply"
        );
        assert_eq!(
            res.turns, 2,
            "turn 1 (edit) + turn 2 (final answer) — no extra turns (never loops)"
        );
        assert_eq!(res.status, "completed");

        let events = logger.read_all();
        let advisory_count = events
            .iter()
            .filter(|e| e["type"] == "verification_advisory")
            .count();
        assert_eq!(
            advisory_count, 1,
            "the verification advisory must fire once at turn end: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (B) Non-code edits are ignored: editing a `.md`, `.toml`, and a
    // `.scratch/*.rs` file never arms the verification gate, so stopping after
    // them produces zero nudges.
    #[tokio::test]
    async fn non_code_edits_do_not_trigger_verification_nudge() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_vb");
        let engine = RecordingEngine::new(vec![
            comp(
                "Updating docs and scratch.",
                vec![
                    tc("m1", "write_file", r#"{"path":"README.md","content":"hi"}"#),
                    tc("m2", "edit_file", r#"{"path":"Cargo.toml","content":"x"}"#),
                    tc(
                        "m3",
                        "write_file",
                        r#"{"path":".scratch/foo.rs","content":"x"}"#,
                    ),
                ],
                None,
            ),
            comp("Done.", Vec::new(), None),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(
                vec!["ok".to_string(), "ok".to_string(), "ok".to_string()].into(),
            ),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("write_file"), tool_schema("edit_file")],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.final_text, "Done.");
        assert_eq!(res.turns, 2);
        let events = logger.read_all();
        let advisory_count = events
            .iter()
            .filter(|e| e["type"] == "verification_advisory")
            .count();
        assert_eq!(
            advisory_count, 0,
            "non-code edits must never trigger the advisory: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (C) A *failed* code edit is ignored: the tool returns `Error: ...`, so no
    // mutation actually occurred and the nudge does not fire.
    #[tokio::test]
    async fn failed_code_edit_does_not_trigger_verification_nudge() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_vc");
        let engine = RecordingEngine::new(vec![
            comp(
                "Editing the module.",
                vec![tc(
                    "e1",
                    "edit_file",
                    r#"{"path":"src/lib.rs","content":"x"}"#,
                )],
                None,
            ),
            comp("Done.", Vec::new(), None),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(
                vec!["Error: Target content not found".to_string()].into(),
            ),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("edit_file")],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.final_text, "Done.");
        assert_eq!(res.turns, 2);
        let events = logger.read_all();
        let advisory_count = events
            .iter()
            .filter(|e| e["type"] == "verification_advisory")
            .count();
        assert_eq!(
            advisory_count, 0,
            "a failed edit must not trigger the advisory: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (D) A code edit followed by a `bash` command in the batch clears the
    // unverified flag, so no advisory fires.
    #[tokio::test]
    async fn code_edit_then_bash_clears_verification_flag() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_vd");
        let engine = RecordingEngine::new(vec![
            comp(
                "Editing the module and checking.",
                vec![
                    tc("e1", "edit_file", r#"{"path":"src/lib.rs","content":"x"}"#),
                    tc("b1", "bash", r#"{"command":"cargo check"}"#),
                ],
                None,
            ),
            comp("Done.", Vec::new(), None),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(vec!["ok".to_string(), "ok".to_string()].into()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("edit_file"), tool_schema("bash")],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.final_text, "Done.");
        assert_eq!(res.turns, 2);
        let events = logger.read_all();
        let advisory_count = events
            .iter()
            .filter(|e| e["type"] == "verification_advisory")
            .count();
        assert_eq!(
            advisory_count, 0,
            "a post-edit bash command clears the flag: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (E) One-shot latch: two consecutive code-edit turns with no `bash`
    // between them. The verification advisory is injected on the first edit
    // turn only (`advisory_count == 1`) — the one-shot latch must prevent it
    // from re-firing at the end of the second edit turn, which would loop
    // the session on repeated identical nudges.
    #[tokio::test]
    async fn consecutive_unverified_code_edits_advisory_fires_exactly_once() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_ve");
        let engine = RecordingEngine::new(vec![
            comp(
                "Editing the module.",
                vec![tc(
                    "e1",
                    "edit_file",
                    r#"{"path":"src/lib.rs","content":"fn x(){}"}"#,
                )],
                None,
            ),
            comp(
                "Editing again.",
                vec![tc(
                    "e2",
                    "edit_file",
                    r#"{"path":"src/lib.rs","content":"fn y(){}"}"#,
                )],
                None,
            ),
            comp("Done.", Vec::new(), None),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(vec!["ok".to_string(), "ok".to_string()].into()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("edit_file"), tool_schema("bash")],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.final_text, "Done.");
        assert_eq!(
            res.turns, 3,
            "turn 1 (edit) + turn 2 (edit) + turn 3 (final answer) — no extra turns"
        );
        assert_eq!(res.status, "completed");

        let events = logger.read_all();
        let advisory_count = events
            .iter()
            .filter(|e| e["type"] == "verification_advisory")
            .count();
        assert_eq!(
            advisory_count, 1,
            "the verification advisory must fire exactly once across two consecutive \
             unverified edit turns: {events:?}"
        );

        // The advisory message must be present in the history seen on the
        // second edit turn (turn 2) and on the final turn (turn 3).
        let calls = engine.calls.lock().await;
        let advisory_marker = "[System: Code modified without verification";
        assert!(
            !calls[0]
                .messages
                .iter()
                .any(|m| m.content.contains(advisory_marker)),
            "first edit turn must not already carry the advisory"
        );
        assert!(
            calls[1]
                .messages
                .iter()
                .any(|m| m.content.contains(advisory_marker)),
            "the advisory must be injected at the end of the first edit turn"
        );
        assert!(
            calls[2]
                .messages
                .iter()
                .any(|m| m.content.contains(advisory_marker)),
            "the advisory must remain in context on the final turn"
        );
        assert_eq!(
            calls[2]
                .messages
                .iter()
                .filter(|m| m.content.contains(advisory_marker))
                .count(),
            1,
            "the advisory message must not be duplicated into the history"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (1) The explicit `finish` control-flow tool terminates the session in a
    // single turn: it is never dispatched to the executor (no workspace side
    // effect) and its non-empty `summary` argument becomes the final report.
    #[tokio::test]
    async fn finish_tool_terminates_session_with_summary() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_1");
        let engine = RecordingEngine::new(vec![comp(
            "",
            vec![tc(
                "f1",
                FINISH_TOOL_NAME,
                r#"{"summary":"All work done and verified."}"#,
            )],
            None,
        )]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(VecDeque::new()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("bash"), tool_schema(FINISH_TOOL_NAME)],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.status, "completed");
        assert_eq!(
            res.final_text, "All work done and verified.",
            "the `finish` summary argument becomes the final report"
        );
        assert_eq!(res.turns, 1);

        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 1, "the session must land on the `finish` turn");

        // `finish` is a control-flow tool: the executor must never run it, so
        // no scripted workspace result is consumed and no other call is made.
        let events = logger.read_all();
        assert!(
            events.iter().any(|e| {
                e["type"] == "tool_result"
                    && e["tool_call_id"] == "f1"
                    && e["output"] == "Task completed."
            }),
            "the `finish` call must be acknowledged with a canned result: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (2) A `finish` call with an empty / malformed `summary` (after real
    // prior tool activity) is rejected — a non-empty-summary request is
    // appended as the `finish` tool result — and the session continues rather
    // than terminating.
    #[tokio::test]
    async fn empty_tool_call_with_explicit_finish_nudges_and_continues() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_2");
        let engine = RecordingEngine::new(vec![
            // Real tool activity first: `tool_activity` becomes true, which is
            // the precondition for the empty-`finish` rejection guard.
            comp("", vec![tc("b1", "bash", r#"{"command":"echo hi"}"#)], None),
            comp(
                "",
                vec![tc("f1", FINISH_TOOL_NAME, r#"{"summary":""}"#)],
                None,
            ),
            comp(
                "",
                vec![tc("f2", FINISH_TOOL_NAME, r#"{"summary":"Done now."}"#)],
                None,
            ),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(vec!["ok".to_string()].into()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("bash"), tool_schema(FINISH_TOOL_NAME)],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.status, "completed");
        assert_eq!(
            res.final_text, "Done now.",
            "the retried non-empty summary wins"
        );
        assert_eq!(
            res.turns, 3,
            "turn 1 (bash) + turn 2 (empty finish, rejected) + turn 3 (good finish)"
        );

        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 3);
        // The rejection result must be in context on the retry turn.
        assert!(
            calls[2]
                .messages
                .iter()
                .any(|m| m.content.contains("provide a non-empty grounded")),
            "the retry turn must carry the rejection tool result: {calls:?}"
        );
        // One-shot latch: exactly one `finish_rejected` event across the run.
        let events = logger.read_all();
        let rejected = events
            .iter()
            .filter(|e| e["type"] == "finish_rejected")
            .count();
        assert_eq!(
            rejected, 1,
            "the empty-`finish` rejection must fire exactly once: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (3) A model that answers every turn with plain text (no tool calls, no
    // `finish`) gets nudged at most `MAX_NO_TOOL_NUDGES` times before the
    // session lands — it can never loop on repeated nudges.
    #[tokio::test]
    async fn empty_tool_call_nudge_is_bounded_and_terminates() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_3");
        // Three plain-text answers with no tool calls. `prompt_tokens` stays
        // well under the 90% in-flight ceiling (900 of 1000), so the nudge
        // path — not the ceiling — governs termination.
        let engine = RecordingEngine::new(vec![
            comp("Still working on it…", Vec::new(), Some(100)),
            comp("Still working on it…", Vec::new(), Some(100)),
            comp("Okay, I am done.", Vec::new(), Some(100)),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(VecDeque::new()),
        };
        let mut options = SessionOptions::with_state(state.clone());
        options.max_context = 1_000; // 90% ceiling = 900 > 100, so it never trips

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("bash"), tool_schema(FINISH_TOOL_NAME)],
            Some(&options),
        )
        .await
        .unwrap();

        assert_eq!(res.status, "completed");
        assert_eq!(
            res.final_text, "Okay, I am done.",
            "lands on the third plain turn"
        );
        assert_eq!(
            res.turns, 3,
            "turn 1 (nudge 1) + turn 2 (nudge 2) + turn 3 (land) — bounded, never loops"
        );

        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 3);
        let events = logger.read_all();
        let nudges = events
            .iter()
            .filter(|e| e["type"] == "no_tool_nudge")
            .count();
        assert_eq!(
            nudges, 2,
            "at most `MAX_NO_TOOL_NUDGES` (2) nudges, then the session lands: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (4) A `finish` issued on the same turn as an unverified code edit is
    // deferred once (verification nudge). After the model runs `bash` and
    // retries `finish`, the session completes.
    #[tokio::test]
    async fn finish_after_unverified_edit_defers_once_then_completes_after_bash() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_4");
        let engine = RecordingEngine::new(vec![
            comp(
                "Editing then stopping.",
                vec![
                    tc(
                        "e1",
                        "edit_file",
                        r#"{"path":"src/lib.rs","content":"fn x(){}"}"#,
                    ),
                    tc("f1", FINISH_TOOL_NAME, r#"{"summary":"Done."}"#),
                ],
                None,
            ),
            comp(
                "Verifying then stopping.",
                vec![
                    tc("b1", "bash", r#"{"command":"cargo check"}"#),
                    tc(
                        "f2",
                        FINISH_TOOL_NAME,
                        r#"{"summary":"Verified and done."}"#,
                    ),
                ],
                None,
            ),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(vec!["ok".to_string(), "ok".to_string()].into()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[
                tool_schema("edit_file"),
                tool_schema("bash"),
                tool_schema(FINISH_TOOL_NAME),
            ],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.status, "completed");
        assert_eq!(
            res.final_text, "Verified and done.",
            "the post-bash retry `finish` summary is the report"
        );
        assert_eq!(res.turns, 2);

        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 2);
        // The deferred `finish` on turn 1 must not land the session: the retry
        // turn carries the deferral tool result.
        assert!(
            calls[1]
                .messages
                .iter()
                .any(|m| m.content.contains("Run verification now")),
            "the retry turn must carry the deferral tool result: {calls:?}"
        );
        let events = logger.read_all();
        let deferred = events
            .iter()
            .filter(|e| e["type"] == "finish_deferred")
            .count();
        assert_eq!(
            deferred, 1,
            "the verification intercept must fire exactly once: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (5) A single turn mixing a phase-1 read, a phase-2 write, a phase-3
    // bash, and a `finish` executes the workspace calls in phase order
    // (read → write → bash) and lands on the `finish` summary — all in one
    // turn, `finish` partitioned out of the executor entirely.
    #[tokio::test]
    async fn batched_phases_1_2_3_and_finish_execute_in_order() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_5");
        let engine = RecordingEngine::new(vec![comp(
            "",
            vec![
                tc("r1", "read_file", r#"{"path":"src/lib.rs"}"#),
                tc(
                    "w1",
                    "write_file",
                    r#"{"path":"src/lib.rs","content":"fn y(){}"}"#,
                ),
                tc("b1", "bash", r#"{"command":"cargo check"}"#),
                tc("f1", FINISH_TOOL_NAME, r#"{"summary":"All phases ran."}"#),
            ],
            None,
        )]);
        // Phase order: phase 1 read → phase 2 write → phase 3 bash.
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(
                vec!["READ".to_string(), "WRITE".to_string(), "BASH".to_string()].into(),
            ),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[
                tool_schema("read_file"),
                tool_schema("write_file"),
                tool_schema("bash"),
                tool_schema(FINISH_TOOL_NAME),
            ],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.status, "completed");
        assert_eq!(res.final_text, "All phases ran.");
        assert_eq!(res.turns, 1, "all four calls resolve on a single turn");

        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 1, "a single turn");
        drop(calls);

        // The `tool_result` ledger events are appended in dispatch-index
        // order, so their sequence proves the phase ordering (read → write →
        // bash) and that `finish` (last) carried the canned result rather than
        // a workspace side effect.
        let events = logger.read_all();
        let results: Vec<(&str, &str, &str)> = events
            .iter()
            .filter(|e| e["type"] == "tool_result")
            .map(|e| {
                (
                    e["tool_call_id"].as_str().unwrap_or(""),
                    e["name"].as_str().unwrap_or(""),
                    e["output"].as_str().unwrap_or(""),
                )
            })
            .collect();
        assert_eq!(
            results,
            vec![
                ("r1", "read_file", "READ"),
                ("w1", "write_file", "WRITE"),
                ("b1", "bash", "BASH"),
                ("f1", FINISH_TOOL_NAME, "Task completed."),
            ],
            "workspace phases run read → write → bash, `finish` last with a canned \
             result (never dispatched to the executor): {results:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (6) On the 90% in-flight ceiling turn the model still emits a `finish`
    // plus a workspace tool: no tool is executed, every `tool_call` is closed
    // with a well-formed `role: "tool"` result (no dangling call on resume),
    // the `finish` summary becomes the report, and the turn is not nudged.
    #[tokio::test]
    async fn context_ceiling_90_percent_with_finish_extracts_summary_and_closes_tool_calls() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_6");
        let big_prompt = "x".repeat(3_700); // ~925 chars/4 → triggers the 90% gate at max_context=1000
        let engine = RecordingEngine::new(vec![comp(
            "done",
            vec![
                tc("f1", FINISH_TOOL_NAME, r#"{"summary":"Ceiling report."}"#),
                tc("b1", "bash", r#"{"command":"cargo check"}"#),
            ],
            Some(950),
        )]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(VecDeque::new()),
        };
        let mut options = SessionOptions::with_state(state.clone());
        options.max_context = 1_000;

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            &big_prompt,
            &[tool_schema("bash"), tool_schema(FINISH_TOOL_NAME)],
            Some(&options),
        )
        .await
        .unwrap();

        assert_eq!(res.status, "completed");
        assert_eq!(
            res.final_text, "Ceiling report.",
            "the `finish` summary is extracted on the ceiling turn"
        );
        assert_eq!(res.turns, 1);

        let calls = engine.calls.lock().await;
        assert_eq!(
            calls.len(),
            1,
            "the ceiling turn lands without another chat call"
        );

        // Every emitted `tool_call` must be answered by a `role: "tool"` result
        // so a resumed session carries no dangling, unanswered call.
        let events = logger.read_all();
        let finish_res = events
            .iter()
            .find(|e| e["type"] == "tool_result" && e["tool_call_id"] == "f1")
            .expect("the `finish` call must be closed");
        assert_eq!(finish_res["output"], "Task completed.");
        let bash_res = events
            .iter()
            .find(|e| e["type"] == "tool_result" && e["tool_call_id"] == "b1")
            .expect("the workspace `bash` call must be closed");
        assert_eq!(
            bash_res["output"],
            "Skipped: 90% working-context ceiling reached."
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (7) On the 90% in-flight ceiling turn the model answers with plain text
    // (no tool calls): the session lands immediately without injecting a
    // `no_tool_nudge` (the ceiling path bypasses the nudge).
    #[tokio::test]
    async fn context_ceiling_90_percent_with_plain_text_breaks_without_nudging() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_7");
        let big_prompt = "x".repeat(3_700);
        let engine = RecordingEngine::new(vec![comp("Final synthesis.", Vec::new(), Some(950))]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(VecDeque::new()),
        };
        let mut options = SessionOptions::with_state(state.clone());
        options.max_context = 1_000;

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            &big_prompt,
            &[tool_schema("bash"), tool_schema(FINISH_TOOL_NAME)],
            Some(&options),
        )
        .await
        .unwrap();

        assert_eq!(res.status, "completed");
        assert_eq!(res.final_text, "Final synthesis.");
        assert_eq!(res.turns, 1);

        let calls = engine.calls.lock().await;
        assert_eq!(
            calls.len(),
            1,
            "the plain-text ceiling turn must not trigger a nudge turn"
        );
        let events = logger.read_all();
        assert!(
            events.iter().any(|e| e["type"] == "context_ceiling"),
            "a context_ceiling event must be recorded: {events:?}"
        );
        assert!(
            !events.iter().any(|e| e["type"] == "no_tool_nudge"),
            "the ceiling turn must break without a `no_tool_nudge`: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (8) A `length` reasoning-ceiling turn in an explicit-`finish` session
    // runs the salvage path first (never nudged), and a `finish(summary)`
    // during salvage is adopted directly — no redundant empty-tools synthesis
    // call is made.
    #[tokio::test]
    async fn reasoning_ceiling_length_with_explicit_finish_runs_salvage_without_nudging() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_8");
        // Force the `length` finish reason on the ceiling turn.
        let mut ceiling = comp("partial thoughts", Vec::new(), None);
        ceiling.finish_reason = Some("length".into());
        let engine = RecordingEngine::new(vec![
            ceiling,
            comp(
                "",
                vec![tc(
                    "f1",
                    FINISH_TOOL_NAME,
                    r#"{"summary":"Salvaged finish report."}"#,
                )],
                None,
            ),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(VecDeque::new()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("bash"), tool_schema(FINISH_TOOL_NAME)],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(
            res.status, "reasoning_budget_exhausted",
            "the honest ceiling status is preserved"
        );
        assert!(
            res.final_text.contains("Salvaged finish report."),
            "the `finish` summary adopted during salvage becomes the report: {}",
            res.final_text
        );

        let calls = engine.calls.lock().await;
        assert_eq!(
            calls.len(),
            2,
            "ceiling turn + one salvage call; a `finish` during salvage must skip the \
             redundant empty-tools synthesis (no third call): {calls:?}"
        );
        let events = logger.read_all();
        assert!(
            !events.iter().any(|e| e["type"] == "no_tool_nudge"),
            "the reasoning-ceiling path must salvage, never nudge: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (9) A degenerate empty final after real tool activity in an
    // explicit-`finish` session runs the (empty-final) salvage path before any
    // nudge, and the salvage retry's text becomes the report.
    #[tokio::test]
    async fn degenerate_empty_final_with_explicit_finish_runs_salvage_without_nudging() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_9");
        let engine = RecordingEngine::new(vec![
            comp(
                "",
                vec![tc("b1", "bash", r#"{"command":"cargo check"}"#)],
                None,
            ),
            comp("", Vec::new(), None),
            comp("Degenerate salvage summary.", Vec::new(), None),
        ]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(vec!["ok".to_string()].into()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "work",
            &[tool_schema("bash"), tool_schema(FINISH_TOOL_NAME)],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        assert_eq!(res.status, "completed");
        assert_eq!(
            res.final_text, "Degenerate salvage summary.",
            "the empty-final salvage retry text becomes the report"
        );
        assert_eq!(res.turns, 2);

        let calls = engine.calls.lock().await;
        assert_eq!(
            calls.len(),
            3,
            "tool turn + empty final + one salvage retry"
        );
        let events = logger.read_all();
        assert!(
            events.iter().any(|e| e["type"] == "salvage"),
            "a salvage event must be recorded: {events:?}"
        );
        assert!(
            !events.iter().any(|e| e["type"] == "no_tool_nudge"),
            "the degenerate path must salvage, never nudge: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (10) A persisted session that ended on a clean `assistant(tool_calls)` +
    // `tool(results)` `finish` tail resumes with a well-formed message list
    // (every assistant `tool_call` answered by a matching `role: "tool"`
    // result, alternating) and the resumed turn continues from there.
    #[tokio::test]
    async fn resumed_session_after_finish_preserves_valid_message_alternation() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_fin_10");

        // Seed a persisted history exactly as the Phase-4 `finish` path would
        // land it: a `finish` tool call closed by its canned `role: "tool"`
        // result, with no dangling unanswered call.
        let prior = vec![
            msg("user", "work"),
            Message {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: vec![tc("f1", FINISH_TOOL_NAME, r#"{"summary":"Prior done."}"#)],
                tool_call_id: None,
            },
            Message {
                role: "tool".into(),
                content: "Task completed.".into(),
                tool_calls: Vec::new(),
                tool_call_id: Some("f1".into()),
            },
        ];
        crate::runner::context::save_messages(&state, logger.session_id(), &prior);

        // The resumed session immediately calls `finish` again (clean
        // termination) — one chat call. Its *pre-turn* message snapshot
        // includes the persisted history plus the new user prompt, which is
        // what we validate for valid alternation.
        let engine = RecordingEngine::new(vec![comp(
            "",
            vec![tc(
                "f2",
                FINISH_TOOL_NAME,
                r#"{"summary":"More work done."}"#,
            )],
            None,
        )]);
        let executor = MockExecutor {
            results: tokio::sync::Mutex::new(VecDeque::new()),
        };

        let res = run_session(
            &engine,
            &executor,
            &logger,
            "sys",
            "more work",
            &[tool_schema("bash"), tool_schema(FINISH_TOOL_NAME)],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();
        assert_eq!(res.status, "completed");
        assert_eq!(res.turns, 1, "the resumed session lands on its single turn");
        assert_eq!(res.final_text, "More work done.");

        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 1, "one chat call on the resumed session");
        let msgs = &calls[0].messages;

        // The persisted `finish` tail must be replayed verbatim (system is
        // re-prepended, the `role: "tool"` result keeps its `tool_call_id`),
        // and the resumed turn's new user prompt is appended last.
        assert!(msgs.iter().any(|m| m.role == "system"));
        // The new user prompt is present (the session resumed, not rolled over).
        assert!(
            msgs.iter()
                .any(|m| m.role == "user" && m.content == "more work"),
            "the new user prompt must be present in the resumed context: {msgs:?}"
        );
        // The persisted finish tool call and its result must both be present.
        assert!(
            msgs.iter().any(|m| {
                m.role == "assistant" && m.tool_calls.iter().any(|t| t.name == FINISH_TOOL_NAME)
            }),
            "the prior `finish` tool call must be replayed: {msgs:?}"
        );
        assert!(
            msgs.iter()
                .any(|m| { m.role == "tool" && m.tool_call_id.as_deref() == Some("f1") }),
            "the prior `finish` tool result (tool_call_id = f1) must be replayed: {msgs:?}"
        );

        // Every assistant message that carries `tool_calls` must be followed by
        // a `role: "tool"` message answering the first call — no dangling call.
        for (i, m) in msgs.iter().enumerate() {
            if m.role == "assistant" && !m.tool_calls.is_empty() {
                let first_id = m.tool_calls[0].id.clone();
                let answered = msgs.get(i + 1..).is_some_and(|tail| {
                    tail.iter().any(|r| {
                        r.role == "tool" && r.tool_call_id.as_deref() == Some(first_id.as_str())
                    })
                });
                assert!(
                    answered,
                    "assistant tool_call {first_id} at index {i} must be answered: {msgs:?}"
                );
            }
        }

        let _ = std::fs::remove_dir_all(state.root());
    }
}
