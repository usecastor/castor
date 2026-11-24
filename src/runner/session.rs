//! Autonomous multi-turn session loop and terminal outcome resolution.

use crate::engine::{Message, ToolCall, ToolSchema};
use crate::runner::budget::{LOOP_THRESHOLD, LOOP_WINDOW};
use crate::runner::context::{
    CONTEXT_CEILING_DIRECTIVE, history_tokens, in_flight_ceiling_tokens, in_flight_context_tokens,
    load_messages, prior_session_summary_block, save_messages, should_resume, watermark_tokens,
};
use crate::runner::events::EventLogger;
use crate::runner::loop_detector::{LoopDetector, LoopState};
use crate::runner::messages::{
    completion_text, is_severed_stop, metrics_json, msg, sanitize_tool_delimiters,
    strip_reasoning_tags,
};
use crate::runner::salvage::{
    is_reasoning_ceiling, run_degenerate_final_salvage, run_reasoning_ceiling_salvage,
};
use crate::runner::skeleton::spill_tool_output;
use crate::runner::watchdog::chat_with_watchdog;
use crate::runner::{
    ChatEngine, RunnerError, SessionOptions, SessionResult, ToolExecutor, ToolOutcome,
};

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

/// The one-time announced-intent continuation nudge for normal (non-ceiling)
/// turns.
const ANNOUNCED_INTENT_NUDGE: &str =
    "Continue and complete the action or final synthesis you just announced.";

/// The one-time announced-intent continuation nudge for the ceiling
/// synthesis turn: it must land a complete final synthesis, never another
/// transitional announcement.
const ANNOUNCED_INTENT_CEILING_NUDGE: &str = "[System: You stated an intent to act or verify without emitting a tool call or final report. \
     Output your complete final synthesis now.]";

/// Pure announced-intent detector. True when an assistant completion's text is
/// a short transitional announcement — it states an intent to act or verify
/// (e.g. "let me run a final verification pass") but carries no markdown
/// structure that would mark it as a finished deliverable. The final
/// sentence/paragraph must be the announcement (it may be prefixed by an
/// evidence summary, as in "I have the core evidence. Let me run a final
/// verification pass...").
pub(crate) fn is_announced_intent(text: &str) -> bool {
    const MAX_CHARS: usize = 600;
    const MAX_PARAGRAPHS: usize = 3;

    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.chars().count() > MAX_CHARS {
        return false;
    }
    // Markdown structure (headers, numbered/list bullets, fenced code) marks a
    // finished deliverable, not a transitional announcement.
    for line in trimmed.lines() {
        let l = line.trim_start();
        if l.starts_with(|c: char| c == '#' || c.is_ascii_digit()) || l.starts_with(['*', '-']) {
            return false;
        }
    }
    if trimmed.contains("```") {
        return false;
    }

    // Paragraphs are blank-line-separated blocks (a single-line announcement
    // is one paragraph even across several physical lines).
    let paragraphs: Vec<&str> = trimmed
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    if paragraphs.is_empty() || paragraphs.len() > MAX_PARAGRAPHS {
        return false;
    }

    // The *last sentence* of the final paragraph must carry the announcement
    // (it may be prefixed by an evidence summary), so a sentence that merely
    // ends with "let me ..." inside a longer paragraph is not counted. Split
    // on sentence boundaries and take the *last non-empty* fragment: a plain
    // `.split(['.', ...]).last()` returns the trailing empty segment (the text
    // after the final period), which would blank the check.
    let last_paragraph = paragraphs.last().expect("paragraphs non-empty");
    let last_sentence = last_paragraph
        .split(['.', '?', '!'])
        .map(str::trim)
        .rfind(|s| !s.is_empty())
        .unwrap_or("");
    let last_lower = last_sentence.to_lowercase();
    if last_lower.is_empty() {
        return false;
    }
    // The final sentence must *start* with an announcement phrase, so
    // "The verification passed." (an ending mention) is not an announcement.
    const OPENERS: [&str; 8] = [
        "let me ",
        "i will now ",
        "i'll now ",
        "let's run ",
        "let me run ",
        "i am going to ",
        "now let me ",
        "now i will ",
    ];
    OPENERS.iter().any(|o| last_lower.starts_with(o))
        || last_lower.contains("i'll run ")
        || last_lower.contains("i will run ")
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

/// The one-time verification nudge sent when the model mutates source code and
/// then tries to stop without running a verification command.
const VERIFICATION_NUDGE: &str = "[System: Source code was modified without a subsequent verification command. \
     Run the relevant build or test check via `bash` to verify your changes, or \
     confirm if no verification command applies.]";

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
            "[Unchanged: byte-identical to prior `{}` result earlier in this session \
             ({} bytes). Reuse the content already in context above.]",
            tc.name,
            sanitized.len()
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
    // is a strict one-shot latch: it arms the nudge exactly once for the session,
    // so the nudge can never fire twice and the session can never loop on it.
    let mut unverified_code_edit = false;
    let mut verification_nudge_sent = false;
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
        // Detect the announced-intent signal before `sanitized_content` is
        // moved into the appended assistant message.
        let announced_intent = is_announced_intent(&sanitized_content);
        messages.push(Message {
            role: "assistant".into(),
            content: sanitized_content,
            tool_calls: completion.tool_calls.clone(),
            tool_call_id: None,
        });

        // In-flight 90% ceiling landed: this is the synthesis turn. A bare
        // transitional announcement ("let me run a final verification pass")
        // must not end the session here, so the completion is followed by a
        // one-time continuation that is forced to output the complete final
        // synthesis. Slipped tool calls a non-compliant engine may emit on
        // this turn are never executed — they are appended as the assistant
        // message (matching what was sent to the engine) so the history stays
        // consistent, and the continuation still forces the synthesis.
        if break_after_turn {
            final_text = completion_text(&completion);
            if completion.tool_calls.is_empty() && announced_intent {
                let _ = logger.append(serde_json::json!({
                    "type": "announced_intent",
                    "turn": turns,
                    "ceiling": true,
                }));
                messages.push(msg("user", ANNOUNCED_INTENT_CEILING_NUDGE));
                match engine.chat(&messages, active_tools, true, effort).await {
                    Ok(cont) => {
                        // The synthesis turn is terminal: land on its content
                        // (any tool calls it still emitted are never executed).
                        final_text = completion_text(&cont);
                    }
                    Err(e) => {
                        tracing::warn!(
                            session = %session_id,
                            "ceiling synthesis continuation failed: {e}; landing on prior content"
                        );
                    }
                }
            }
            break;
        }

        if completion.tool_calls.is_empty() {
            // Compute the ceiling signal before moving `completion.content`.
            let ceiling_hit = is_reasoning_ceiling(&completion);

            // Capture the severed-stop signal (reasoning closed, but
            // the model stopped with no tool call and no real content) before
            // moving `completion`. This triggers a one-time continuation prompt
            // so the engine is given a chance to act after deliberating.
            let severed_stop = completion
                .finish_reason
                .as_deref()
                .map(|f| f == "stop")
                .unwrap_or(false)
                && completion.reasoning_content.is_some()
                && is_severed_stop(&completion.content, &close_tag);

            // Final content: `completion_text` falls back to the trimmed
            // `reasoning_content` (common when a reasoning model finishes
            // deliberation or emits its conclusion in thought tokens) instead
            // of discarding the trace.
            final_text = completion_text(&completion);

            // One-time continuation: the model announced an intent to act or
            // verify (a short transitional statement such as "let me run a
            // final verification pass") without emitting a tool call or a
            // final report. Treat it like a severed stop: nudge once to
            // continue acting; if the continuation emits tool calls the loop
            // continues naturally, otherwise we land on its content.
            if announced_intent {
                let _ = logger.append(serde_json::json!({
                    "type": "announced_intent",
                    "turn": turns,
                    "ceiling": false,
                }));
                let continuation_prompt = ANNOUNCED_INTENT_NUDGE;
                messages.push(msg("user", continuation_prompt));
                accounted_msgs = messages.len();
                match engine.chat(&messages, tools, true, effort).await {
                    Ok(cont) => {
                        if let Some(pt) = cont.metrics.prompt_tokens {
                            last_prompt_tokens = Some(pt as usize);
                        }
                        if cont.tool_calls.is_empty() {
                            // Terminal: land on the continuation's content (or
                            // its reasoning if the content is empty).
                            final_text = completion_text(&cont);
                            break;
                        }
                        // The model acted: append the assistant message with
                        // its tool calls so the loop continues and executes them.
                        messages.push(Message {
                            role: "assistant".into(),
                            content: strip_reasoning_tags(&cont.content, &open_tag, &close_tag),
                            tool_calls: cont.tool_calls.clone(),
                            tool_call_id: None,
                        });
                        let scratch_root = options.and_then(|o| o.scratch_root());
                        let spill_threshold = options.and_then(|o| o.tool_spill_bytes);
                        for tc in &cont.tool_calls {
                            let outcome = match executor.execute(&tc.name, &tc.arguments).await {
                                Ok(o) => o,
                                Err(e) => ToolOutcome {
                                    text: format!("Error: {e}"),
                                },
                            };
                            tool_activity = true;
                            record_verification_state(
                                &mut unverified_code_edit,
                                break_after_turn,
                                tc,
                                outcome.text.starts_with("Error:"),
                            );
                            let (tool_content, spilled) = spill_tool_output(
                                &outcome.text,
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
                            let _ = logger.append(serde_json::json!({
                                "type": "tool_result",
                                "turn": turns,
                                "tool_call_id": tc.id,
                                "name": tc.name,
                                "is_error": outcome.text.starts_with("Error:"),
                                "output": tool_content,
                            }));
                            // Sanitize embedded reasoning delimiters so a
                            // downstream tool-call parser never sees a valid
                            // delimiter inside this continuation's tool output
                            // (mirrors the main tool loop, DRY).
                            let sanitized_tool_content =
                                sanitize_tool_delimiters(&tool_content, &open_tag, &close_tag);
                            let tool_message_content =
                                dedup_tool_content(&messages, tc, &sanitized_tool_content);
                            messages.push(Message {
                                role: "tool".into(),
                                content: tool_message_content,
                                tool_call_id: Some(tc.id.clone()),
                                tool_calls: Vec::new(),
                            });
                        }
                        // Loop continues: `turns` was already incremented for
                        // this iteration, so the next iteration consumes a fresh
                        // budget unit for the follow-up assistant turn.
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!(
                            session = %session_id,
                            "continuation call after announced intent failed: {e}; \
                             landing on prior content"
                        );
                        break;
                    }
                }
            }

            // One-time continuation: the model deliberated (closed its reasoning
            // block) but stopped without a tool call and without a real deliverable.
            // Prompt it once to continue acting; if it then emits tool calls, the
            // loop continues naturally — otherwise we land on the new content.
            if severed_stop {
                let continuation_prompt = concat!(
                    "[System: Continuation] Your last turn closed with ",
                    "th_ink",
                    " reasoning but produced no tool call or final deliverable. \
                     If you still need to take an action (read, write, run a tool, \
                     or synthesize a final answer), do so now. If the task is \
                     already complete, reply with your final answer."
                );
                messages.push(msg("user", continuation_prompt));
                accounted_msgs = messages.len();
                match engine.chat(&messages, tools, true, effort).await {
                    Ok(cont) => {
                        if let Some(pt) = cont.metrics.prompt_tokens {
                            last_prompt_tokens = Some(pt as usize);
                        }
                        if cont.tool_calls.is_empty() {
                            // Terminal: land on the continuation's content (or its
                            // reasoning if the content is empty).
                            final_text = if !cont.content.trim().is_empty() {
                                cont.content
                            } else if let Some(rc) = cont
                                .reasoning_content
                                .as_deref()
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                            {
                                rc.to_string()
                            } else {
                                cont.content
                            };
                            break;
                        }
                        // The model acted: append the assistant message with its
                        // tool calls so the loop continues and executes them.
                        messages.push(Message {
                            role: "assistant".into(),
                            content: cont.content,
                            tool_calls: cont.tool_calls.clone(),
                            tool_call_id: None,
                        });
                        let scratch_root = options.and_then(|o| o.scratch_root());
                        let spill_threshold = options.and_then(|o| o.tool_spill_bytes);
                        for tc in &cont.tool_calls {
                            let outcome = match executor.execute(&tc.name, &tc.arguments).await {
                                Ok(o) => o,
                                Err(e) => ToolOutcome {
                                    text: format!("Error: {e}"),
                                },
                            };
                            tool_activity = true;
                            record_verification_state(
                                &mut unverified_code_edit,
                                break_after_turn,
                                tc,
                                outcome.text.starts_with("Error:"),
                            );
                            let (tool_content, spilled) = spill_tool_output(
                                &outcome.text,
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
                            let _ = logger.append(serde_json::json!({
                                "type": "tool_result",
                                "turn": turns,
                                "tool_call_id": tc.id,
                                "name": tc.name,
                                "is_error": outcome.text.starts_with("Error:"),
                                "output": tool_content,
                            }));
                            // Sanitize embedded reasoning delimiters so a
                            // downstream tool-call parser never sees a valid
                            // delimiter inside this continuation's tool output
                            // (mirrors the main tool loop, DRY).
                            let sanitized_tool_content =
                                sanitize_tool_delimiters(&tool_content, &open_tag, &close_tag);
                            let tool_message_content =
                                dedup_tool_content(&messages, tc, &sanitized_tool_content);
                            messages.push(Message {
                                role: "tool".into(),
                                content: tool_message_content,
                                tool_call_id: Some(tc.id.clone()),
                                tool_calls: Vec::new(),
                            });
                        }
                        // Loop continues: `turns` was already incremented for this
                        // iteration, so the next iteration consumes a fresh budget
                        // unit for the follow-up assistant turn.
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!(
                            session = %session_id,
                            "continuation call after severed stop failed: {e}; \
                             landing on prior content"
                        );
                        break;
                    }
                }
            }

            // Post-mutation verification nudge: the model mutated verifiable
            // source code (see [`is_verifiable_code_mutation`]) and is now
            // trying to stop without a subsequent `bash` verification command.
            // The `verification_nudge_sent` latch is set the instant the nudge
            // fires, so it can fire at most once per session — the model is
            // nudged, and if it still refuses to verify (a text-only reply) we
            // land on that reply and exit. The session therefore cannot loop
            // on this nudge.
            if !break_after_turn && !ceiling_hit && unverified_code_edit && !verification_nudge_sent
            {
                verification_nudge_sent = true;
                unverified_code_edit = false;
                let _ = logger.append(serde_json::json!({
                    "type": "verification_nudge",
                    "turn": turns,
                }));
                messages.push(msg("user", VERIFICATION_NUDGE));
                accounted_msgs = messages.len();
                match engine.chat(&messages, tools, true, effort).await {
                    Ok(cont) => {
                        if let Some(pt) = cont.metrics.prompt_tokens {
                            last_prompt_tokens = Some(pt as usize);
                        }
                        if cont.tool_calls.is_empty() {
                            // Terminal: the model confirmed (or refused) the
                            // verification; land on its content.
                            final_text = completion_text(&cont);
                            break;
                        }
                        // The model ran a verification command: execute it and
                        // continue the main loop so the result feeds the next
                        // turn. `unverified_code_edit` resets in the main loop
                        // on the `bash` call, so this nudge does not re-arm.
                        messages.push(Message {
                            role: "assistant".into(),
                            content: strip_reasoning_tags(&cont.content, &open_tag, &close_tag),
                            tool_calls: cont.tool_calls.clone(),
                            tool_call_id: None,
                        });
                        let scratch_root = options.and_then(|o| o.scratch_root());
                        let spill_threshold = options.and_then(|o| o.tool_spill_bytes);
                        for tc in &cont.tool_calls {
                            let outcome = match executor.execute(&tc.name, &tc.arguments).await {
                                Ok(o) => o,
                                Err(e) => ToolOutcome {
                                    text: format!("Error: {e}"),
                                },
                            };
                            tool_activity = true;
                            record_verification_state(
                                &mut unverified_code_edit,
                                break_after_turn,
                                tc,
                                outcome.text.starts_with("Error:"),
                            );
                            let (tool_content, spilled) = spill_tool_output(
                                &outcome.text,
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
                            let _ = logger.append(serde_json::json!({
                                "type": "tool_result",
                                "turn": turns,
                                "tool_call_id": tc.id,
                                "name": tc.name,
                                "is_error": outcome.text.starts_with("Error:"),
                                "output": tool_content,
                            }));
                            let sanitized_tool_content =
                                sanitize_tool_delimiters(&tool_content, &open_tag, &close_tag);
                            let tool_message_content =
                                dedup_tool_content(&messages, tc, &sanitized_tool_content);
                            messages.push(Message {
                                role: "tool".into(),
                                content: tool_message_content,
                                tool_call_id: Some(tc.id.clone()),
                                tool_calls: Vec::new(),
                            });
                        }
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!(
                            session = %session_id,
                            "continuation call after verification nudge failed: {e}; \
                             landing on prior content"
                        );
                        break;
                    }
                }
            }

            // Deliberation-ceiling salvage pass: the model stopped at its
            // reasoning ceiling on a non-tool turn. The honest status is kept,
            // the accumulated conversation history is preserved, the peer
            // engineer is empowered with full tools to inspect scratchpads or
            // verify test outputs, and a grounded status report is extracted.
            if ceiling_hit {
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

            // Degenerate final: empty/whitespace after real tool activity.
            if final_text.trim().is_empty() && tool_activity {
                let outcome =
                    run_degenerate_final_salvage(engine, logger, &mut messages, effort, turns)
                        .await?;
                final_text = outcome.final_text;
                status = outcome.status;
            }

            break;
        }

        // Execute each tool call.
        //
        // A loop-detection advisory emitted while the batch runs is collected
        // and pushed *after* the whole batch (below), so the
        // `[assistant(tool_calls), tool, tool, ...]` grouping is never split
        // by a `role: "user"` advisory mid-batch — splitting it would rewrite
        // the prompt shape and invalidate the KV prefix cache.
        let mut deferred_advisory: Option<String> = None;
        for tc in &completion.tool_calls {
            let outcome = match executor.execute(&tc.name, &tc.arguments).await {
                Ok(o) => o,
                Err(e) => ToolOutcome {
                    text: format!("Error: {e}"),
                },
            };
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

            // Append the tool result as a tool message. Sanitize any embedded
            // reasoning delimiters so a downstream tool-call parser never sees a
            // valid `<th_ink>`-style delimiter inside tool output. (The tag pair
            // is hoisted to function scope; see the pre-loop definition.)
            let sanitized_tool_content =
                sanitize_tool_delimiters(&tool_content, &open_tag, &close_tag);
            // Tail-only byte-identical deduplication: if an earlier turn already
            // produced the same result for the same (name, arguments), the new
            // tail message carries a compact pointer instead of a duplicate
            // payload (the earlier message is never mutated — the KV prefix
            // stays 100% intact).
            let tool_message_content = dedup_tool_content(&messages, tc, &sanitized_tool_content);
            messages.push(Message {
                role: "tool".into(),
                content: tool_message_content,
                tool_call_id: Some(tc.id.clone()),
                tool_calls: Vec::new(),
            });

            if let Some(data_url) = image_attachment {
                // The engine carries a 1-image limit: clear any prior visual
                // attachment user messages so multi-turn visual tasks never
                // exceed it, then append the fresh attachment.
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

            // Record the tool_result event.
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

            // Loop detection (catches single-action and multi-action periodic
            // cycles). The advisory is only collected here; it is pushed after
            // the whole batch completes (see `deferred_advisory` below) so the
            // `[assistant(tool_calls), tool, ...]` grouping stays intact.
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

        // Push the deferred loop advisory only after the entire tool batch has
        // been appended, keeping the tool results contiguous with the
        // assistant turn that produced them.
        if let Some(advisory) = deferred_advisory {
            messages.push(msg("user", advisory));
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
    #[derive(Default, Clone)]
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
        assert_eq!(res.final_text, "Done.");

        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 2, "ceiling turn + one forced synthesis");
        // Both the ceiling turn and the forced synthesis carry the full tool
        // list (never an empty slice): the byte-identical tool prefix is
        // preserved across the ceiling turn.
        for (idx, call) in calls.iter().enumerate() {
            assert_eq!(
                call.tools.len(),
                1,
                "call {idx} must pass tools unchanged, got {} tool schemas",
                call.tools.len()
            );
            assert_eq!(call.tools[0].name, "bash");
        }
        let events = logger.read_all();
        assert!(
            events.iter().any(|e| e["type"] == "context_ceiling"),
            "a context_ceiling event must be recorded: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| e["type"] == "announced_intent" && e["ceiling"] == true),
            "the ceiling announced-intent event must be recorded: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (b) An announced-intent completion on a normal (non-ceiling) turn
    // triggers a one-time continuation (the "Continue and complete..." nudge)
    // instead of terminating on the bare announcement.
    #[tokio::test]
    async fn announced_intent_triggers_one_time_continuation() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_b");
        let engine = RecordingEngine::new(vec![
            comp(
                "I have the core evidence. Now let me run a final verification pass.",
                Vec::new(),
                None,
            ),
            comp("Done.", Vec::new(), None),
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
            &[tool_schema("bash")],
            Some(&SessionOptions::with_state(state.clone())),
        )
        .await
        .unwrap();

        // The session did not terminate on the announcement; it landed on the
        // continuation's final synthesis.
        assert_eq!(res.final_text, "Done.");
        assert_eq!(res.turns, 1, "the continuation is one-time (same turn)");

        let events = logger.read_all();
        assert!(
            events
                .iter()
                .any(|e| e["type"] == "announced_intent" && e["ceiling"] == false),
            "a normal announced-intent event must be recorded: {events:?}"
        );

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
            second_result
                .content
                .contains(&format!("({} bytes)", big.len()))
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
    // verification nudge fires exactly ONCE; the agent replies with text and no
    // tool calls, and the session terminates immediately on that reply. `turns`
    // is 2 (the edit turn + the single nudge continuation turn) — the nudge
    // cannot re-fire, so the session can never loop on it.
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
            comp("Verified visually, no test needed", Vec::new(), None),
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
            res.final_text, "Verified visually, no test needed",
            "session must terminate on the refusal reply"
        );
        assert_eq!(
            res.turns, 2,
            "turn 1 (edit) + turn 2 (nudge continuation) — no third turn (never loops)"
        );
        assert_eq!(res.status, "completed");

        let events = logger.read_all();
        let nudge_count = events
            .iter()
            .filter(|e| e["type"] == "verification_nudge")
            .count();
        assert_eq!(
            nudge_count, 1,
            "the verification nudge must fire exactly once: {events:?}"
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
        // Two engine turns: the edit batch (turn 1) + the final text stop
        // (turn 2). The nudge never fires here (the point of this test), so
        // `turns` reflects the edit turn plus the ordinary final turn.
        assert_eq!(res.turns, 2);
        let events = logger.read_all();
        let nudge_count = events
            .iter()
            .filter(|e| e["type"] == "verification_nudge")
            .count();
        assert_eq!(
            nudge_count, 0,
            "non-code edits must never trigger the nudge: {events:?}"
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
        // Two engine turns: the failed-edit batch (turn 1) + the final text
        // stop (turn 2). The nudge never fires here (the point of this test),
        // so `turns` reflects the edit turn plus the ordinary final turn.
        assert_eq!(res.turns, 2);
        let events = logger.read_all();
        let nudge_count = events
            .iter()
            .filter(|e| e["type"] == "verification_nudge")
            .count();
        assert_eq!(
            nudge_count, 0,
            "a failed edit must not trigger the nudge: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }

    // (D) A code edit followed by a `bash` command clears the unverified flag,
    // so a subsequent stop does not trigger the nudge (the bash call *is* the
    // verification command).
    #[tokio::test]
    async fn code_edit_then_bash_clears_verification_flag() {
        let state = tmp_state();
        let logger = EventLogger::new(&state, "sess_vd");
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
            comp(
                "Checking it compiles.",
                vec![tc("b1", "bash", r#"{"command":"cargo check"}"#)],
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
        assert_eq!(res.turns, 3);
        let events = logger.read_all();
        let nudge_count = events
            .iter()
            .filter(|e| e["type"] == "verification_nudge")
            .count();
        assert_eq!(
            nudge_count, 0,
            "a post-edit bash command clears the flag: {events:?}"
        );

        let _ = std::fs::remove_dir_all(state.root());
    }
}
