//! Wall-clock timeout watchdog and token monitoring for chat dispatches.

use tracing::warn;

use crate::engine::{ChatOptions, Completion, EngineError, Message, Metrics, ToolSchema};
use crate::runner::ChatEngine;
use crate::runner::events::EventLogger;
use crate::runner::messages::msg;

/// Shared zero-invalidation stall retry: preserves the exact message prefix
/// and `reasoning_effort`, logs `turn_stall_retry`, and appends `nudge` at the
/// tail. When `turn_timeout_secs > 0` the retry call is itself wrapped in the
/// same wall-clock timeout as the primary call, so a hung retry is surfaced as
/// a typed `EngineError::Http { status: 408, .. }` (Request Timeout) instead of
/// blocking the session forever.
#[allow(clippy::too_many_arguments)] // shared single-call stall retry helper
async fn retry_with_nudge(
    engine: &dyn ChatEngine,
    messages: &[Message],
    tools: &[ToolSchema],
    effort: Option<&str>,
    chat_options: ChatOptions,
    turn_timeout_secs: u64,
    logger: &EventLogger,
    turn: u32,
    reason: &str,
    extra: serde_json::Value,
    nudge: &str,
) -> Result<Completion, EngineError> {
    let mut event = serde_json::json!({
        "type": "turn_stall_retry",
        "turn": turn,
        "reason": reason,
        "retry_effort": effort,
    });
    if let (Some(dst), Some(src)) = (event.as_object_mut(), extra.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    let _ = logger.append(event);
    let mut retry_messages = messages.to_vec();
    retry_messages.push(msg("user", nudge));
    let retry = engine.chat_with_options(&retry_messages, tools, true, effort, chat_options);
    if turn_timeout_secs > 0 {
        match tokio::time::timeout(std::time::Duration::from_secs(turn_timeout_secs), retry).await {
            Ok(r) => r,
            Err(_elapsed) => {
                warn!(
                    turn = turn,
                    "turn retry stalled on wall-clock timeout ({}s)", turn_timeout_secs
                );
                Err(EngineError::Http {
                    status: 408,
                    body: format!("Turn retry timed out after {turn_timeout_secs}s"),
                })
            }
        }
    } else {
        retry.await
    }
}

pub(crate) const REPETITION_RETRY_NUDGE: &str = "[System: Previous response entered a repetition loop. Stop repeating prior phrasing and immediately emit your next tool call or final response.]";

const REASONING_BUDGET_RETRY_NUDGE: &str = "[System: Previous deliberation exceeded the single-turn reasoning \
     budget. Be concise and emit your next tool call or final response \
     immediately.]";

/// Chat with watchdog: wraps `engine.chat` in a wall-clock timeout, aborts
/// and retries runaway repetition loops, and monitors reasoning token ceilings.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // each argument is a distinct watchdog input
pub(crate) async fn chat_with_watchdog(
    engine: &dyn ChatEngine,
    messages: &[Message],
    tools: &[ToolSchema],
    effort: Option<&str>,
    turn_timeout_secs: u64,
    max_turn_reasoning_tokens: u32,
    logger: &EventLogger,
    turn: u32,
) -> Result<Completion, EngineError> {
    chat_with_watchdog_bounded(
        engine,
        messages,
        tools,
        effort,
        turn_timeout_secs,
        max_turn_reasoning_tokens,
        None,
        logger,
        turn,
    )
    .await
}

/// Bounded variant of [`chat_with_watchdog`] forwarding an optional `max_tokens`
/// completion ceiling and mid-stream `max_turn_reasoning_tokens` cutoff.
#[allow(clippy::too_many_arguments)] // each argument is a distinct watchdog input
pub(crate) async fn chat_with_watchdog_bounded(
    engine: &dyn ChatEngine,
    messages: &[Message],
    tools: &[ToolSchema],
    effort: Option<&str>,
    turn_timeout_secs: u64,
    max_turn_reasoning_tokens: u32,
    max_tokens: Option<u32>,
    logger: &EventLogger,
    turn: u32,
) -> Result<Completion, EngineError> {
    let chat_options = ChatOptions {
        max_tokens,
        max_turn_reasoning_tokens: Some(max_turn_reasoning_tokens),
    };

    // 1. Issue the primary call, optionally under a wall-clock timeout.
    let first = if turn_timeout_secs > 0 {
        match tokio::time::timeout(
            std::time::Duration::from_secs(turn_timeout_secs),
            engine.chat_with_options(messages, tools, true, effort, chat_options),
        )
        .await
        {
            Ok(r) => r,
            Err(_elapsed) => {
                warn!(
                    turn = turn,
                    "turn stalled on wall-clock timeout ({}s); retrying once (prefix preserved, effort = {effort:?})",
                    turn_timeout_secs
                );
                return retry_with_nudge(
                    engine,
                    messages,
                    tools,
                    effort,
                    chat_options,
                    turn_timeout_secs,
                    logger,
                    turn,
                    "timeout",
                    serde_json::json!({ "turn_timeout_secs": turn_timeout_secs }),
                    "[System: Previous turn timed out. Do not deliberate; emit your next tool call or final response immediately.]",
                )
                .await;
            }
        }
    } else {
        engine
            .chat_with_options(messages, tools, true, effort, chat_options)
            .await
    };

    let completion = match first {
        Ok(c) => c,
        Err(EngineError::RepetitionLoop {
            kind,
            pattern,
            count,
        }) => {
            warn!(
                turn = turn,
                kind = %kind,
                count = count,
                "turn hit runaway repetition loop; retrying once (prefix preserved, effort = {effort:?})"
            );
            return retry_with_nudge(
                engine,
                messages,
                tools,
                effort,
                chat_options,
                turn_timeout_secs,
                logger,
                turn,
                "repetition_loop",
                serde_json::json!({
                    "kind": kind,
                    "pattern": pattern,
                    "count": count,
                }),
                REPETITION_RETRY_NUDGE,
            )
            .await;
        }
        Err(EngineError::ReasoningBudgetExceeded { tokens, limit }) => {
            warn!(
                turn = turn,
                reasoning_tokens = tokens,
                limit = limit,
                "turn aborted mid-stream at reasoning-token budget ({}); retrying once (prefix preserved, effort = {effort:?})",
                limit
            );
            let retry_res = retry_with_nudge(
                engine,
                messages,
                tools,
                effort,
                chat_options,
                turn_timeout_secs,
                logger,
                turn,
                "reasoning_token_budget",
                serde_json::json!({
                    "reasoning_tokens": tokens,
                    "max_turn_reasoning_tokens": limit,
                    "mid_stream_abort": true,
                }),
                REASONING_BUDGET_RETRY_NUDGE,
            )
            .await;
            return match retry_res {
                Err(EngineError::ReasoningBudgetExceeded {
                    tokens: retry_tokens,
                    ..
                }) => Ok(Completion {
                    content: String::new(),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    finish_reason: Some("length".into()),
                    metrics: Metrics {
                        reasoning_tokens: Some(retry_tokens),
                        ..Default::default()
                    },
                }),
                other => other,
            };
        }
        Err(err) => return Err(err),
    };

    // 2. Reasoning-token budget stall: `length` finish and the model burned at
    //    least the budget in reasoning tokens (even if tool calls were pending).
    let stalled = max_turn_reasoning_tokens > 0
        && completion.finish_reason.as_deref() == Some("length")
        && completion.metrics.reasoning_tokens.unwrap_or(0) >= u64::from(max_turn_reasoning_tokens);

    if stalled {
        warn!(
            turn = turn,
            reasoning_tokens = completion.metrics.reasoning_tokens,
            "turn stalled at reasoning-token budget ({}); retrying once (prefix preserved, effort = {effort:?})",
            max_turn_reasoning_tokens
        );
        return retry_with_nudge(
            engine,
            messages,
            tools,
            effort,
            chat_options,
            turn_timeout_secs,
            logger,
            turn,
            "reasoning_token_budget",
            serde_json::json!({
                "reasoning_tokens": completion.metrics.reasoning_tokens,
                "max_turn_reasoning_tokens": max_turn_reasoning_tokens,
            }),
            REASONING_BUDGET_RETRY_NUDGE,
        )
        .await;
    }

    Ok(completion)
}
