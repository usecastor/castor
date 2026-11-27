//! Token counting, in-flight ceilings, and session message persistence.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::engine::{Message, ToolCall};
use crate::state::StateDir;

use super::messages::strip_think_tags;

pub(crate) const CONTEXT_CEILING_DIRECTIVE: &str = "[System: Working context budget has reached 90%. Do not execute further workspace tools. \
     Call finish(summary) or synthesize your final conclusions and response immediately.]";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedMessage {
    pub(crate) role: String,
    pub(crate) content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) tool_calls: Vec<PersistedToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedToolCall {
    pub(crate) index: usize,
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

impl From<&PersistedMessage> for Message {
    fn from(p: &PersistedMessage) -> Self {
        Message {
            role: p.role.clone(),
            content: p.content.clone(),
            tool_calls: p
                .tool_calls
                .iter()
                .map(|t| ToolCall {
                    index: t.index,
                    id: t.id.clone(),
                    name: t.name.clone(),
                    arguments: t.arguments.clone(),
                })
                .collect(),
            tool_call_id: p.tool_call_id.clone(),
        }
    }
}

pub(crate) fn to_persisted(m: &Message) -> PersistedMessage {
    let content = if m.role == "assistant" {
        strip_think_tags(&m.content)
    } else {
        m.content.clone()
    };
    PersistedMessage {
        role: m.role.clone(),
        content,
        tool_calls: m
            .tool_calls
            .iter()
            .map(|t| PersistedToolCall {
                index: t.index,
                id: t.id.clone(),
                name: t.name.clone(),
                arguments: t.arguments.clone(),
            })
            .collect(),
        tool_call_id: m.tool_call_id.clone(),
    }
}

pub(crate) fn session_messages_path(state: &StateDir, session_id: &str) -> PathBuf {
    state.sessions().join(session_id).join("messages.json")
}

pub(crate) fn save_messages(state: &StateDir, session_id: &str, messages: &[Message]) {
    let path = session_messages_path(state, session_id);
    let sanitized: Vec<PersistedMessage> = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(to_persisted)
        .collect();
    if let Some(parent) = path.parent()
        && let Err(e) = fs::create_dir_all(parent)
    {
        warn!(session = %session_id, "failed to create session history dir: {e}");
        return;
    }
    match serde_json::to_vec_pretty(&sanitized) {
        Ok(bytes) => {
            if let Err(e) = fs::write(&path, bytes) {
                warn!(session = %session_id, "failed to persist session history: {e}");
            }
        }
        Err(e) => {
            warn!(session = %session_id, "failed to serialize session history: {e}");
        }
    }
}

pub(crate) fn load_messages(state: &StateDir, session_id: &str) -> Option<Vec<Message>> {
    let path = session_messages_path(state, session_id);
    let bytes = fs::read(&path).ok()?;
    let persisted: Vec<PersistedMessage> = serde_json::from_slice(&bytes).ok()?;
    Some(persisted.iter().map(Message::from).collect())
}

pub(crate) fn approx_tokens(text: &str) -> usize {
    text.chars().count() / 4
}

pub(crate) fn history_tokens(messages: &[Message]) -> usize {
    messages.iter().map(|m| approx_tokens(&m.content)).sum()
}

/// Resume watermark for a persisted session: 90% of `max_context` — the
/// same fraction the in-flight ceiling uses (`in_flight_ceiling_tokens`),
/// so a warm session at any level below the in-flight ceiling resumes its
/// byte-identical KV prefix instead of being rolled over.
pub(crate) fn watermark_tokens(max_context: usize) -> usize {
    in_flight_ceiling_tokens(max_context)
}

/// The last non-empty assistant response in a persisted history (scanned
/// backwards, so the most recent substantive reply wins), reasoning tags
/// stripped. `None` when the history holds no such message.
pub(crate) fn last_non_empty_assistant(history: &[Message]) -> Option<String> {
    // Scan backwards, *skipping* whitespace-only assistant messages (including
    // ones that become empty once reasoning tags are stripped) until the most
    // recent substantive reply is found. Filtering the single `find` result
    // would stop at a recent-but-empty assistant and never reach the one
    // before it.
    history
        .iter()
        .rev()
        .filter(|m| m.role == "assistant")
        .map(|m| strip_think_tags(&m.content))
        .find(|t| !t.is_empty())
}

/// The file-mutating tools whose successful results count as ground-truth
/// mutations (the same set the post-mutation verification gate watches).
fn is_file_mutating_tool(name: &str) -> bool {
    matches!(name, "edit_file" | "write_file" | "ast_replace")
}

/// Extract the `"path"` argument from a tool call's JSON `arguments`.
fn tool_call_path(arguments: &str) -> Option<String> {
    let v = serde_json::from_str::<serde_json::Value>(arguments).ok()?;
    Some(v.get("path")?.as_str()?.to_string())
}

/// Deterministic ground-truth list of file paths the prior session *actually*
/// mutated: scans `history` for `edit_file` / `write_file` / `ast_replace`
/// tool calls whose corresponding `role: "tool"` result did **not** start with
/// `Error` / `error` (a failed edit changed nothing on disk and is excluded).
/// Paths are collected in first-seen order, de-duplicated, and capped at 20.
/// This anchors a rolled-over session in real tool state so it cannot
/// hallucinate which files were modified.
fn mutated_file_paths(history: &[Message]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut paths = Vec::new();
    for (i, m) in history.iter().enumerate() {
        if m.role != "assistant" || m.tool_calls.is_empty() {
            continue;
        }
        for tc in &m.tool_calls {
            if !is_file_mutating_tool(&tc.name) {
                continue;
            }
            // The matching `role: "tool"` result carries the same
            // `tool_call_id` and appears after the assistant message.
            let result = match history.get(i + 1..) {
                Some(tail) => tail.iter().find(|r| {
                    r.role == "tool" && r.tool_call_id.as_deref() == Some(tc.id.as_str())
                }),
                None => None,
            };
            // A failed result (leading `Error` / `error`) did not mutate the
            // file, so it is excluded from the receipt.
            if result
                .is_some_and(|r| r.content.starts_with("Error") || r.content.starts_with("error"))
            {
                continue;
            }
            let Some(path) = tool_call_path(&tc.arguments) else {
                continue;
            };
            if seen.insert(path.clone()) && paths.len() < 20 {
                paths.push(path);
            }
        }
    }
    paths
}

/// Compact `[Prior Session Summary]` block prepended to the new tail user
/// prompt when a persisted history rolls over over the resume ceiling, so
/// the rolled session keeps the prior run's conclusions.
///
/// The block carries, at the top, a grounded `Mutated files:` receipt (the
/// ground-truth list of source paths the prior session successfully modified,
/// from [`mutated_file_paths`]) above the last non-empty assistant synthesis.
/// `None` when the history holds no substantive assistant reply (the receipt
/// is only emitted alongside a real summary).
pub(crate) fn prior_session_summary_block(history: &[Message]) -> Option<String> {
    let text = last_non_empty_assistant(history)?;
    let mut block = String::from("[Prior Session Summary]\n");
    let mutated = mutated_file_paths(history);
    if !mutated.is_empty() {
        block.push_str("Mutated files: ");
        block.push_str(&mutated.join(", "));
        block.push('\n');
    }
    block.push_str(&text);
    block.push('\n');
    Some(block)
}

pub(crate) fn in_flight_ceiling_tokens(max_context: usize) -> usize {
    (max_context as u64 * 90 / 100) as usize
}

/// Conservative token estimate for newly appended messages not yet measured by
/// an engine `usage.prompt_tokens` report (code/JSON tokenizes closer to ~2.8
/// chars/token than 4 chars/token, and includes tool-call arguments).
pub(crate) fn uncounted_tail_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| {
            let content_chars = m.content.chars().count();
            let tc_chars: usize = m
                .tool_calls
                .iter()
                .map(|tc| tc.name.len() + tc.arguments.chars().count() + 16)
                .sum();
            (content_chars + tc_chars).div_ceil(3)
        })
        .sum()
}

/// Effective in-flight context footprint in tokens.
///
/// Prefers the last engine-reported `prompt_tokens` (exact upstream tokenizer
/// count for `messages[..accounted_msgs]` + tool schemas + chat template) plus
/// a conservative token count for any messages appended since that call, falling
/// back to `history_tokens` only before the first engine usage report.
pub(crate) fn in_flight_context_tokens(
    messages: &[Message],
    accounted_msgs: usize,
    last_prompt_tokens: Option<usize>,
) -> usize {
    match last_prompt_tokens {
        Some(pt) => {
            let tail = messages.get(accounted_msgs..).unwrap_or(&[]);
            pt.saturating_add(uncounted_tail_tokens(tail))
        }
        None => history_tokens(messages),
    }
}

pub fn resumed_history_tokens(state: &StateDir, session_id: &str, max_context: Option<u32>) -> u32 {
    let Some(mc) = max_context else {
        return 0;
    };
    if mc == 0 {
        return 0;
    }
    let Some(history) = load_messages(state, session_id) else {
        return 0;
    };
    if history.is_empty() {
        return 0;
    }
    let wm = in_flight_ceiling_tokens(mc as usize);
    if history_tokens(&history) <= wm {
        history_tokens(&history) as u32
    } else {
        0
    }
}

/// Resume decision for a persisted history: the approximate token cost of
/// `history` plus the new user prompt must stay within `watermark_tokens`
/// (90% of `max_context`, aligned with the in-flight ceiling). `0`
/// `max_context` disables the check (always resume).
pub(crate) fn should_resume(
    history: &[Message],
    new_user_prompt: &str,
    max_context: usize,
) -> bool {
    if max_context == 0 {
        return true;
    }
    let wm = watermark_tokens(max_context);
    let est = history_tokens(history) + approx_tokens(new_user_prompt);
    est <= wm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::messages::msg;

    fn tmp_state() -> StateDir {
        let dir = std::env::temp_dir().join(format!("castor_test_ctx_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        StateDir::new(&dir)
    }

    fn tc(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            index: 0,
            id: id.into(),
            name: name.into(),
            arguments: args.into(),
        }
    }

    #[test]
    fn save_and_load_round_trips_sanitized_non_system_history() {
        let state = tmp_state();
        let messages = vec![
            msg("system", "SYS"),
            msg("user", "do the thing"),
            Message {
                role: "assistant".into(),
                content: "<think> reasoning 42 </think> the answer is 42".into(),
                tool_calls: vec![tc("c1", "bash", r#"{"cmd":"ls"}"#)],
                tool_call_id: None,
            },
            Message {
                role: "tool".into(),
                content: "file1".into(),
                tool_calls: Vec::new(),
                tool_call_id: Some("c1".into()),
            },
        ];
        save_messages(&state, "roundtrip", &messages);

        let path = state.sessions().join("roundtrip").join("messages.json");
        assert!(path.exists(), "messages.json must be written: {path:?}");

        let loaded = load_messages(&state, "roundtrip").expect("history must load");
        assert_eq!(loaded.len(), 3, "system must be dropped; got {loaded:?}");
        assert!(!loaded.iter().any(|m| m.role == "system"));

        let assistant = loaded.iter().find(|m| m.role == "assistant").unwrap();
        assert!(!assistant.content.contains("<think>"));
        assert!(assistant.content.contains("42"));
        assert_eq!(assistant.tool_calls.len(), 1);
        assert_eq!(assistant.tool_calls[0].name, "bash");
        assert_eq!(assistant.tool_calls[0].id, "c1");

        let tool = loaded.iter().find(|m| m.role == "tool").unwrap();
        assert_eq!(tool.tool_call_id.as_deref(), Some("c1"));
        assert_eq!(tool.content, "file1");

        let _ = std::fs::remove_dir_all(state.root());
    }

    #[test]
    fn load_messages_absent_file_is_none() {
        let state = tmp_state();
        let loaded = load_messages(&state, "never_saved");
        assert!(loaded.is_none());
        let _ = std::fs::remove_dir_all(state.root());
    }

    #[test]
    fn watermark_within_budget_resumes() {
        let history = vec![msg("user", "a".repeat(400))];
        assert!(
            should_resume(&history, &"b".repeat(100), 1000),
            "history within the 90% ceiling-aligned watermark must resume"
        );
    }

    #[test]
    fn watermark_over_budget_rolls_over() {
        let history = vec![msg("user", "a".repeat(400))];
        assert!(
            !should_resume(&history, &"b".repeat(100), 100),
            "history over the 90% ceiling-aligned watermark must roll over"
        );
    }

    #[test]
    fn watermark_aligns_with_in_flight_ceiling() {
        // The resume watermark must equal the in-flight ceiling at every
        // `max_context` value so no warm session between 75% and 90% is wiped.
        for max_context in [0, 1, 79, 100, 150_000] {
            assert_eq!(
                watermark_tokens(max_context),
                in_flight_ceiling_tokens(max_context),
                "watermark must align with the in-flight ceiling at {max_context}"
            );
        }
        // A history between 75% and 90% of max_context resumes (not rolled over).
        // 1800 chars = 450 tokens: 45% of 1000 < 900 watermark, and 75% of 1000
        // would have been 750 — the distinguishing case is 80%: 800 chars =
        // 200 tokens against max_context = 250 (90% = 225, old 75% = 187).
        let history = vec![msg("user", "a".repeat(800))];
        assert!(
            should_resume(&history, "p", 250),
            "a warm session between the old 75% and the 90% ceiling must resume"
        );
    }

    #[test]
    fn prior_session_summary_block_extracts_last_non_empty_assistant() {
        let history = vec![
            msg("user", "do it"),
            Message {
                role: "assistant".into(),
                content: "first attempt".into(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            msg("user", "again"),
            Message {
                role: "assistant".into(),
                content: "second answer".into(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            msg("tool", "x"),
        ];
        let block = prior_session_summary_block(&history).expect("summary block");
        assert_eq!(block, "[Prior Session Summary]\nsecond answer\n");
        assert_eq!(
            last_non_empty_assistant(&history).as_deref(),
            Some("second answer")
        );
    }

    #[test]
    fn prior_session_summary_block_strips_reasoning_tags_and_skips_empty() {
        let open = concat!("<th", "ink>");
        let close = concat!("</th", "ink>");
        let history = vec![
            msg("user", "u"),
            Message {
                role: "assistant".into(),
                content: format!("{open} deliberation {close} the conclusion"),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            msg("user", "v"),
            Message {
                role: "assistant".into(),
                content: "   ".into(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
        ];
        let block = prior_session_summary_block(&history).expect("summary block");
        assert_eq!(block, "[Prior Session Summary]\nthe conclusion\n");
        // An all-empty / no-assistant history yields no block.
        assert!(prior_session_summary_block(&[msg("user", "u")]).is_none());
        let only_empty = vec![
            msg("user", "u"),
            Message {
                role: "assistant".into(),
                content: "  ".into(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
        ];
        assert!(prior_session_summary_block(&only_empty).is_none());
    }

    #[test]
    fn watermark_disabled_when_max_context_zero() {
        let history = vec![msg("user", "a".repeat(1_000_000))];
        assert!(
            should_resume(&history, "prompt", 0),
            "zero max_context disables the watermark — always resume"
        );
    }

    #[test]
    fn in_flight_context_tokens_prefers_engine_prompt_tokens_plus_tail() {
        let messages = vec![
            msg("system", "sys"),
            msg("user", "u"),
            msg("assistant", "a"),
            msg("tool", "x".repeat(30)),
        ];
        assert_eq!(in_flight_ceiling_tokens(150_000), 135_000);
        // Without engine usage, falls back to history_tokens.
        assert_eq!(
            in_flight_context_tokens(&messages, 0, None),
            history_tokens(&messages)
        );
        // With engine prompt_tokens=1000 covering the first 3 messages, adds only the 30-char tail (10 tokens).
        assert_eq!(in_flight_context_tokens(&messages, 3, Some(1000)), 1010);
    }

    // (E) Grounded rollover receipt: a successful `edit_file` path is included
    // in the `Mutated files:` receipt; a failed `edit_file` result (leading
    // `Error`) changed nothing on disk and is excluded. The receipt is anchored
    // above the last assistant synthesis, never after it.
    #[test]
    fn prior_session_summary_block_grounded_mutated_files_receipt() {
        let history = vec![
            msg("user", "please fix both files"),
            // Assistant turn issuing two edit_file calls.
            Message {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: vec![
                    tc("e1", "edit_file", r#"{"path":"src/a.rs","content":"x"}"#),
                    tc("e2", "edit_file", r#"{"path":"src/b.rs","content":"y"}"#),
                ],
                tool_call_id: None,
            },
            // The matching tool results: `e1` succeeded, `e2` failed.
            Message {
                role: "tool".into(),
                content: "Applied edit to src/a.rs".into(),
                tool_calls: Vec::new(),
                tool_call_id: Some("e1".into()),
            },
            Message {
                role: "tool".into(),
                content: "Error: Target content not found".into(),
                tool_calls: Vec::new(),
                tool_call_id: Some("e2".into()),
            },
            // Final assistant synthesis (becomes the last non-empty assistant).
            Message {
                role: "assistant".into(),
                content: "Applied the a.rs fix; b.rs edit failed and was left as-is.".into(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
        ];

        let block = prior_session_summary_block(&history).expect("summary block");
        // The grounded receipt appears, carries only the *successful* path,
        // and sits at the top of the block (before the synthesis).
        assert!(
            block.starts_with("[Prior Session Summary]\nMutated files: src/a.rs\n"),
            "receipt must lead the block with only the successful path, got: {block}"
        );
        assert!(
            block.contains("Applied the a.rs fix"),
            "the last assistant synthesis must follow the receipt: {block}"
        );
        // The failed path must never appear in the receipt.
        assert!(
            !block.contains("src/b.rs"),
            "a failed edit must not be reported as mutated: {block}"
        );

        // A history with *only* failed edits yields a summary with no receipt
        // line at all (the synthesis is still carried).
        let failed_only = vec![
            msg("user", "try again"),
            Message {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: vec![tc(
                    "f1",
                    "write_file",
                    r#"{"path":"src/c.rs","content":"z"}"#,
                )],
                tool_call_id: None,
            },
            Message {
                role: "tool".into(),
                content: "Error: Access denied".into(),
                tool_calls: Vec::new(),
                tool_call_id: Some("f1".into()),
            },
            msg("assistant", "Could not write c.rs."),
        ];
        let block2 = prior_session_summary_block(&failed_only).expect("summary block");
        assert!(
            block2 == "[Prior Session Summary]\nCould not write c.rs.\n",
            "failed-only edits produce no receipt line, got: {block2}"
        );

        // No mutations at all: the block is unchanged (just the synthesis).
        let no_mut = vec![msg("user", "u"), msg("assistant", "final answer")];
        let block3 = prior_session_summary_block(&no_mut).expect("summary block");
        assert_eq!(block3, "[Prior Session Summary]\nfinal answer\n");
    }
}
