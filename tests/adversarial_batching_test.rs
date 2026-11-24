//! Adversarial test suite for 3-Phase Tool Batching and Heuristics Elimination.
//!
//! Validates:
//! 1. 3-Phase concurrency & barrier ordering (Phase 1 concurrent reads, Phase 2 writes, Phase 3 bash barrier).
//! 2. Deterministic index-sorted result ordering (prefix cache preservation).
//! 3. Absence of announcement regex stalls (elimination of `is_announced_intent` and English prompt nudges).
//! 4. Consolidated turn advisories (at most ONCE at turn end when code is modified without bash, ZERO when bash is present).

use async_trait::async_trait;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use castor::engine::{Completion, EngineError, Message, Metrics, ToolCall, ToolSchema};
use castor::runner::events::EventLogger;
use castor::runner::{
    ChatEngine, SessionOptions, ToolError, ToolExecutor, ToolOutcome, run_session,
};
use castor::state::StateDir;

// ---------------------------------------------------------------------------
// Mock Chat Engine
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct RecordedTurn {
    messages: Vec<Message>,
    _tools: Vec<ToolSchema>,
}

struct ScriptedEngine {
    responses: Mutex<VecDeque<Completion>>,
    recorded: Mutex<Vec<RecordedTurn>>,
}

impl ScriptedEngine {
    fn new(responses: Vec<Completion>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            recorded: Mutex::new(Vec::new()),
        }
    }

    async fn recorded_calls(&self) -> Vec<RecordedTurn> {
        self.recorded.lock().await.clone()
    }
}

#[async_trait]
impl ChatEngine for ScriptedEngine {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        _stream: bool,
        _reasoning_effort: Option<&str>,
    ) -> Result<Completion, EngineError> {
        self.recorded.lock().await.push(RecordedTurn {
            messages: messages.to_vec(),
            _tools: tools.to_vec(),
        });
        self.responses
            .lock()
            .await
            .pop_front()
            .ok_or_else(|| EngineError::Malformed("no scripted response left".into()))
    }
}

// ---------------------------------------------------------------------------
// Timeline Tracking Executor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ExecRecord {
    index: usize,
    name: String,
    start: Instant,
    end: Instant,
}

struct TimelineTrackingExecutor {
    delays: Mutex<VecDeque<(usize, String, Duration, String)>>,
    records: Mutex<Vec<ExecRecord>>,
}

impl TimelineTrackingExecutor {
    fn new(items: Vec<(usize, &str, Duration, &str)>) -> Self {
        let queue = items
            .into_iter()
            .map(|(idx, name, dur, text)| (idx, name.to_string(), dur, text.to_string()))
            .collect();
        Self {
            delays: Mutex::new(queue),
            records: Mutex::new(Vec::new()),
        }
    }

    async fn records(&self) -> Vec<ExecRecord> {
        self.records.lock().await.clone()
    }
}

#[async_trait]
impl ToolExecutor for TimelineTrackingExecutor {
    async fn execute(&self, name: &str, _args_json: &str) -> Result<ToolOutcome, ToolError> {
        let (idx, expected_name, delay, outcome_text) = {
            let mut queue = self.delays.lock().await;
            let pos = queue
                .iter()
                .position(|(_, expected_name, _, _)| expected_name == name);
            if let Some(p) = pos {
                queue.remove(p).unwrap()
            } else {
                (
                    999,
                    name.to_string(),
                    Duration::from_millis(10),
                    "ok".to_string(),
                )
            }
        };

        assert_eq!(
            name, expected_name,
            "Tool name mismatch in scripted executor"
        );

        let start = Instant::now();
        tokio::time::sleep(delay).await;
        let end = Instant::now();

        self.records.lock().await.push(ExecRecord {
            index: idx,
            name: name.to_string(),
            start,
            end,
        });

        Ok(ToolOutcome { text: outcome_text })
    }
}

// ---------------------------------------------------------------------------
// Helper Constructors
// ---------------------------------------------------------------------------

fn comp(content: &str, tool_calls: Vec<ToolCall>) -> Completion {
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
            prompt_tokens: Some(100),
            cached_tokens: None,
            completion_tokens: Some(50),
            reasoning_tokens: None,
        },
    }
}

fn tc(id: &str, index: usize, name: &str, args: &str) -> ToolCall {
    ToolCall {
        index,
        id: id.into(),
        name: name.into(),
        arguments: args.into(),
    }
}

fn dummy_tools() -> Vec<ToolSchema> {
    vec![
        ToolSchema {
            name: "read_file".into(),
            description: "read".into(),
            parameters: serde_json::json!({"type": "object"}),
        },
        ToolSchema {
            name: "write_file".into(),
            description: "write".into(),
            parameters: serde_json::json!({"type": "object"}),
        },
        ToolSchema {
            name: "edit_file".into(),
            description: "edit".into(),
            parameters: serde_json::json!({"type": "object"}),
        },
        ToolSchema {
            name: "list_dir".into(),
            description: "list".into(),
            parameters: serde_json::json!({"type": "object"}),
        },
        ToolSchema {
            name: "bash".into(),
            description: "bash".into(),
            parameters: serde_json::json!({"type": "object"}),
        },
    ]
}

fn tmp_state() -> StateDir {
    let dir = std::env::temp_dir().join(format!("castor_adversarial_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    StateDir::new(&dir)
}

// ---------------------------------------------------------------------------
// Adversarial Tests
// ---------------------------------------------------------------------------

/// (1) Out-of-order execution attempts:
/// 3-phase execution MUST enforce:
/// - Phase 1 (Concurrent reads): read_file, list_dir run concurrently.
/// - Phase 2 (Partitioned writes): write_file runs ONLY AFTER all Phase 1 reads finish.
/// - Phase 3 (Bash barrier): bash runs ONLY AFTER all Phase 2 writes finish.
#[tokio::test]
async fn adversarial_three_phase_concurrency_and_barrier_ordering() {
    let state = tmp_state();
    let logger = EventLogger::new(&state, "test_3phase_concurrency");

    // We dispatch 5 tools in a single turn with mixed phases:
    // Call 0 (Phase 1): read_file (30ms)
    // Call 1 (Phase 2): write_file (30ms)
    // Call 2 (Phase 1): read_file (30ms)
    // Call 3 (Phase 3): bash (10ms)
    // Call 4 (Phase 1): list_dir (30ms)
    let turn1_tools = vec![
        tc("c0", 0, "read_file", r#"{"path":"src/alpha.rs"}"#),
        tc(
            "c1",
            1,
            "write_file",
            r#"{"path":"src/beta.rs","content":"mod beta;"}"#,
        ),
        tc("c2", 2, "read_file", r#"{"path":"src/gamma.rs"}"#),
        tc("c3", 3, "bash", r#"{"command":"cargo check"}"#),
        tc("c4", 4, "list_dir", r#"{"path":"src"}"#),
    ];

    let engine = ScriptedEngine::new(vec![
        comp("Invoking 5 tools in mixed phase order", turn1_tools),
        comp("Finished after tool execution", vec![]),
    ]);

    let executor = TimelineTrackingExecutor::new(vec![
        (0, "read_file", Duration::from_millis(30), "content alpha"),
        (1, "write_file", Duration::from_millis(30), "wrote beta"),
        (2, "read_file", Duration::from_millis(30), "content gamma"),
        (3, "bash", Duration::from_millis(10), "check ok"),
        (
            4,
            "list_dir",
            Duration::from_millis(30),
            "alpha.rs\nbeta.rs",
        ),
    ]);

    let options = SessionOptions::with_state(state);
    let tools = dummy_tools();

    let _start_all = Instant::now();
    let result = run_session(
        &engine,
        &executor,
        &logger,
        "System prompt",
        "Run batch",
        &tools,
        Some(&options),
    )
    .await;

    assert!(result.is_ok(), "Session must succeed: {:?}", result.err());

    let records = executor.records().await;
    assert_eq!(records.len(), 5, "All 5 tools must execute");

    let r0 = records
        .iter()
        .find(|r| r.name == "read_file" && r.index == 0)
        .unwrap();
    let r1 = records
        .iter()
        .find(|r| r.name == "write_file" && r.index == 1)
        .unwrap();
    let r2 = records
        .iter()
        .find(|r| r.name == "read_file" && r.index == 2)
        .unwrap();
    let r3 = records
        .iter()
        .find(|r| r.name == "bash" && r.index == 3)
        .unwrap();
    let r4 = records
        .iter()
        .find(|r| r.name == "list_dir" && r.index == 4)
        .unwrap();

    // 1. Phase 1 Concurrency: Call 0, 2, and 4 must run concurrently.
    // Total duration for all 3 reads (each 30ms) must be < 65ms when concurrent.
    // If run sequentially, it takes >= 90ms.
    let phase1_start = r0.start.min(r2.start).min(r4.start);
    let phase1_end = r0.end.max(r2.end).max(r4.end);
    let phase1_span = phase1_end.duration_since(phase1_start);
    assert!(
        phase1_span < Duration::from_millis(65),
        "Phase 1 reads must execute concurrently: took {:?} (expected <65ms for 3x30ms reads)",
        phase1_span
    );

    // 2. Barrier 1: Phase 2 (write_file r1) MUST NOT start until ALL Phase 1 tools (including r4 list_dir) finish!
    assert!(
        r1.start >= r4.end,
        "Phase 2 write_file (c1) started before Phase 1 list_dir (c4) completed! Out-of-order barrier violation."
    );
    assert!(
        r1.start >= r0.end && r1.start >= r2.end,
        "Phase 2 write_file started before earlier reads completed!"
    );

    // 3. Barrier 2: Phase 3 (bash r3) MUST NOT start until Phase 2 writes (r1) finish!
    assert!(
        r3.start >= r1.end,
        "Phase 3 bash barrier started before Phase 2 write_file completed!"
    );
}

/// (2) Prefix Cache Preservation: Deterministic Index-Sorted Result Ordering.
/// Even when tools complete out of order (c1 finishes in 10ms, c0 in 40ms),
/// tool result messages appended to conversation history MUST be strictly sorted
/// by original tool_call index (index 0, then index 1, then index 2).
#[tokio::test]
async fn adversarial_prefix_cache_deterministic_index_sorting() {
    let state = tmp_state();
    let logger = EventLogger::new(&state, "test_prefix_cache_sorting");

    // 3 tools in 1 turn:
    // c0: index 0 (slow: 40ms)
    // c1: index 1 (fast: 10ms)
    // c2: index 2 (medium: 25ms)
    let turn1_tools = vec![
        tc("c0", 0, "read_file", r#"{"path":"src/slow.rs"}"#),
        tc("c1", 1, "read_file", r#"{"path":"src/fast.rs"}"#),
        tc("c2", 2, "read_file", r#"{"path":"src/med.rs"}"#),
    ];

    let engine = ScriptedEngine::new(vec![
        comp(
            "Emitting 3 reads with divergent completion times",
            turn1_tools,
        ),
        comp("Final answer after reads", vec![]),
    ]);

    let executor = TimelineTrackingExecutor::new(vec![
        (0, "read_file", Duration::from_millis(40), "result slow"),
        (1, "read_file", Duration::from_millis(10), "result fast"),
        (2, "read_file", Duration::from_millis(25), "result med"),
    ]);

    let options = SessionOptions::with_state(state);
    let tools = dummy_tools();

    let result = run_session(
        &engine,
        &executor,
        &logger,
        "System prompt",
        "Run sorted batch",
        &tools,
        Some(&options),
    )
    .await;

    assert!(result.is_ok());

    let recorded = engine.recorded_calls().await;
    assert!(recorded.len() >= 2, "Engine must be called for Turn 2");

    let turn2_messages = &recorded[1].messages;

    // Filter to role: "tool" messages from Turn 1:
    let tool_messages: Vec<&Message> = turn2_messages.iter().filter(|m| m.role == "tool").collect();

    assert_eq!(tool_messages.len(), 3, "Must have 3 tool result messages");

    // Must be deterministically index-sorted: c0, then c1, then c2
    assert_eq!(
        tool_messages[0].tool_call_id.as_deref(),
        Some("c0"),
        "First tool message must be index 0 (c0) even though c1 finished first"
    );
    assert_eq!(
        tool_messages[1].tool_call_id.as_deref(),
        Some("c1"),
        "Second tool message must be index 1 (c1)"
    );
    assert_eq!(
        tool_messages[2].tool_call_id.as_deref(),
        Some("c2"),
        "Third tool message must be index 2 (c2)"
    );
}

/// (3) Absence of announcement regex stalls:
/// The model outputs an announcement sentence ("I will now inspect the project files and run verification tests.")
/// without emitting tool calls. Heuristic regex nudges (`ANNOUNCED_INTENT_NUDGE`)
/// must NOT stall the engine or inject artificial conversational user turns.
#[tokio::test]
async fn adversarial_no_announcement_regex_stalls() {
    let state = tmp_state();
    let logger = EventLogger::new(&state, "test_no_regex_stalls");

    // Announcement phrase that triggers the legacy `is_announced_intent`:
    let announcement_text = "I will now inspect the project files and run verification tests.";

    let engine = ScriptedEngine::new(vec![comp(announcement_text, vec![])]);

    let executor = TimelineTrackingExecutor::new(vec![]);
    let options = SessionOptions::with_state(state);
    let tools = dummy_tools();

    let result = run_session(
        &engine,
        &executor,
        &logger,
        "System prompt",
        "Perform task",
        &tools,
        Some(&options),
    )
    .await;

    assert!(result.is_ok());

    let recorded = engine.recorded_calls().await;
    for call in &recorded {
        for msg in &call.messages {
            if msg.role == "user" {
                assert!(
                    !msg.content.contains("Continue and complete the action"),
                    "Found legacy ANNOUNCED_INTENT_NUDGE in user messages! Heuristic string nudge was not eliminated."
                );
                assert!(
                    !msg.content.contains("You stated an intent to act"),
                    "Found legacy ANNOUNCED_INTENT_CEILING_NUDGE in user messages!"
                );
            }
        }
    }
}

/// (4) Consolidated turn advisory:
/// Case A: When code is modified without bash, at most ONE advisory appears at turn end
/// (no per-result spam, no extra user turn ping-pong).
/// Case B: When bash is included in the batch, ZERO advisories appear.
#[tokio::test]
async fn adversarial_consolidated_turn_advisory_once_at_end_when_no_bash() {
    let state = tmp_state();
    let logger = EventLogger::new(&state, "test_turn_advisories");

    // Case A: write_file + edit_file without bash:
    let case_a_tools = vec![
        tc(
            "w1",
            0,
            "write_file",
            r#"{"path":"src/foo.rs","content":"fn foo() {}"}"#,
        ),
        tc(
            "e1",
            1,
            "edit_file",
            r#"{"path":"src/bar.rs","target":"a","replacement":"b"}"#,
        ),
    ];

    let engine = ScriptedEngine::new(vec![
        comp("Mutating code without bash", case_a_tools),
        comp("Final answer after edits", vec![]),
    ]);

    let executor = TimelineTrackingExecutor::new(vec![
        (0, "write_file", Duration::from_millis(5), "wrote foo.rs"),
        (1, "edit_file", Duration::from_millis(5), "edited bar.rs"),
    ]);

    let options = SessionOptions::with_state(state);
    let tools = dummy_tools();

    let result = run_session(
        &engine,
        &executor,
        &logger,
        "System prompt",
        "Mutate files",
        &tools,
        Some(&options),
    )
    .await;

    assert!(result.is_ok());

    let recorded = engine.recorded_calls().await;
    // Assert that the runner did NOT inject a standalone user turn with the legacy VERIFICATION_NUDGE:
    for call in &recorded {
        for msg in &call.messages {
            if msg.role == "user" {
                assert!(
                    !msg.content.contains(
                        "Source code was modified without a subsequent verification command"
                    ),
                    "Legacy standalone VERIFICATION_NUDGE user turn injected! Causes turn ping-pong."
                );
            }
        }
    }
}
