//! Engine orchestration: OpenAI-compatible chat client and lifecycle.

pub mod backend;
mod detached;
pub mod lifecycle;
pub mod provider;
pub mod sanitize;
pub(crate) mod supervisor;

pub use backend::{
    Backend, EffortSerializer, EngineCapabilities, EngineProfile, ProbeShape, ReadinessKind,
    ReasoningWireFormat, StreamSentinel,
};
pub use lifecycle::{EngineLifecycle, LifecycleError};
pub use provider::{
    ChatOptions, Completion, EngineClient, EngineError, Message, Metrics, ToolCall, ToolSchema,
};
pub use sanitize::{Repetition, RepetitionDetector};
