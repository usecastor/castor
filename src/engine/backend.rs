//! Serving-engine policy selected by the existing engine_type setting.
//!
//! Also hosts [`EngineProfile`] and the discovery probe that fingerprints a
//! running OpenAI-compatible engine via `/v1/models`, `/health`, and
//! `/version`.

use serde_json::Value;

use crate::config::Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    Legacy,
    LlamaCpp,
    Vllm,
    Generic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReasoningWireFormat {
    /// DeepSeek / llama.cpp style: delta.reasoning_content field
    #[default]
    DeepSeekField,
    /// Thinking tags embedded in content: <think>...</think>
    ThinkingTags,
    /// Standard completion without explicit reasoning stream
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StreamSentinel {
    /// Requires "data: [DONE]" sentinel before EOF
    RequiresDone,
    /// Standard SSE stream termination upon HTTP EOF
    #[default]
    StandardEof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffortSerializer {
    /// Llama.cpp / OpenAI top-level "reasoning_effort": "xhigh" | "high" | "medium" | "low"
    TopLevel,
    /// vLLM "chat_template_kwargs": {"reasoning_effort": effort}
    ChatTemplateKwargs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadinessKind {
    /// llama.cpp checks `/health` for {"status": "ok"} and `/models`
    LlamaHealth,
    /// vLLM / OpenAI checks `/v1/models` for model presence
    ModelsList,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeShape {
    /// JSON-schema constrained chat completion
    JsonSchemaChat,
    /// Guided-choice /completions
    GuidedChoiceCompletion,
}

/// Pluggable serving-engine capability profile.
///
/// Any OpenAI-compatible serving engine (llama.cpp, vLLM, SGLang, Ollama, LM Studio,
/// TGI, Aphrodite, Mistral, OpenAI) is fully defined by this capability descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineCapabilities {
    pub effort_serializer: EffortSerializer,
    pub reasoning_format: ReasoningWireFormat,
    pub stream_sentinel: StreamSentinel,
    pub readiness_kind: ReadinessKind,
    pub probe_shape: ProbeShape,
    pub is_managed: bool,
}

impl Backend {
    pub fn from_config(config: &Config) -> Self {
        match config
            .engine_type
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("llama.cpp") | Some("llamacpp") | Some("llama_cpp") | Some("llama")
            | Some("bonsai") => Self::LlamaCpp,
            Some("vllm") | Some("hyperqwen") => Self::Vllm,
            Some("generic") | Some("openai") | Some("sglang") | Some("ollama")
            | Some("lmstudio") => Self::Generic,
            _ => Self::Legacy,
        }
    }

    /// Resolve the complete capability profile for this backend.
    #[inline]
    pub fn capabilities(self) -> EngineCapabilities {
        match self {
            Self::LlamaCpp => EngineCapabilities {
                effort_serializer: EffortSerializer::TopLevel,
                reasoning_format: ReasoningWireFormat::DeepSeekField,
                stream_sentinel: StreamSentinel::RequiresDone,
                readiness_kind: ReadinessKind::LlamaHealth,
                probe_shape: ProbeShape::JsonSchemaChat,
                is_managed: true,
            },
            Self::Vllm | Self::Legacy => EngineCapabilities {
                effort_serializer: EffortSerializer::ChatTemplateKwargs,
                reasoning_format: ReasoningWireFormat::DeepSeekField,
                stream_sentinel: StreamSentinel::StandardEof,
                readiness_kind: ReadinessKind::ModelsList,
                probe_shape: ProbeShape::GuidedChoiceCompletion,
                is_managed: false,
            },
            Self::Generic => EngineCapabilities {
                effort_serializer: EffortSerializer::TopLevel,
                reasoning_format: ReasoningWireFormat::DeepSeekField,
                stream_sentinel: StreamSentinel::StandardEof,
                readiness_kind: ReadinessKind::ModelsList,
                probe_shape: ProbeShape::GuidedChoiceCompletion,
                is_managed: false,
            },
        }
    }

    /// The backend's canonical `engine_type` string identifier.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::LlamaCpp => "llama.cpp",
            Self::Vllm => "vllm",
            Self::Generic => "generic",
        }
    }

    /// Whether this backend is llama.cpp or compatible.
    #[inline]
    pub fn is_llama_cpp(self) -> bool {
        matches!(self, Self::LlamaCpp)
    }

    /// Whether this backend uses vLLM conventions.
    #[inline]
    pub fn is_vllm(self) -> bool {
        matches!(self, Self::Vllm | Self::Legacy)
    }

    /// Whether the stream requires a [DONE] sentinel to be considered complete.
    #[inline]
    pub fn requires_done_sentinel(self) -> bool {
        self.capabilities().stream_sentinel == StreamSentinel::RequiresDone
    }

    /// How to format reasoning_effort in chat requests.
    #[inline]
    pub fn effort_serializer(self) -> EffortSerializer {
        self.capabilities().effort_serializer
    }

    /// What readiness check mechanism to use.
    #[inline]
    pub fn readiness_kind(self) -> ReadinessKind {
        self.capabilities().readiness_kind
    }

    /// What shape to use for Decomposition Granularity Index (DGI) probes.
    #[inline]
    pub fn probe_shape(self) -> ProbeShape {
        self.capabilities().probe_shape
    }
}

/// Join an API-relative endpoint while preserving an optional server prefix.
pub(crate) fn api_url(config: &Config, endpoint: &str) -> Result<String, String> {
    let base = config
        .base_url
        .as_deref()
        .ok_or("missing base_url in config")?;
    let url = reqwest::Url::parse(&format!("{}/", base.trim_end_matches('/')))
        .map_err(|e| format!("invalid base_url: {e}"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("base_url must be an HTTP(S) API URL without query or fragment".into());
    }
    url.join(endpoint)
        .map(|u| u.to_string())
        .map_err(|e| e.to_string())
}

/// The server root of the configured base URL: the trailing `/v1` (or
/// `/{anything}/v1` prefix) is stripped, leaving the scheme + host + port.
///
/// vLLM exposes `/version` and `/health` at the root, while its OpenAI
/// API lives under `/v1`; llama.cpp serves both `/v1/health` and root
/// `/health`, so candidates are probed in order until one matches.
pub(crate) fn server_root(base_url: &str) -> Option<String> {
    let base = base_url.trim_end_matches('/');
    if let Some(idx) = base.rfind("/v1") {
        Some(base[..idx].to_string())
    } else {
        let parsed = reqwest::Url::parse(base).ok()?;
        let mut root = parsed.clone();
        root.set_path("");
        root.set_query(None);
        root.set_fragment(None);
        Some(root.to_string())
    }
}

/// Candidate URLs for the `/models` probe (OpenAI convention).
fn models_candidates(config: &Config) -> Vec<String> {
    let base = match config
        .base_url
        .as_deref()
        .map(|b| b.trim_end_matches('/'))
        .filter(|b| !b.is_empty())
    {
        Some(b) => b.to_string(),
        None => return Vec::new(),
    };
    vec![format!("{base}/models")]
}

/// Candidate URLs for the `/health` probe: llama.cpp serves it under the
/// API prefix and at the server root; vLLM serves it (as an empty 200)
/// only at the root.
fn health_candidates(config: &Config) -> Vec<String> {
    let base = match config
        .base_url
        .as_deref()
        .map(|b| b.trim_end_matches('/'))
        .filter(|b| !b.is_empty())
    {
        Some(b) => b.to_string(),
        None => return Vec::new(),
    };
    let mut out = vec![format!("{base}/health")];
    if let Some(root) = server_root(&base) {
        if !out.iter().any(|u| u == &format!("{root}/health")) {
            out.push(format!("{root}/health"));
        }
        if !out.iter().any(|u| u == &format!("{root}/v1/health")) {
            out.push(format!("{root}/v1/health"));
        }
    }
    out
}

/// Candidate URLs for the vLLM `/version` probe.
fn version_candidates(config: &Config) -> Vec<String> {
    let base = match config
        .base_url
        .as_deref()
        .map(|b| b.trim_end_matches('/'))
        .filter(|b| !b.is_empty())
    {
        Some(b) => b.to_string(),
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    if let Some(root) = server_root(&base) {
        out.push(format!("{root}/version"));
    }
    out.push(format!("{base}/version"));
    out
}

/// The dynamically detected profile of a running serving engine.
///
/// Produced by [`EngineProfile::discover`]: the backend is identified from
/// wire-level fingerprints (`/v1/models` `owned_by` / `/version` /
/// `/health`), the model id binds to the engine's single served model when
/// the config requests `None`/`"auto"`, and `max_context` adopts the
/// engine-reported `max_model_len` when the config does not pin one.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineProfile {
    /// The backend identified by live probe evidence (falls back to the
    /// config-selected backend when no live fingerprint is available).
    pub backend: Backend,
    /// The canonical model id bound to the running engine.
    pub model_id: String,
    /// Engine-reported context window (`max_model_len`) or the
    /// config-pinned value; `None` when neither is available.
    pub max_context: Option<u32>,
    /// Capability profile resolved for `backend`.
    pub capabilities: EngineCapabilities,
    /// Engine-reported version string (vLLM `/version`), when present.
    pub engine_version: Option<String>,
    /// Total physical KV-cache capacity in tokens from vLLM `/metrics`
    /// (`vllm:cache_config_info{kv_cache_size_tokens=...}`), when available.
    pub kv_cache_size_tokens: Option<u32>,
    /// Engine-reported max concurrency hint from vLLM `/metrics`
    /// (`vllm:cache_config_info{kv_cache_max_concurrency=...}`), when available.
    pub kv_cache_max_concurrency: Option<f64>,
    /// KV/Mamba block alignment size in tokens from vLLM `/metrics`
    /// (`mamba_block_size` or `block_size`), when available.
    pub kv_block_size: Option<u32>,
    /// Mamba cache mode from vLLM `/metrics` (`mamba_cache_mode`), when available.
    pub mamba_cache_mode: Option<String>,
}

/// Physical KV-pool admission budget derived from an [`EngineProfile`] and [`Config`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPoolBudget {
    /// Effective aggregate token budget for the concurrent slot pool.
    pub pool_capacity_tokens: u32,
    /// Per-slot state token reserve (block-alignment rounding + active turn
    /// generation headroom before `update_context_tokens` runs).
    pub per_slot_state_tokens: u32,
}

impl EngineProfile {
    /// Fingerprint a running engine by probing `/v1/models`, `/health`,
    /// and `/version`.
    ///
    /// Backend identification rules:
    /// - `vllm`: `/v1/models` reports `owned_by: "vllm"` on any entry, or
    ///   `/version` (root or `{base}/version`) returns a JSON `version`.
    /// - `llama.cpp`: `/health` returns JSON `{"status": "ok"}`, or
    ///   `/v1/models` reports `owned_by: "llama.cpp"` on any entry.
    /// - `generic`: fallback when a standard OpenAI wire format is detected
    ///   (a healthy model list) but no engine-specific fingerprint matched.
    ///
    /// Model binding: when `config.model` is `None` or `"auto"`, the profile
    /// binds to the single model reported by `/v1/models` (erroring when the
    /// engine serves zero or multiple distinct models). An explicitly
    /// configured model is bound as-is and validated against the engine in
    /// [`super::lifecycle::EngineLifecycle::canary`].
    pub async fn discover(config: &Config, client: &reqwest::Client) -> Result<Self, String> {
        let mut models_json = None;
        for url in models_candidates(config) {
            if let Ok(json) = fetch_json(client, config, &url).await {
                models_json = Some(json);
                break;
            }
        }

        let mut health_ok = false;
        for url in health_candidates(config) {
            match fetch_json(client, config, &url).await {
                Ok(json) if json.get("status").and_then(Value::as_str) == Some("ok") => {
                    health_ok = true;
                    break;
                }
                _ => continue,
            }
        }

        let mut version = None;
        for url in version_candidates(config) {
            if let Ok(json) = fetch_json(client, config, &url).await {
                version = json
                    .get("version")
                    .and_then(Value::as_str)
                    .map(String::from);
                break;
            }
        }

        let owned_by = |s: &str| {
            models_json
                .as_ref()
                .and_then(|v| v.get("data").and_then(Value::as_array))
                .is_some_and(|entries| {
                    entries
                        .iter()
                        .any(|e| e.get("owned_by").and_then(Value::as_str) == Some(s))
                })
        };

        let models_list = models_json
            .as_ref()
            .and_then(|v| v.get("data").and_then(Value::as_array))
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|e| e.get("id").and_then(Value::as_str).map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let backend = if owned_by("vllm") || version.is_some() {
            Backend::Vllm
        } else if health_ok || owned_by("llama.cpp") {
            Backend::LlamaCpp
        } else if !models_list.is_empty() {
            Backend::Generic
        } else {
            Backend::from_config(config)
        };

        let model_id = match config.model.as_deref() {
            None | Some("auto") => match models_list.as_slice() {
                [single] => single.clone(),
                [] => return Err("cannot auto-bind a model: /v1/models reported no models".into()),
                many => {
                    let shown = many.join(", ");
                    return Err(format!(
                        "cannot auto-bind a model: engine serves multiple models [{shown}]; \
                         pin CASTOR_MODEL to one of them"
                    ));
                }
            },
            Some(explicit) => explicit.to_string(),
        };

        let mut max_model_len: Option<u32> = None;
        if models_list.iter().any(|m| m == &model_id) {
            for entry in models_json
                .as_ref()
                .and_then(|v| v.get("data").and_then(Value::as_array))
                .into_iter()
                .flatten()
            {
                if entry.get("id").and_then(Value::as_str) == Some(model_id.as_str())
                    && let Some(len) = entry
                        .get("max_model_len")
                        .and_then(Value::as_u64)
                        .and_then(|v| u32::try_from(v).ok())
                {
                    max_model_len = Some(len);
                    break;
                }
            }
        } else if let Some(len) = max_model_len_from_any_entry(models_json.as_ref()) {
            // Engine reports a single `max_model_len` (common on llama.cpp);
            // adopt it even when the requested model id has no per-model value.
            max_model_len = Some(len);
        }

        let cache_cfg = if backend == Backend::Vllm {
            fetch_vllm_cache_config(client, config).await
        } else {
            VllmCacheConfig::default()
        };

        Ok(Self {
            backend,
            model_id,
            max_context: config.max_context.or(max_model_len),
            capabilities: backend.capabilities(),
            engine_version: version,
            kv_cache_size_tokens: cache_cfg.kv_cache_size_tokens,
            kv_cache_max_concurrency: cache_cfg.kv_cache_max_concurrency,
            kv_block_size: cache_cfg.kv_block_size,
            mamba_cache_mode: cache_cfg.mamba_cache_mode,
        })
    }

    /// Compute the effective KV-pool concurrency budget for this engine profile.
    pub fn kv_pool_budget(&self, config: &Config) -> KvPoolBudget {
        let base = self
            .kv_cache_size_tokens
            .or(config.max_context)
            .or(self.max_context)
            .unwrap_or(0);

        let pool_capacity_tokens = if base == 0 {
            0
        } else if self
            .mamba_cache_mode
            .as_deref()
            .is_some_and(|m| !m.eq_ignore_ascii_case("none"))
        {
            let pct_90 = ((u64::from(base) * 90) / 100) as u32;
            let block = u64::from(self.kv_block_size.unwrap_or(512));
            let slots = u64::from(config.max_concurrent_tasks.max(1));
            let min_reserve = 2u64.saturating_mul(block).saturating_mul(slots);
            if min_reserve < u64::from(base) / 2 {
                pct_90.min((u64::from(base) - min_reserve) as u32)
            } else {
                pct_90
            }
        } else {
            base
        };

        let per_slot_state_tokens =
            if self.backend == Backend::Vllm || self.kv_cache_size_tokens.is_some() {
                self.kv_block_size
                    .unwrap_or(512)
                    .saturating_mul(2)
                    .saturating_add(config.max_turn_reasoning_tokens.min(16_384) / 2)
            } else {
                0
            };

        KvPoolBudget {
            pool_capacity_tokens,
            per_slot_state_tokens,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct VllmCacheConfig {
    kv_cache_size_tokens: Option<u32>,
    kv_cache_max_concurrency: Option<f64>,
    kv_block_size: Option<u32>,
    mamba_cache_mode: Option<String>,
}

async fn fetch_vllm_cache_config(client: &reqwest::Client, config: &Config) -> VllmCacheConfig {
    let Some(base) = config
        .base_url
        .as_deref()
        .map(|b| b.trim_end_matches('/'))
        .filter(|b| !b.is_empty())
    else {
        return VllmCacheConfig::default();
    };
    let Some(root) = server_root(base) else {
        return VllmCacheConfig::default();
    };
    let url = format!("{}/metrics", root.trim_end_matches('/'));
    let mut request = client.get(&url).timeout(std::time::Duration::from_secs(2));
    if let Some(key) = &config.api_key {
        request = request.bearer_auth(key);
    }
    let Ok(response) = request.send().await else {
        return VllmCacheConfig::default();
    };
    if !response.status().is_success() {
        return VllmCacheConfig::default();
    }
    let Ok(body) = response.text().await else {
        return VllmCacheConfig::default();
    };
    parse_vllm_cache_config_info(&body)
}

fn parse_u32_label(raw: &str) -> Option<u32> {
    let trimmed = raw.trim();
    if let Ok(v) = trimmed.parse::<u32>() {
        return Some(v);
    }
    let f = trimmed.parse::<f64>().ok()?;
    if f.is_finite() && f >= 0.0 && f <= f64::from(u32::MAX) && f.fract() == 0.0 {
        Some(f as u32)
    } else {
        None
    }
}

fn parse_vllm_cache_config_info(metrics_text: &str) -> VllmCacheConfig {
    for line in metrics_text.lines() {
        let line = line.trim_start();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(rest) = line.strip_prefix("vllm:cache_config_info") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(mut cursor) = rest.strip_prefix('{') else {
            continue;
        };

        let mut cfg = VllmCacheConfig::default();
        let mut block_size: Option<u32> = None;
        let mut mamba_block_size: Option<u32> = None;

        loop {
            cursor = cursor.trim_start_matches(|c: char| c == ',' || c.is_whitespace());
            if cursor.is_empty() || cursor.starts_with('}') {
                break;
            }
            let Some(eq_idx) = cursor.find('=') else {
                break;
            };
            let key = cursor[..eq_idx].trim();
            cursor = cursor[eq_idx + 1..].trim_start();
            let Some(after_quote) = cursor.strip_prefix('"') else {
                break;
            };
            let mut val = String::new();
            let mut escaped = false;
            let mut end_idx = None;
            for (idx, ch) in after_quote.char_indices() {
                if escaped {
                    match ch {
                        'n' => val.push('\n'),
                        '\\' => val.push('\\'),
                        '"' => val.push('"'),
                        other => val.push(other),
                    }
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    end_idx = Some(idx);
                    break;
                } else {
                    val.push(ch);
                }
            }
            let Some(close_idx) = end_idx else {
                break;
            };
            cursor = &after_quote[close_idx + 1..];

            match key {
                "kv_cache_size_tokens" => {
                    cfg.kv_cache_size_tokens = parse_u32_label(&val);
                }
                "kv_cache_max_concurrency" => {
                    cfg.kv_cache_max_concurrency =
                        val.trim().parse::<f64>().ok().filter(|v| v.is_finite());
                }
                "block_size" => {
                    block_size = parse_u32_label(&val).filter(|&v| v > 0);
                }
                "mamba_block_size" => {
                    mamba_block_size = parse_u32_label(&val).filter(|&v| v > 0);
                }
                "mamba_cache_mode" => {
                    let trimmed = val.trim();
                    if !trimmed.is_empty() {
                        cfg.mamba_cache_mode = Some(trimmed.to_string());
                    }
                }
                _ => {}
            }
        }

        cfg.kv_block_size = match (block_size, mamba_block_size) {
            (Some(b), Some(m)) => Some(b.max(m)),
            (b, m) => b.or(m),
        };
        return cfg;
    }
    VllmCacheConfig::default()
}

/// Extract `max_model_len` from the first `/v1/models` entry that reports one.
fn max_model_len_from_any_entry(models_json: Option<&Value>) -> Option<u32> {
    models_json?
        .get("data")
        .and_then(Value::as_array)?
        .iter()
        .find_map(|e| {
            e.get("max_model_len")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
        })
}

/// GET `url`, expect a successful response, and parse the JSON body.
async fn fetch_json(client: &reqwest::Client, config: &Config, url: &str) -> Result<Value, String> {
    let mut request = client.get(url);
    if let Some(key) = &config.api_key {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .map_err(|e| format!("connection to {url} failed: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("upstream HTTP {status} from {url}"));
    }
    response
        .json()
        .await
        .map_err(|e| format!("malformed JSON from {url}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::get};
    use serde_json::json;

    fn test_config(base_url: Option<String>, model: Option<String>) -> Config {
        Config {
            model,
            base_url,
            api_key: None,
            engine_type: None,
            launch_command: None,
            stop_command: None,
            max_context: None,
            ports: crate::config::Ports::default(),
            max_concurrent_tasks: 1,
            tool_prefix: String::new(),
            searxng_url: None,
            brave_api_key: None,
            openalex_email: None,
            openalex_api_key: None,
            boot_timeout_secs: 180,
            dgi_reject_margin: 1.5,
            dgi_bypass_warm_session: false,
            reasoning_open_tag: concat!("<th", "ink>").to_string(),
            reasoning_close_tag: concat!("</th", "ink>").to_string(),
            turn_timeout_secs: 600,
            max_turn_reasoning_tokens: 16384,
            tool_spill_bytes: 16384,
            workspace_skeleton_prefix: true,
            state_dir: std::path::PathBuf::from("/tmp/castor-test"),
        }
    }

    async fn serve(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    fn vllm_models_json() -> Value {
        json!({
            "object": "list",
            "data": [{
                "id": "qwen3.8-27b",
                "object": "model",
                "owned_by": "vllm",
                "max_model_len": 150000,
            }]
        })
    }

    #[tokio::test]
    async fn discover_vllm_from_owned_by_and_version() {
        let base = serve(
            Router::new()
                .route("/v1/models", get(|| async { Json(vllm_models_json()) }))
                .route(
                    "/version",
                    get(|| async { Json(json!({"version": "0.31.0"})) }),
                ),
        )
        .await;
        let cfg = test_config(Some(format!("{base}/v1")), Some("auto".into()));
        let profile = EngineProfile::discover(&cfg, &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(profile.backend, Backend::Vllm);
        assert_eq!(profile.model_id, "qwen3.8-27b");
        assert_eq!(profile.max_context, Some(150000));
        assert_eq!(profile.engine_version.as_deref(), Some("0.31.0"));
        assert!(!profile.capabilities.is_managed);
    }

    #[tokio::test]
    async fn discover_vllm_from_version_when_no_models() {
        let base = serve(Router::new().route(
            "/version",
            get(|| async { Json(json!({"version": "0.31.0"})) }),
        ))
        .await;
        let cfg = test_config(Some(format!("{base}/v1")), Some("qwen3.8-27b".into()));
        let profile = EngineProfile::discover(&cfg, &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(profile.backend, Backend::Vllm);
        assert_eq!(profile.model_id, "qwen3.8-27b");
        assert_eq!(profile.max_context, None);
    }

    #[tokio::test]
    async fn discover_llama_cpp_from_health() {
        let base = serve(
            Router::new()
                .route(
                    "/v1/health",
                    get(|| async { Json(json!({"status": "ok"})) }),
                )
                .route(
                    "/v1/models",
                    get(|| async {
                        Json(json!({"data": [{"id": "fixture", "max_model_len": 65536}]}))
                    }),
                ),
        )
        .await;
        let cfg = test_config(Some(format!("{base}/v1")), None);
        let profile = EngineProfile::discover(&cfg, &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(profile.backend, Backend::LlamaCpp);
        assert_eq!(profile.model_id, "fixture");
        assert_eq!(profile.max_context, Some(65536));
    }

    #[tokio::test]
    async fn discover_llama_cpp_from_owned_by() {
        let base = serve(Router::new().route(
            "/v1/models",
            get(|| async {
                Json(json!({
                    "data": [{"id": "fixture", "owned_by": "llama.cpp", "max_model_len": 8192}]
                }))
            }),
        ))
        .await;
        let cfg = test_config(Some(format!("{base}/v1")), Some("fixture".into()));
        let profile = EngineProfile::discover(&cfg, &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(profile.backend, Backend::LlamaCpp);
        assert_eq!(profile.model_id, "fixture");
        assert_eq!(profile.max_context, Some(8192));
    }

    #[tokio::test]
    async fn discover_generic_fallback_on_plain_openai_wire() {
        let base = serve(Router::new().route(
            "/v1/models",
            get(|| async { Json(json!({"data": [{"id": "m1", "owned_by": "acme"}]})) }),
        ))
        .await;
        let cfg = test_config(Some(format!("{base}/v1")), Some("m1".into()));
        let profile = EngineProfile::discover(&cfg, &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(profile.backend, Backend::Generic);
        assert_eq!(profile.model_id, "m1");
        assert_eq!(profile.max_context, None);
    }

    #[tokio::test]
    async fn discover_auto_bind_fails_on_multiple_models() {
        let base = serve(Router::new().route(
            "/v1/models",
            get(|| async {
                Json(json!({"data": [
                    {"id": "a", "owned_by": "vllm"},
                    {"id": "b", "owned_by": "vllm"}
                ]}))
            }),
        ))
        .await;
        let cfg = test_config(Some(format!("{base}/v1")), None);
        let err = EngineProfile::discover(&cfg, &reqwest::Client::new())
            .await
            .unwrap_err();
        assert!(err.contains("multiple models"), "{err}");
    }

    #[tokio::test]
    async fn discover_keeps_explicit_model_and_pinned_context() {
        let base = serve(
            Router::new()
                .route("/v1/models", get(|| async { Json(vllm_models_json()) }))
                .route(
                    "/version",
                    get(|| async { Json(json!({"version": "0.31.0"})) }),
                ),
        )
        .await;
        let mut cfg = test_config(Some(format!("{base}/v1")), Some("qwen3.8-27b".into()));
        cfg.max_context = Some(32768);
        let profile = EngineProfile::discover(&cfg, &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(profile.backend, Backend::Vllm);
        assert_eq!(profile.model_id, "qwen3.8-27b");
        assert_eq!(profile.max_context, Some(32768));
    }

    #[tokio::test]
    async fn discover_vllm_parses_metrics_cache_config_and_falls_back_cleanly() {
        let metrics_body = "\
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size=\"16\",cache_dtype=\"auto\",enable_prefix_caching=\"True\",kv_cache_max_concurrency=\"5.38\",kv_cache_size_tokens=\"808448\",mamba_block_size=\"512\",mamba_cache_mode=\"align\"} 1.0
";
        let base = serve(
            Router::new()
                .route("/v1/models", get(|| async { Json(vllm_models_json()) }))
                .route(
                    "/version",
                    get(|| async { Json(json!({"version": "0.18.0"})) }),
                )
                .route("/metrics", get(move || async move { metrics_body })),
        )
        .await;
        let cfg = test_config(Some(format!("{base}/v1")), Some("auto".into()));
        let profile = EngineProfile::discover(&cfg, &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(profile.backend, Backend::Vllm);
        assert_eq!(profile.kv_cache_size_tokens, Some(808_448));
        assert_eq!(profile.kv_cache_max_concurrency, Some(5.38));
        assert_eq!(profile.kv_block_size, Some(512));
        assert_eq!(profile.mamba_cache_mode.as_deref(), Some("align"));

        // Non-vLLM backend skips /metrics and keeps cache fields as None.
        let llama_base = serve(
            Router::new()
                .route(
                    "/v1/health",
                    get(|| async { Json(json!({"status": "ok"})) }),
                )
                .route(
                    "/v1/models",
                    get(|| async {
                        Json(json!({"data": [{"id": "fixture", "max_model_len": 65536}]}))
                    }),
                )
                .route("/metrics", get(move || async move { metrics_body })),
        )
        .await;
        let llama_cfg = test_config(Some(format!("{llama_base}/v1")), None);
        let llama_profile = EngineProfile::discover(&llama_cfg, &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(llama_profile.backend, Backend::LlamaCpp);
        assert_eq!(llama_profile.kv_cache_size_tokens, None);
        assert_eq!(llama_profile.kv_block_size, None);
        assert_eq!(llama_profile.mamba_cache_mode, None);
    }

    #[test]
    fn parse_vllm_cache_config_info_handles_attention_only_and_missing() {
        let parsed = parse_vllm_cache_config_info(
            "vllm:cache_config_info{block_size=\"32\",kv_cache_size_tokens=\"131072\",mamba_cache_mode=\"none\"} 1.0\n",
        );
        assert_eq!(parsed.kv_cache_size_tokens, Some(131_072));
        assert_eq!(parsed.kv_cache_max_concurrency, None);
        assert_eq!(parsed.kv_block_size, Some(32));
        assert_eq!(parsed.mamba_cache_mode.as_deref(), Some("none"));

        let empty = parse_vllm_cache_config_info("# empty metrics\nvllm:num_requests_running 0\n");
        assert_eq!(empty, VllmCacheConfig::default());
    }

    #[test]
    fn kv_pool_budget_computes_mamba_headroom_and_per_slot_overhead() {
        let mut cfg = test_config(None, None);
        cfg.max_concurrent_tasks = 4;
        cfg.max_turn_reasoning_tokens = 16_384;

        let vllm_mamba = EngineProfile {
            backend: Backend::Vllm,
            model_id: "qwen3.5".into(),
            max_context: Some(150_000),
            capabilities: Backend::Vllm.capabilities(),
            engine_version: Some("0.18.0".into()),
            kv_cache_size_tokens: Some(808_448),
            kv_cache_max_concurrency: Some(5.38),
            kv_block_size: Some(512),
            mamba_cache_mode: Some("align".into()),
        };
        let b1 = vllm_mamba.kv_pool_budget(&cfg);
        assert_eq!(b1.pool_capacity_tokens, 727_603);
        assert_eq!(b1.per_slot_state_tokens, 512 * 2 + 8192);

        // Smaller pool where 2 * kv_block_size * max_concurrent_tasks (4096) > 10% (3276).
        let small_mamba = EngineProfile {
            kv_cache_size_tokens: Some(32_768),
            ..vllm_mamba.clone()
        };
        let b2 = small_mamba.kv_pool_budget(&cfg);
        assert_eq!(b2.pool_capacity_tokens, 32_768 - 4096);

        // mamba_cache_mode="none" skips 10% headroom, still reserves per-slot overhead for vLLM.
        let vllm_plain = EngineProfile {
            mamba_cache_mode: Some("none".into()),
            kv_block_size: Some(16),
            ..vllm_mamba
        };
        let b3 = vllm_plain.kv_pool_budget(&cfg);
        assert_eq!(b3.pool_capacity_tokens, 808_448);
        assert_eq!(b3.per_slot_state_tokens, 16 * 2 + 8192);

        // Non-vLLM uses config.max_context and 0 per-slot overhead.
        cfg.max_context = Some(65_536);
        let llama = EngineProfile {
            backend: Backend::LlamaCpp,
            model_id: "fixture".into(),
            max_context: Some(65_536),
            capabilities: Backend::LlamaCpp.capabilities(),
            engine_version: None,
            kv_cache_size_tokens: None,
            kv_cache_max_concurrency: None,
            kv_block_size: None,
            mamba_cache_mode: None,
        };
        let b4 = llama.kv_pool_budget(&cfg);
        assert_eq!(b4.pool_capacity_tokens, 65_536);
        assert_eq!(b4.per_slot_state_tokens, 0);
    }
}
