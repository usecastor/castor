//! DGI Gatekeeper — Decomposition Granularity Index.
//!
//! Classifies dispatch prompts with a serving-engine probe and returns a
//! strictly binary ADMIT / REJECT verdict: the gate either admits a bounded
//! single-concern spec or fails fast on a monolithic multi-subsystem
//! dispatch. It never injects advisory text into the worker prompt.
//!
//! The probe comes in two shapes, and they are intentionally not comparable in
//! latency: the vLLM path is a single-token guided_choice completion pass,
//! while the llama.cpp path is a schema-constrained JSON chat decode, whose
//! duration scales with decode speed on CPU models. Both shapes share one
//! request-builder / verdict-decoder core ([`probe_request`],
//! [`decode_verdict`]) so the decision rule stays DRY and engine-agnostic.

use serde_json::json;

/// Shared classification rubric for the DGI gatekeeper probe.
///
/// Emphasizes cohesion (one coherent concern within a single subsystem, however
/// many files it spans) versus unrelated multi-subsystem overhauls, so raw
/// prompt length is not the deciding signal. The compact contrastive
/// few-shot examples pin down the two failure modes of length-based
/// classifiers: a long-but-cohesive task must be ADMITted, a short-but-sprawly
/// multi-subsystem chain must be flagged OVERLOADED.
const CLASSIFY_RUBRIC: &str = concat!(
    "You are a decomposition gatekeeper. Classify the task below. ",
    "ADMIT: a cohesive, bounded single-subsystem engineering task — one coherent ",
    "concern, even if it touches several files within that one subsystem, or a focused ",
    "audit of a single module/subsystem. ",
    "OVERLOADED: (1) an unrelated multi-subsystem overhaul chaining distinct, ",
    "independent subsystems (e.g. redesign A, rewrite B, re-architect C) with no single ",
    "coherent goal; (2) a dispatch combining broad external web/paper research across ",
    "multiple domains with deep codebase surgery and hardware sizing; or (3) a broad ",
    "exhaustive read/audit across 2+ independent subsystems or >5 disparate files across ",
    "multiple unrelated review criteria (which should be split into per-subsystem slices). ",
    "The deciding test is cohesion and scope: if all work serves one bounded subsystem ",
    "or single concern it is ADMIT; if it spans multiple independent subsystems or an ",
    "exhaustive multi-file sweep it is OVERLOADED. ",
    "Examples: ",
    "ADMIT — 'Add retry-with-backoff to the DGI probe in src/mcp/dgi.rs and thread the ",
    "new config knob through src/config.rs; add unit tests and run cargo test.' ",
    "ADMIT — 'Read-only audit of src/proxy/mod.rs and src/proxy/sanitize.rs for SSE framing bugs.' ",
    "OVERLOADED — 'Rewrite the config loader, re-architect the engine client onto a new ",
    "wire protocol, and rework the task registry schema, all in one dispatch.' ",
    "OVERLOADED — 'Conduct an exhaustive line-by-line audit across src/runner/, src/engine/, ",
    "src/task/, and src/proxy/ across 4 orthogonal criteria, or audit 14 website files across 5 criteria.' ",
);

/// Single-token guided-choice vocabulary for the DGI probe.
const VERDICT_ADMIT: &str = "ADMIT";
const VERDICT_OVERLOADED: &str = "OVERLOADED";

/// Structured DGI evaluation result including optional log-odds margin and source.
#[derive(Debug, Clone, PartialEq)]
pub struct DgiEvaluation {
    pub verdict: DgiVerdict,
    pub margin: Option<f64>,
    pub source: &'static str,
}

/// Verdict of the DGI Gatekeeper for a dispatch prompt. Strictly binary: the
/// gate either admits a bounded single-concern spec or fails fast on a
/// monolithic multi-subsystem dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum DgiVerdict {
    /// Bounded single-concern spec: proceed without commentary.
    Admit,
    /// The model classified an overloaded dispatch: fail fast.
    Reject(Vec<String>),
}

impl DgiVerdict {
    pub fn is_reject(&self) -> bool {
        matches!(self, Self::Reject(_))
    }
}

/// Turn the probe's first decoded token into a verdict (used only when an
/// endpoint does not return `top_logprobs`). Rejects empty or unrecognized
/// text rather than silently admitting.
fn verdict_from_text(token: &str) -> Result<DgiVerdict, String> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Err("DGI probe returned empty completion text".into());
    }
    if is_choice_prefix(trimmed, VERDICT_ADMIT, 2) {
        Ok(DgiVerdict::Admit)
    } else if is_choice_prefix(trimmed, VERDICT_OVERLOADED, 2) {
        Ok(DgiVerdict::Reject(vec![
            "Model-classified monolithic multi-subsystem dispatch (OVERLOADED)".to_string(),
        ]))
    } else {
        Err(format!("unexpected DGI probe completion text: {trimmed:?}"))
    }
}

/// Returns `true` when `token` is a non-empty case-normalized prefix of `target`
/// of at least `min_len` characters (ignoring leading/trailing whitespace and quotes).
fn is_choice_prefix(token: &str, target: &str, min_len: usize) -> bool {
    let t = token.trim().trim_matches('"');
    if t.len() >= min_len && target.len() >= t.len() && target[..t.len()].eq_ignore_ascii_case(t) {
        return true;
    }
    t.eq_ignore_ascii_case(target)
}

fn logsumexp(a: Option<f64>, b: f64) -> f64 {
    match a {
        None => b,
        Some(prev) => {
            let m = prev.max(b);
            m + ((prev - m).exp() + (b - m).exp()).ln()
        }
    }
}

/// Extract `(token, logprob)` pairs from either `/v1/completions` map format
/// (`{"AD": -0.01, "OVER": -5.3, ...}`) or Chat Completions array format
/// (`[{"token": " ADMIT", "logprob": -0.01}, ...]`).
fn extract_top_logprobs(top_logprobs: &serde_json::Value) -> Option<Vec<(String, f64)>> {
    if let Some(obj) = top_logprobs.as_object() {
        let pairs: Vec<(String, f64)> = obj
            .iter()
            .filter_map(|(k, v)| v.as_f64().map(|lp| (k.clone(), lp)))
            .collect();
        return if pairs.is_empty() { None } else { Some(pairs) };
    }
    if let Some(arr) = top_logprobs.as_array() {
        let pairs: Vec<(String, f64)> = arr
            .iter()
            .filter_map(|e| {
                let tok = e.get("token")?.as_str()?.to_string();
                let lp = e.get("logprob")?.as_f64()?;
                Some((tok, lp))
            })
            .collect();
        return if pairs.is_empty() { None } else { Some(pairs) };
    }
    None
}

/// Binary log-odds decision rule: `delta_z = logP(OVERLOADED) - logP(ADMIT)`
/// measured on the probe's first token. Reject when `delta_z > reject_margin`;
/// otherwise admit. Supports multi-token BPE first-token prefixes
/// (`AD`/`OVER`) via `logsumexp`, and handles one-sided top-K truncation using
/// `min(top_logprobs)` as a strict upper bound when the losing choice falls
/// outside top-K.
fn verdict_and_margin_from_logprobs(
    top_logprobs: &serde_json::Value,
    reject_margin: f64,
) -> Result<(DgiVerdict, f64), String> {
    let pairs = extract_top_logprobs(top_logprobs)
        .ok_or_else(|| "top_logprobs empty or unrecognized format".to_string())?;

    let mut p_admit: Option<f64> = None;
    let mut p_over: Option<f64> = None;
    let mut floor_lp = f64::INFINITY;

    for (tok, lp) in &pairs {
        floor_lp = floor_lp.min(*lp);
        // Restrict prefix pooling to canonical uppercase BPE prefixes (`AD`/`ADM`/`ADMIT`,
        // `OVER`/`OVERLOADED`) so stray lowercase/partial tokens don't skew logsumexp; fall
        // back to case-insensitive match only if exact-case yielded nothing.
        let t = tok.trim().trim_matches('"');
        if t.len() >= 2 && VERDICT_ADMIT.starts_with(t) {
            p_admit = Some(logsumexp(p_admit, *lp));
        } else if t.len() >= 2 && VERDICT_OVERLOADED.starts_with(t) {
            p_over = Some(logsumexp(p_over, *lp));
        }
    }

    if p_admit.is_none() || p_over.is_none() {
        for (tok, lp) in &pairs {
            if p_admit.is_none() && is_choice_prefix(tok, VERDICT_ADMIT, 2) {
                p_admit = Some(logsumexp(p_admit, *lp));
            } else if p_over.is_none() && is_choice_prefix(tok, VERDICT_OVERLOADED, 2) {
                p_over = Some(logsumexp(p_over, *lp));
            }
        }
    }

    // Fallback for tokenizers that split choices at a 1-char first token ("A"/"O").
    if p_admit.is_none() || p_over.is_none() {
        for (tok, lp) in &pairs {
            let t = tok.trim().trim_matches('"');
            if p_admit.is_none() && t == "A" {
                p_admit = Some(*lp);
            } else if p_over.is_none() && t == "O" {
                p_over = Some(*lp);
            }
        }
    }

    match (p_admit, p_over) {
        (Some(a), Some(o)) => {
            let delta_z = o - a;
            if delta_z > reject_margin {
                Ok((
                    DgiVerdict::Reject(vec![format!(
                        "Model-classified monolithic multi-subsystem dispatch (OVERLOADED, log-odds margin {delta_z:.3} > {reject_margin})"
                    )]),
                    delta_z,
                ))
            } else {
                Ok((DgiVerdict::Admit, delta_z))
            }
        }
        // Only ADMIT appeared in top-K: OVERLOADED logprob is strictly <= floor_lp.
        (Some(a), None) if (floor_lp - a) < -reject_margin => Ok((DgiVerdict::Admit, floor_lp - a)),
        // Only OVERLOADED appeared in top-K: ADMIT logprob is strictly <= floor_lp.
        (None, Some(o)) if (o - floor_lp) > reject_margin => {
            let lb = o - floor_lp;
            Ok((
                DgiVerdict::Reject(vec![format!(
                    "Model-classified monolithic multi-subsystem dispatch (OVERLOADED, lower-bound margin {lb:.3} > {reject_margin})"
                )]),
                lb,
            ))
        }
        _ => Err("top_logprobs missing probe choices".into()),
    }
}

#[cfg(test)]
fn verdict_from_logprobs(
    top_logprobs: &serde_json::Value,
    reject_margin: f64,
) -> Result<DgiVerdict, String> {
    verdict_and_margin_from_logprobs(top_logprobs, reject_margin).map(|(v, _)| v)
}

/// Parse the probe response: log-odds margin when `top_logprobs` is present,
/// decoded text otherwise.
fn decode_verdict_with_margin(
    val: &serde_json::Value,
    reject_margin: f64,
) -> Result<(DgiVerdict, Option<f64>), String> {
    let choice = &val["choices"][0];
    if let Some(lps) = choice["logprobs"]["top_logprobs"].as_array() {
        let first = lps.first().ok_or("empty top_logprobs")?;
        verdict_and_margin_from_logprobs(first, reject_margin).map(|(v, m)| (v, Some(m)))
    } else {
        let text = choice["text"].as_str().unwrap_or("");
        verdict_from_text(text).map(|v| (v, None))
    }
}

#[cfg(test)]
fn decode_verdict(val: &serde_json::Value, reject_margin: f64) -> Result<DgiVerdict, String> {
    decode_verdict_with_margin(val, reject_margin).map(|(v, _)| v)
}

/// The one request builder for both probe shapes (the only DRY seam):
/// the guided-choice completion body or the schema-constrained chat body.
fn probe_request(
    probe_shape: crate::engine::backend::ProbeShape,
    model: &str,
    prompt: &str,
) -> serde_json::Value {
    match probe_shape {
        crate::engine::backend::ProbeShape::GuidedChoiceCompletion => json!({
            "model": model,
            "prompt": format!(
                "\u{3c}|im_start|\u{3e}system\n{CLASSIFY_RUBRIC}Answer with only ADMIT or OVERLOADED.\u{3c}|im_end|\u{3e}\n\u{3c}|im_start|\u{3e}user\n{prompt}\u{3c}|im_end|\u{3e}\n\u{3c}|im_start|\u{3e}assistant\n\u{3c}think\u{3e}\n\u{3c}/think\u{3e}\n"
            ),
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 20,
            "guided_choice": [VERDICT_ADMIT, VERDICT_OVERLOADED],
        }),
        crate::engine::backend::ProbeShape::JsonSchemaChat => json!({
            "model": model,
            "messages": [
                {"role": "system", "content": format!("{CLASSIFY_RUBRIC}Return only JSON with a verdict field.")},
                {"role": "user", "content": prompt}
            ],
            "stream": false,
            "temperature": 0.0,
            "max_tokens": 64,
            "reasoning_effort": "none",
            "logprobs": true,
            "top_logprobs": 20,
            "response_format": {"type": "json_object", "schema": {
                "type": "object", "properties": {"verdict": {"type": "string", "enum": [VERDICT_ADMIT, VERDICT_OVERLOADED]}},
                "required": ["verdict"], "additionalProperties": false
            }}
        }),
    }
}

/// Extract a calibrated log-odds verdict from `choices[0].logprobs.content`
/// on Chat Completions (`JsonSchemaChat`) responses. Locates the token step
/// where the verdict value (`ADMIT` / `OVERLOADED`) begins and feeds its
/// quote-trimmed `top_logprobs` array into [`verdict_from_logprobs`].
fn decode_chat_logprobs_with_margin(
    choice: &serde_json::Value,
    reject_margin: f64,
) -> Option<(DgiVerdict, f64)> {
    let steps = choice.get("logprobs")?.get("content")?.as_array()?;
    fn normalize(tok: &str) -> &str {
        tok.trim().trim_matches('"')
    }

    let verdict_step = steps
        .iter()
        .find(|step| {
            let Some(tok) = step.get("token").and_then(|v| v.as_str()) else {
                return false;
            };
            let t = normalize(tok);
            is_choice_prefix(t, VERDICT_ADMIT, 2) || is_choice_prefix(t, VERDICT_OVERLOADED, 2)
        })
        .or_else(|| {
            steps.iter().find(|step| {
                let Some(tok) = step.get("token").and_then(|v| v.as_str()) else {
                    return false;
                };
                matches!(normalize(tok), "A" | "O")
            })
        })?;

    let top_lps = verdict_step.get("top_logprobs")?.as_array()?;
    let normalized_lps: Vec<serde_json::Value> = top_lps
        .iter()
        .filter_map(|entry| {
            let tok = entry.get("token")?.as_str()?;
            let lp = entry.get("logprob")?.as_f64()?;
            Some(json!({
                "token": normalize(tok),
                "logprob": lp,
            }))
        })
        .collect();
    if normalized_lps.is_empty() {
        return None;
    }
    verdict_and_margin_from_logprobs(&serde_json::Value::Array(normalized_lps), reject_margin).ok()
}

/// Decode the probe response for either shape: the guided-choice shape reads
/// the first decoded token (with a logprobs margin when available); the chat
/// shape validates the JSON verdict field and scores `choices[0].logprobs.content`
/// when present, falling back to the discrete JSON verdict when omitted.
fn decode_probe_response_with_margin(
    probe_shape: crate::engine::backend::ProbeShape,
    val: &serde_json::Value,
    reject_margin: f64,
) -> Result<(DgiVerdict, Option<f64>), String> {
    match probe_shape {
        crate::engine::backend::ProbeShape::GuidedChoiceCompletion => {
            decode_verdict_with_margin(val, reject_margin)
        }
        crate::engine::backend::ProbeShape::JsonSchemaChat => {
            let choice = &val["choices"][0];
            if choice["finish_reason"] == "length" {
                return Err("dispatch gate response exceeded its token limit".into());
            }
            let content = choice["message"]["content"]
                .as_str()
                .ok_or("missing dispatch gate content")?;
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct ProbeVerdict {
                verdict: String,
            }
            let verdict: ProbeVerdict = serde_json::from_str(content)
                .map_err(|e| format!("malformed dispatch gate verdict: {e}"))?;
            let discrete = match verdict.verdict.as_str() {
                "ADMIT" => DgiVerdict::Admit,
                "OVERLOADED" => DgiVerdict::Reject(vec![
                    "Model-classified monolithic multi-subsystem dispatch (OVERLOADED)".into(),
                ]),
                _ => return Err("dispatch gate verdict must be ADMIT or OVERLOADED".into()),
            };
            Ok(
                match decode_chat_logprobs_with_margin(choice, reject_margin) {
                    Some((v, m)) => (v, Some(m)),
                    None => (discrete, None),
                },
            )
        }
    }
}

fn decode_probe_response(
    probe_shape: crate::engine::backend::ProbeShape,
    val: &serde_json::Value,
    reject_margin: f64,
) -> Result<DgiVerdict, String> {
    decode_probe_response_with_margin(probe_shape, val, reject_margin).map(|(v, _)| v)
}

/// Query the serving engine for a single-token logit probe.
///
/// Uses the vLLM guided_choice completion format with a one-token output
/// budget and `logprobs: 2` so the verdict is a calibrated log-odds margin.
pub async fn evaluate_model_probe(
    base_url: &str,
    model: &str,
    prompt: &str,
    reject_margin: f64,
    http: &reqwest::Client,
) -> Result<DgiVerdict, String> {
    let url = format!("{}/completions", base_url.trim_end_matches('/'));
    let req_body = probe_request(
        crate::engine::backend::ProbeShape::GuidedChoiceCompletion,
        model,
        prompt,
    );
    let resp = http
        .post(&url)
        .json(&req_body)
        .send()
        .await
        .map_err(|e| format!("connection to {url} failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("upstream HTTP {status}: {body}"));
    }
    let val: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("malformed JSON response from {url}: {e}"))?;
    decode_probe_response(
        crate::engine::backend::ProbeShape::GuidedChoiceCompletion,
        &val,
        reject_margin,
    )
}

/// Query the configured serving engine using its capability-selected probe
/// format, returning both the binary verdict and raw log-odds margin for telemetry.
pub async fn evaluate_configured_model_probe_with_metrics(
    config: &crate::config::Config,
    prompt: &str,
    http: &reqwest::Client,
) -> Result<DgiEvaluation, String> {
    use crate::engine::backend::{Backend, ProbeShape, api_url};
    let model = config.model.as_deref().ok_or("missing model in config")?;
    let shape = Backend::from_config(config).probe_shape();
    let url = match shape {
        ProbeShape::GuidedChoiceCompletion => api_url(config, "completions")?,
        ProbeShape::JsonSchemaChat => api_url(config, "chat/completions")?,
    };
    let payload = probe_request(shape, model, prompt);

    let mut attempt = 0;
    loop {
        let mut request = http.post(&url).json(&payload);
        if let Some(key) = &config.api_key {
            request = request.bearer_auth(key);
        }
        match request.send().await {
            Ok(response) => {
                let status = response.status();
                let body = response
                    .text()
                    .await
                    .map_err(|e| format!("reading upstream HTTP {status} body: {e}"))?;
                if (status.as_u16() == 502 || status.as_u16() == 503 || status.as_u16() == 504)
                    && attempt == 0
                {
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    continue;
                }
                if !status.is_success() {
                    return Err(format!("upstream HTTP {status}: {body}"));
                }
                let val: serde_json::Value = serde_json::from_str(&body)
                    .map_err(|e| format!("malformed probe response: {e}"))?;
                let (verdict, margin) =
                    decode_probe_response_with_margin(shape, &val, config.dgi_reject_margin)?;
                return Ok(DgiEvaluation {
                    verdict,
                    margin,
                    source: "probe",
                });
            }
            Err(e) if attempt == 0 => {
                attempt += 1;
                tracing::warn!(url = %url, "DGI probe transport error ({e}); retrying once");
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
            Err(e) => return Err(format!("connection to {url} failed: {e}")),
        }
    }
}

/// Query the configured serving engine using its capability-selected probe
/// format, retrying once on transient socket/5xx blips before failing fast.
pub async fn evaluate_configured_model_probe(
    config: &crate::config::Config,
    prompt: &str,
    http: &reqwest::Client,
) -> Result<DgiVerdict, String> {
    evaluate_configured_model_probe_with_metrics(config, prompt, http)
        .await
        .map(|eval| eval.verdict)
}

/// Warm-session bypass: a session that already contains at least one prior
/// `assistant` turn in its persisted `messages.json` is a *continuation*,
/// not a fresh dispatch — admit it without probing. Empty arrays (`[]`) or
/// user-only crash stubs never bypass DGI.
pub fn warm_session_bypass(
    config: &crate::config::Config,
    state: &crate::state::StateDir,
    session_id: &str,
) -> bool {
    if !config.dgi_bypass_warm_session {
        return false;
    }
    let path = state.sessions().join(session_id).join("messages.json");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return false;
    };
    let Ok(msgs) = serde_json::from_str::<Vec<serde_json::Value>>(&raw) else {
        return false;
    };
    msgs.iter()
        .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_completions_map_top_logprobs_decodes_bpe_prefixes() {
        let top = json!({
            "AD": -0.0081,
            "ADM": -5.88,
            "ad": -6.38,
            "OVER": -5.3831
        });
        assert_eq!(verdict_from_logprobs(&top, 1.5).unwrap(), DgiVerdict::Admit);

        let top_mon = json!({
            "OVER": -0.1,
            "AD": -2.5
        });
        assert!(verdict_from_logprobs(&top_mon, 1.5).unwrap().is_reject());
    }

    #[test]
    fn vllm_completions_one_sided_top_logprobs_bounds_missing_choice() {
        // Only AD appeared in top-20; floor is -7.13 -> margin <= -7.12 < -1.5 -> Admit.
        let top = json!({
            "AD": -0.0075,
            "ADM": -5.88,
            "The": -7.13
        });
        assert_eq!(verdict_from_logprobs(&top, 1.5).unwrap(), DgiVerdict::Admit);

        // Only OVER appeared in top-20; floor is -6.0 -> margin >= 5.9 > 1.5 -> Reject.
        let top_over = json!({
            "OVER": -0.1,
            "The": -6.0
        });
        assert!(verdict_from_logprobs(&top_over, 1.5).unwrap().is_reject());
    }

    #[test]
    fn text_fallback_rejects_empty_or_garbage_output() {
        assert!(verdict_from_text("").is_err());
        assert!(verdict_from_text("hello world").is_err());
        assert_eq!(verdict_from_text("AD").unwrap(), DgiVerdict::Admit);
        assert!(verdict_from_text("OVER").unwrap().is_reject());
    }

    #[test]
    fn logprob_margin_rejects_overloaded() {
        // OVERLOADED logprob -0.2 is less negative (more probable) than ADMIT -3.0.
        // delta_z = logP(OVERLOADED) - logP(ADMIT) = -0.2 - (-3.0) = 2.8 > 1.5 → REJECT.
        let top = json!([
            {"token": " ADMIT", "logprob": -3.0},
            {"token": " OVERLOADED", "logprob": -0.2}
        ]);
        assert!(verdict_from_logprobs(&top, 1.5).unwrap().is_reject());
    }

    #[test]
    fn logprob_margin_admits_when_admit_dominant() {
        // ADMIT logprob -0.1 is more probable than OVERLOADED -2.0.
        // delta_z = -2.0 - (-0.1) = -1.9. Not > 1.5, |−1.9| > 1.5 → Admit.
        let top = json!([
            {"token": " ADMIT", "logprob": -0.1},
            {"token": " OVERLOADED", "logprob": -2.0}
        ]);
        assert_eq!(verdict_from_logprobs(&top, 1.5).unwrap(), DgiVerdict::Admit);
    }

    #[test]
    fn logprob_margin_is_binary() {
        // Near-tie: delta_z = -0.7 - (-0.5) = -0.2 < 1.5 → Admit (binary rule).
        let top = json!([
            {"token": " ADMIT", "logprob": -0.5},
            {"token": " OVERLOADED", "logprob": -0.7}
        ]);
        assert_eq!(verdict_from_logprobs(&top, 1.5).unwrap(), DgiVerdict::Admit);
        // Strong ADMIT: delta_z = -1.5 - 0.0 = -1.5 < 1.5 → Admit.
        let top = json!([
            {"token": " ADMIT", "logprob": 0.0},
            {"token": " OVERLOADED", "logprob": -1.5}
        ]);
        assert_eq!(verdict_from_logprobs(&top, 1.5).unwrap(), DgiVerdict::Admit);
        // Mildly overloaded below reject_margin: delta_z = -0.2 - (-1.0) = 0.8 <= 1.5 → Admit.
        let top = json!([
            {"token": " ADMIT", "logprob": -1.0},
            {"token": " OVERLOADED", "logprob": -0.2}
        ]);
        assert_eq!(verdict_from_logprobs(&top, 1.5).unwrap(), DgiVerdict::Admit);
        // Just past: delta_z = 1.6 - 0.0 = 1.6 > 1.5 → Reject.
        let top = json!([
            {"token": " ADMIT", "logprob": -1.6},
            {"token": " OVERLOADED", "logprob": 0.0}
        ]);
        assert!(verdict_from_logprobs(&top, 1.5).unwrap().is_reject());
    }

    #[test]
    fn logprobs_missing_both_choices_errors() {
        let top = json!([{"token": " FOO", "logprob": -0.5}]);
        assert!(verdict_from_logprobs(&top, 1.5).is_err());
    }

    #[test]
    fn decode_verdict_falls_back_to_text_without_logprobs() {
        let val = json!({"choices": [{"text": "ADMIT", "finish_reason": "stop"}]});
        assert_eq!(decode_verdict(&val, 1.5).unwrap(), DgiVerdict::Admit);
        let val = json!({"choices": [{"text": "OVERLOADED", "finish_reason": "stop"}]});
        assert!(decode_verdict(&val, 1.5).unwrap().is_reject());
    }

    #[test]
    fn warm_session_bypass_requires_flag_file_and_content() {
        let dir = std::env::temp_dir().join(format!(
            "castor_dgi_warm_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let state = crate::state::StateDir::new(&dir);
        let _ = state.ensure();
        let loaded = crate::config::load().unwrap();
        let mut config = loaded.config;
        config.state_dir = dir.clone();
        config.dgi_bypass_warm_session = true;

        let sid = "sess_warm";
        let sess_dir = state.sessions().join(sid);
        std::fs::create_dir_all(&sess_dir).unwrap();

        // Disabled flag → never bypasses.
        config.dgi_bypass_warm_session = false;
        std::fs::write(
            sess_dir.join("messages.json"),
            r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}]"#,
        )
        .unwrap();
        assert!(!warm_session_bypass(&config, &state, sid));

        // Enabled, has prior assistant turn → bypass.
        config.dgi_bypass_warm_session = true;
        assert!(warm_session_bypass(&config, &state, sid));

        // Enabled, empty JSON array or user-only history → no bypass.
        std::fs::write(sess_dir.join("messages.json"), "[]").unwrap();
        assert!(!warm_session_bypass(&config, &state, sid));
        std::fs::write(
            sess_dir.join("messages.json"),
            r#"[{"role":"user","content":"hi"}]"#,
        )
        .unwrap();
        assert!(!warm_session_bypass(&config, &state, sid));

        // Enabled, file empty → no bypass.
        std::fs::write(sess_dir.join("messages.json"), "").unwrap();
        assert!(!warm_session_bypass(&config, &state, sid));

        // Enabled, file absent → no bypass.
        let _ = std::fs::remove_file(sess_dir.join("messages.json"));
        assert!(!warm_session_bypass(&config, &state, sid));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zero_margin_rejects_whenever_overloaded_beats_admit() {
        let top = json!({
            "OVER": -0.2882,
            "AD": -1.5382,
            "ADM": -3.5382
        });
        assert!(verdict_from_logprobs(&top, 0.0).unwrap().is_reject());
        assert_eq!(verdict_from_logprobs(&top, 1.2).unwrap(), DgiVerdict::Admit);
    }

    async fn probe_server(response: serde_json::Value) -> (String, reqwest::Client) {
        use axum::Json;
        use axum::Router;
        use axum::routing::post;

        let app = Router::new().route(
            "/completions",
            post(move |_body: Json<serde_json::Value>| {
                let r = response.clone();
                async move { Json(r) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), reqwest::Client::new())
    }

    #[tokio::test]
    async fn model_probe_rejects_on_overloaded() {
        let (base_url, http) = probe_server(json!({
            "choices": [{"text": "OVERLOADED", "finish_reason": "stop"}]
        }))
        .await;
        let verdict = evaluate_model_probe(&base_url, "test-model", "mega task", 1.5, &http)
            .await
            .unwrap();
        assert!(verdict.is_reject());
    }

    #[tokio::test]
    async fn model_probe_admits_on_admissible() {
        let (base_url, http) = probe_server(json!({
            "choices": [{"text": "ADMIT", "finish_reason": "stop"}]
        }))
        .await;
        let verdict = evaluate_model_probe(&base_url, "test-model", "single slice", 1.5, &http)
            .await
            .unwrap();
        assert_eq!(verdict, DgiVerdict::Admit);
    }

    #[tokio::test]
    async fn model_probe_requests_logprobs_and_decodes_margin() {
        // logprobs present: the decoded text says ADMIT and the margin
        // delta_z = -0.6 - (-0.5) = -0.1 < 1.5 → Admit.
        let (base_url, http) = probe_server(json!({
            "choices": [{
                "text": "ADMIT",
                "finish_reason": "stop",
                "logprobs": {"top_logprobs": [[
                    {"token": " ADMIT", "logprob": -0.5},
                    {"token": " OVERLOADED", "logprob": -0.6}
                ]]}
            }]
        }))
        .await;
        let verdict = evaluate_model_probe(&base_url, "test-model", "task", 1.5, &http)
            .await
            .unwrap();
        assert_eq!(verdict, DgiVerdict::Admit);
    }

    #[test]
    fn guided_choice_probe_request_uses_integer_logprobs_without_chat_top_logprobs() {
        let req = probe_request(
            crate::engine::backend::ProbeShape::GuidedChoiceCompletion,
            "qwen3.8-27b",
            "fix bug",
        );
        assert_eq!(req["logprobs"].as_u64(), Some(20));
        assert!(
            req.get("top_logprobs").is_none(),
            "completions endpoint must not receive chat-only top_logprobs field"
        );
        assert_eq!(req["guided_choice"], json!(["ADMIT", "OVERLOADED"]));
    }

    /// Fixture-integrity test for the DGI calibration anchors in
    /// `dgi_anchors.json` (regenerated by `scripts/gen_dgi_fixture.py`).
    ///
    /// The anchors are the live model probe's ground-truth labels: 4
    /// audit-verified false positives (bounded single-concern specs that must
    /// be ADMITted) and 3 genuine monoliths (coupled multi-subsystem sprawl that
    /// must be REJECTed). This test asserts fixture *integrity* only — it parses
    /// the file, checks the class taxonomy, confirms the 4/3 split, and verifies
    /// each prompt's character length matches the recorded `expected_len`.
    #[test]
    fn dgi_anchors_fixture_integrity() {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Anchor {
            class: String,
            name: String,
            expected_len: usize,
            prompt: String,
        }

        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fixture {
            #[allow(dead_code)]
            description: String,
            anchors: std::collections::BTreeMap<String, Anchor>,
        }

        const FIXTURE: &str = include_str!("dgi_anchors.json");
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("dgi_anchors.json must parse");

        // The fixture description promises exactly 7 anchors: 4 FP + 3 MON.
        let total = fixture.anchors.len();
        let fp = fixture.anchors.values().filter(|a| a.class == "FP").count();
        let mon = fixture
            .anchors
            .values()
            .filter(|a| a.class == "MON")
            .count();
        assert_eq!(total, 7, "expected 7 calibration anchors, found {total}");
        assert_eq!(fp, 4, "expected 4 false positives (FP), found {fp}");
        assert_eq!(mon, 3, "expected 3 monoliths (MON), found {mon}");

        // Per-anchor: closed class taxonomy, non-empty prompt, and a character
        // length matching the recorded expected_len. The generator records char
        // counts (not byte counts), so several anchors with non-ASCII would fail
        // a byte-length assertion — use chars().
        for (key, a) in &fixture.anchors {
            assert!(
                matches!(a.class.as_str(), "FP" | "MON"),
                "anchor {key} ({}) has invalid class {:?}",
                a.name,
                a.class
            );
            assert!(
                !a.prompt.trim().is_empty(),
                "anchor {key} ({}) has an empty prompt",
                a.name
            );
            let chars = a.prompt.chars().count();
            assert_eq!(
                chars, a.expected_len,
                "anchor {key} ({}) prompt length drifted: {chars} chars != recorded {}",
                a.name, a.expected_len
            );
        }
    }

    #[test]
    fn single_char_first_token_fallback_decodes_when_no_multi_char_match() {
        let top = json!({
            "A": -0.05,
            "O": -4.0
        });
        assert_eq!(verdict_from_logprobs(&top, 1.5).unwrap(), DgiVerdict::Admit);
    }

    #[tokio::test]
    async fn configured_probe_retries_once_on_transient_503() {
        use axum::Json;
        use axum::Router;
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        use axum::routing::post;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_ref = calls.clone();
        let app = Router::new().route(
            "/v1/completions",
            post(move || {
                let c = calls_ref.fetch_add(1, Ordering::SeqCst);
                async move {
                    if c == 0 {
                        (StatusCode::SERVICE_UNAVAILABLE, "overloaded".to_string()).into_response()
                    } else {
                        Json(json!({
                            "choices": [{
                                "text": "AD",
                                "finish_reason": "length",
                                "logprobs": {"top_logprobs": [{"AD": -0.01, "OVER": -4.5}]}
                            }]
                        }))
                        .into_response()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let loaded = crate::config::load().expect("load default config");
        let mut cfg = loaded.config;
        cfg.base_url = Some(format!("http://{addr}/v1"));
        cfg.model = Some("qwen3.8-27b".into());
        cfg.engine_type = Some("vllm".into());

        let verdict = evaluate_configured_model_probe(&cfg, "fix bug", &reqwest::Client::new())
            .await
            .unwrap();
        assert_eq!(verdict, DgiVerdict::Admit);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn json_schema_chat_probe_requests_logprobs_and_decodes_binary() {
        let req = probe_request(
            crate::engine::backend::ProbeShape::JsonSchemaChat,
            "Ternary-Bonsai-2-9B-Q2_0",
            "refactor subsystem",
        );
        assert_eq!(req["logprobs"], json!(true));
        assert_eq!(req["top_logprobs"], json!(20));

        // Near-tie in choices[0].logprobs.content (delta_z = -0.25 - (-0.50) = +0.25 <= 1.2) -> Admit.
        let chat_admit = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": r#"{"verdict":"OVERLOADED"}"#},
                "logprobs": {
                    "content": [
                        {"token": "{\"", "logprob": 0.0, "top_logprobs": [{"token": "{\"", "logprob": 0.0}]},
                        {"token": "verdict", "logprob": 0.0, "top_logprobs": [{"token": "verdict", "logprob": 0.0}]},
                        {"token": "\":\"", "logprob": 0.0, "top_logprobs": [{"token": "\":\"", "logprob": 0.0}]},
                        {"token": "OVER", "logprob": -0.25, "top_logprobs": [
                            {"token": "OVER", "logprob": -0.25},
                            {"token": "AD", "logprob": -0.50}
                        ]},
                        {"token": "LOADED", "logprob": 0.0, "top_logprobs": [{"token": "LOADED", "logprob": 0.0}]}
                    ]
                }
            }]
        });
        // The logprob margin (0.25 <= 1.2) overrides the discrete JSON token -> Admit.
        assert_eq!(
            decode_probe_response(
                crate::engine::backend::ProbeShape::JsonSchemaChat,
                &chat_admit,
                1.2,
            )
            .unwrap(),
            DgiVerdict::Admit
        );

        // Strong OVERLOADED margin (delta_z = -0.25 - (-1.583) = 1.333 > 1.2) -> Reject.
        let chat_reject = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": r#"{"verdict":"OVERLOADED"}"#},
                "logprobs": {
                    "content": [
                        {"token": "{\"verdict\":\"", "logprob": 0.0, "top_logprobs": [{"token": "{\"verdict\":\"", "logprob": 0.0}]},
                        {"token": "\"OVER", "logprob": -0.25, "top_logprobs": [
                            {"token": "\"OVER", "logprob": -0.25},
                            {"token": "\"AD", "logprob": -1.583}
                        ]}
                    ]
                }
            }]
        });
        assert!(
            decode_probe_response(
                crate::engine::backend::ProbeShape::JsonSchemaChat,
                &chat_reject,
                1.2,
            )
            .unwrap()
            .is_reject()
        );

        // When logprobs are omitted, discrete JSON verdict fallback applies.
        let chat_no_lps = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": r#"{"verdict":"ADMIT"}"#}
            }]
        });
        assert_eq!(
            decode_probe_response(
                crate::engine::backend::ProbeShape::JsonSchemaChat,
                &chat_no_lps,
                1.2,
            )
            .unwrap(),
            DgiVerdict::Admit
        );
    }
}
