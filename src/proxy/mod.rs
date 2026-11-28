//! Stream proxy: reverse-proxy to the engine with SSE sanitization.
//!
//! - Listens on `config.ports.proxy` (loopback only).
//! - Forwards `/v1/chat/completions` with `stream:true` to the engine and
//!   sanitizes the SSE response (UTF-8 reassembly, mid-stream error
//!   translation, multimodal guard, repetition circuit breaker).
//! - All other requests are byte-passthrough (including upstream 4xx, which
//!   are forwarded verbatim — never a fake 200 or synthetic finish_reason).
//!
//! Singleton: acquires `Locks::acquire("proxy")` at startup. If the lock is
//! held by another live process, the server exits 0 with a single stderr
//! line. Stale locks are reclaimed via tombstone (see `state::Locks`).

pub mod sanitize;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing;
use futures_util::stream::unfold;
use serde_json::json;

use crate::proxy::sanitize::{SseSanitizer, sanitize_request_body};
use crate::state::{AcquireResult, Locks, StateDir};

/// Inbound request body limit (50 MB).
const MAX_BODY_BYTES: usize = 50 * 1024 * 1024;

/// Singleton lock TTL (30 s).
const LOCK_TTL_MS: u64 = 30_000;

/// The stream proxy, cheaply cloneable (all state behind `Arc`).
#[derive(Clone)]
pub struct ProxyServer {
    inner: Arc<Inner>,
}

struct Inner {
    upstream_root: String,
    locks: Locks,
    state_dir: PathBuf,
    client: reqwest::Client,
}

impl ProxyServer {
    pub fn new(state: &StateDir, upstream: SocketAddr) -> Self {
        Self::from_upstream_url(state, format!("http://{upstream}"))
    }

    pub fn from_upstream_url(state: &StateDir, upstream_root: impl Into<String>) -> Self {
        let locks = Locks::from_state(state);
        let client = reqwest::Client::new();
        let root = upstream_root.into().trim_end_matches('/').to_string();
        Self {
            inner: Arc::new(Inner {
                upstream_root: root,
                locks,
                state_dir: state.root().to_path_buf(),
                client,
            }),
        }
    }

    /// Acquire the singleton lock. Returns `true` if we own it, `false` if
    /// another live process holds it (caller should exit 0).
    pub fn try_acquire_lock(&self) -> bool {
        match self.try_acquire_lock_result() {
            Ok(AcquireResult::Acquired) => true,
            Ok(AcquireResult::HeldBy { .. }) => false,
            Err(e) => {
                eprintln!("castor: proxy lock acquire error: {e}");
                false
            }
        }
    }

    fn try_acquire_lock_result(&self) -> std::io::Result<AcquireResult> {
        self.inner.locks.acquire_live("proxy", LOCK_TTL_MS)
    }

    /// Acquire the singleton lock with short backoff to absorb startup races
    /// when launched by `ensure_proxy_server_for_upstream`.
    pub async fn try_acquire_lock_with_retry(&self, retries: usize, delay_ms: u64) -> bool {
        for i in 0..=retries {
            if self.try_acquire_lock() {
                return true;
            }
            if i < retries {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
        false
    }

    /// Release the singleton lock (best-effort).
    pub fn release_lock(&self) {
        let _ = self.inner.locks.release("proxy");
    }

    /// Build the axum router.
    fn router(&self) -> Router {
        Router::new()
            .route("/health", routing::get(health_handler))
            .fallback(proxy_handler)
            .with_state(self.clone())
    }

    /// Listen on the given port (falling back to candidate ports if in use).
    /// Blocks until the server is dropped.
    pub async fn serve(self, port: u16) -> std::io::Result<()> {
        // Periodic lock heartbeat: refresh `"proxy"` lock every 10 s so other
        // instances never reclaim it as stale while we are alive.
        let heartbeat_handle = {
            let server = self.clone();
            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(Duration::from_secs(10));
                ticker.tick().await; // skip immediate tick
                loop {
                    ticker.tick().await;
                    match server.inner.locks.refresh("proxy", LOCK_TTL_MS) {
                        Ok(true) => {}
                        Ok(false) => {
                            let _ = server.try_acquire_lock();
                        }
                        Err(e) => {
                            eprintln!("[castor-proxy] lock refresh error: {e}");
                        }
                    }
                }
            })
        };

        let candidates = if port == 0 {
            vec![0]
        } else {
            vec![port, port.saturating_add(4), port.saturating_add(5), 0]
        };
        let mut bound_listener = None;
        for &p in &candidates {
            let addr = SocketAddr::from(([127, 0, 0, 1], p));
            match tokio::net::TcpListener::bind(addr).await {
                Ok(l) => {
                    bound_listener = Some(l);
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                    eprintln!("[castor-proxy] port {p} in use, trying next candidate");
                    continue;
                }
                Err(e) => {
                    heartbeat_handle.abort();
                    return Err(e);
                }
            }
        }
        let listener = match bound_listener {
            Some(l) => l,
            None => {
                heartbeat_handle.abort();
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AddrInUse,
                    "all candidate proxy ports in use",
                ));
            }
        };
        let bound_addr = listener.local_addr()?;
        let bound_port = bound_addr.port();
        eprintln!(
            "[castor-proxy] listening on {bound_addr} -> {}",
            self.inner.upstream_root
        );

        let ports_dir = self.inner.state_dir.join("ports");
        let _ = std::fs::create_dir_all(&ports_dir);
        let port_file = ports_dir.join("proxy.port");
        let tmp_port_file = ports_dir.join(format!(".proxy.port.tmp.{}", std::process::id()));
        if std::fs::write(&tmp_port_file, bound_port.to_string()).is_ok() {
            if std::fs::rename(&tmp_port_file, &port_file).is_err() {
                let _ = std::fs::remove_file(&tmp_port_file);
                let _ = std::fs::write(&port_file, bound_port.to_string());
            }
        } else {
            let _ = std::fs::write(&port_file, bound_port.to_string());
        }

        let res = axum::serve(listener, self.router()).await;
        heartbeat_handle.abort();
        let _ = std::fs::remove_file(&port_file);
        res
    }

    /// Listen on an ephemeral port (for tests). Returns the bound address.
    #[cfg(test)]
    pub async fn serve_ephemeral(self) -> std::io::Result<SocketAddr> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move {
            let _ = axum::serve(listener, self.router()).await;
        });
        Ok(addr)
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /health — local health check.
async fn health_handler(State(server): State<ProxyServer>) -> Response {
    let body = json!({
        "status": "ok",
        "service": "castor-proxy",
        "upstream": server.inner.upstream_root,
        "pid": std::process::id(),
    });
    (StatusCode::OK, body.to_string()).into_response()
}

/// Catch-all: forward to the upstream engine.
async fn proxy_handler(
    State(server): State<ProxyServer>,
    req: axum::extract::Request,
) -> Result<Response, StatusCode> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let query = req
        .uri()
        .query()
        .map(|q| format!("?{q}"))
        .unwrap_or_default();
    let headers = req.headers().clone();
    let body = req.into_body();

    // Read the inbound body (with a size limit).
    let body_bytes = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(b) => b,
        Err(_) => {
            return Ok((
                StatusCode::PAYLOAD_TOO_LARGE,
                json!({"error":{"message":"Payload Too Large","type":"payload_too_large","code":413}}).to_string(),
            )
                .into_response());
        }
    };

    // Determine if this is a streaming chat request.
    let is_chat = method == axum::http::Method::POST && path.starts_with("/v1/chat/completions");
    let accept_stream = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains("text/event-stream"));
    let body_stream = body_bytes
        .windows(15)
        .any(|w| w == b"\"stream\":true" || w == b"\"stream\": true");
    let is_stream = is_chat && (accept_stream || body_stream);

    // Multimodal guard: replace image blocks with text placeholders.
    let mut body_to_send = body_bytes.clone();
    if is_chat
        && let Ok(mut v) = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        && sanitize_request_body(&mut v)
    {
        body_to_send = Bytes::from(v.to_string());
    }

    // Forward to the upstream engine.
    let upstream_url = format!("{}{}{}", server.inner.upstream_root, path, query);
    let mut req_builder = server.inner.client.request(method, &upstream_url);
    for (name, value) in &headers {
        if name != axum::http::header::HOST && name != axum::http::header::CONTENT_LENGTH {
            req_builder = req_builder.header(name, value);
        }
    }
    let res = match req_builder.body(body_to_send).send().await {
        Ok(r) => r,
        Err(_) => {
            return Ok((
                StatusCode::BAD_GATEWAY,
                json!({"error":{"message":"upstream connection error","type":"upstream_connection_error"}}).to_string(),
            )
                .into_response());
        }
    };

    let status = res.status();
    let content_type = res
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let is_event_stream = content_type.contains("text/event-stream") || is_stream;

    // Build the response, forwarding the upstream status and headers.
    let mut resp_builder = Response::builder().status(status);
    for (name, value) in res.headers() {
        if name.as_str() != "content-length" && name.as_str() != "transfer-encoding" {
            resp_builder = resp_builder.header(name, value);
        }
    }

    if is_event_stream && status.is_success() {
        // Streaming SSE: sanitize the response.
        resp_builder = resp_builder
            .header("content-type", "text/event-stream; charset=utf-8")
            .header("cache-control", "no-cache, no-transform");
        let stream = sse_stream(res);
        Ok(resp_builder.body(Body::from_stream(stream)).unwrap())
    } else {
        // Byte passthrough (including upstream 4xx — verbatim, never faked).
        let stream = passthrough_stream(res);
        Ok(resp_builder.body(Body::from_stream(stream)).unwrap())
    }
}

// ---------------------------------------------------------------------------
// Stream builders
// ---------------------------------------------------------------------------

struct SseState {
    res: reqwest::Response,
    sanitizer: SseSanitizer,
    finished: bool,
}

/// Stream the upstream SSE response through the sanitizer.
fn sse_stream(
    res: reqwest::Response,
) -> impl futures_util::Stream<Item = Result<Bytes, Infallible>> {
    let state = SseState {
        res,
        sanitizer: SseSanitizer::new(),
        finished: false,
    };
    unfold(state, |mut state| async move {
        if state.finished {
            return None;
        }
        match state.res.chunk().await {
            Ok(Some(bytes)) => {
                let out = state.sanitizer.feed(&bytes);
                Some((Ok(Bytes::from(out)), state))
            }
            Ok(None) => {
                let out = state.sanitizer.finish();
                state.finished = true;
                Some((Ok(Bytes::from(out)), state))
            }
            Err(_) => {
                let out = "event: error\ndata: {\"error\":{\"message\":\"upstream error\",\"type\":\"upstream_error\"}}\n\n";
                state.finished = true;
                Some((Ok(Bytes::from(out)), state))
            }
        }
    })
}

struct PassthroughState {
    res: reqwest::Response,
    finished: bool,
}

/// Stream the upstream response body verbatim (byte passthrough).
fn passthrough_stream(
    res: reqwest::Response,
) -> impl futures_util::Stream<Item = Result<Bytes, Infallible>> {
    let state = PassthroughState {
        res,
        finished: false,
    };
    unfold(state, |mut state| async move {
        if state.finished {
            return None;
        }
        match state.res.chunk().await {
            Ok(Some(bytes)) => Some((Ok(bytes), state)),
            Ok(None) | Err(_) => None,
        }
    })
}

// ---------------------------------------------------------------------------
// Spawn helper (used by the `mcp` subcommand)
// ---------------------------------------------------------------------------

/// Derive the upstream server root URL (scheme + host + port + optional path
/// prefix, without trailing `/v1`) dynamically from CLI override or `Config`.
pub fn resolve_upstream_root(
    config: &crate::config::Config,
    engine_port_override: Option<u16>,
) -> String {
    if let Some(port) = engine_port_override {
        return format!("http://127.0.0.1:{port}");
    }
    if let Some(base) = config.base_url.as_deref()
        && let Some(root) = crate::engine::backend::server_root(base)
    {
        return root.trim_end_matches('/').to_string();
    }
    format!("http://127.0.0.1:{}", config.ports.engine)
}

/// Best-effort, non-blocking bring-up of the proxy, invoked by the `mcp`
/// subcommand before it starts serving.
///
/// The sequence is **lock → probe → spawn if down → release** (mirroring
/// `ensure_status_server`). Every step is best-effort: a failure is logged
/// to stderr and swallowed. This never blocks the MCP server from starting.
pub async fn ensure_proxy_server(state: &StateDir, proxy_port: u16, engine_port: u16) {
    ensure_proxy_server_for_upstream(
        state,
        proxy_port,
        &format!("http://127.0.0.1:{engine_port}"),
    )
    .await;
}

/// Read the active proxy port recorded by a running proxy server, falling back
/// to `default_port` if absent or invalid.
pub fn active_proxy_port(state: &StateDir, default_port: u16) -> u16 {
    let port_file = state.ports().join("proxy.port");
    std::fs::read_to_string(&port_file)
        .ok()
        .and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(default_port)
}

/// Same best-effort proxy bring-up using a resolved upstream URL root (supports
/// non-default `CASTOR_BASE_URL` hosts, ports, and path prefixes).
pub async fn ensure_proxy_server_for_upstream(
    state: &StateDir,
    proxy_port: u16,
    upstream_root: &str,
) {
    let server = ProxyServer::from_upstream_url(state, upstream_root);

    // 1. probe: check recorded active port first; if stale and different from
    //    the configured default port, remove the stale file and probe `proxy_port`.
    let recorded_port = active_proxy_port(state, proxy_port);
    let mut active_port = recorded_port;
    let mut already_up = probe_port(recorded_port).await;
    if !already_up && recorded_port != proxy_port {
        let _ = std::fs::remove_file(state.ports().join("proxy.port"));
        if probe_port(proxy_port).await {
            active_port = proxy_port;
            already_up = true;
        }
    }
    if already_up {
        eprintln!("[castor] proxy already up (port {active_port}); not spawning");
        return;
    }

    // 2. lock & spawn if down.
    match server.try_acquire_lock_result() {
        Ok(AcquireResult::Acquired) => {
            let p = active_proxy_port(state, proxy_port);
            if probe_port(p).await || (p != proxy_port && probe_port(proxy_port).await) {
                server.release_lock();
                return;
            }
            // Release before spawning so the detached child acquires cleanly
            // without waiting on a parent hold.
            server.release_lock();
            spawn_and_wait(state, proxy_port, upstream_root).await;
        }
        Ok(AcquireResult::HeldBy { pid }) => {
            eprintln!(
                "[castor] proxy unreachable on port {recorded_port}, lock held by pid {pid}; waiting briefly to retry"
            );
            for _ in 0..5 {
                tokio::time::sleep(Duration::from_millis(150)).await;
                let p = active_proxy_port(state, proxy_port);
                if probe_port(p).await || (p != proxy_port && probe_port(proxy_port).await) {
                    return;
                }
            }
            match server.try_acquire_lock_result() {
                Ok(AcquireResult::Acquired) => {
                    server.release_lock();
                    spawn_and_wait(state, proxy_port, upstream_root).await;
                }
                Ok(AcquireResult::HeldBy { pid: holder_pid }) => {
                    eprintln!(
                        "[castor] proxy still unreachable on port {recorded_port} (lock held by pid {holder_pid})"
                    );
                }
                Err(e) => {
                    eprintln!("[castor] proxy lock retry error: {e}");
                }
            }
        }
        Err(e) => {
            eprintln!("[castor] proxy lock acquire error: {e}");
        }
    }
}

async fn spawn_and_wait(state: &StateDir, proxy_port: u16, upstream_root: &str) {
    let Some(mut child) = spawn_detached(state, proxy_port, upstream_root) else {
        return;
    };
    for _ in 0..20 {
        let p = active_proxy_port(state, proxy_port);
        if probe_port(p).await || (p != proxy_port && probe_port(proxy_port).await) {
            return;
        }
        if let Ok(Some(exit_status)) = child.try_wait() {
            let p = active_proxy_port(state, proxy_port);
            if !probe_port(p).await && (p == proxy_port || !probe_port(proxy_port).await) {
                eprintln!(
                    "[castor] proxy child exited early ({exit_status}); see {}",
                    state.logs().join("proxy.log").display()
                );
            }
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    eprintln!(
        "[castor] proxy did not become ready within 2s (port {proxy_port}); see {}",
        state.logs().join("proxy.log").display()
    );
}

/// Probe the proxy port with a short-timeout HTTP GET on `/health`.
/// Returns `true` only if a Castor proxy responds with 2xx.
pub async fn probe_port(port: u16) -> bool {
    if port == 0 {
        return false;
    }
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(800))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let url = format!("http://127.0.0.1:{port}/health");
    match client.get(&url).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Spawn the proxy as a fully detached child process (new process group on
/// Unix, detached console on Windows, stdout/stderr appended to
/// `<state>/logs/proxy.log`). Never waits for the child.
fn spawn_detached(
    state: &StateDir,
    proxy_port: u16,
    upstream_root: &str,
) -> Option<std::process::Child> {
    let Some(exe) = std::env::current_exe().ok() else {
        eprintln!("[castor] cannot determine current exe; skipping proxy spawn");
        return None;
    };
    let state_str = state.root().to_string_lossy().into_owned();
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("proxy")
        .arg("--state-dir")
        .arg(&state_str)
        .arg("--port")
        .arg(proxy_port.to_string())
        .arg("--upstream")
        .arg(upstream_root)
        .stdin(std::process::Stdio::null());

    let logs_dir = state.logs();
    let _ = std::fs::create_dir_all(&logs_dir);
    let log_path = logs_dir.join("proxy.log");
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        Ok(file) => {
            if let Ok(dup) = file.try_clone() {
                cmd.stdout(dup);
            } else {
                cmd.stdout(std::process::Stdio::null());
            }
            cmd.stderr(file);
        }
        Err(e) => {
            eprintln!("[castor] failed to open {}: {e}", log_path.display());
            cmd.stdout(std::process::Stdio::null());
            cmd.stderr(std::process::Stdio::null());
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW (0x08000000) | CREATE_NEW_PROCESS_GROUP (0x00000200).
        cmd.creation_flags(0x0800_0200);
    }
    match cmd.spawn() {
        Ok(child) => {
            eprintln!("[castor] proxy spawned (port {proxy_port})");
            Some(child)
        }
        Err(e) => {
            eprintln!("[castor] failed to spawn proxy: {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::Query;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering as Ord};

    static TMP: AtomicUsize = AtomicUsize::new(0);

    fn tmp_state() -> StateDir {
        let n = TMP.fetch_add(1, Ord::SeqCst);
        let p = std::env::temp_dir().join(format!("castor-proxy-{}-{}", std::process::id(), n));
        let s = StateDir::new(p);
        s.ensure().unwrap();
        s
    }

    /// Start a mock upstream engine on an ephemeral port.
    async fn start_mock_upstream() -> SocketAddr {
        let app = Router::new().route("/v1/chat/completions", routing::post(mock_upstream_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        addr
    }

    /// Mock upstream: serves different test vectors based on the `vector`
    /// query parameter.
    async fn mock_upstream_handler(Query(q): Query<HashMap<String, String>>) -> Response {
        let vector = q.get("vector").cloned().unwrap_or_default();
        match vector.as_str() {
            "4xx" => (
                StatusCode::BAD_REQUEST,
                json!({"error":{"message":"bad request","type":"invalid_request"}}).to_string(),
            )
                .into_response(),
            "error" => {
                let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n\n";
                Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(body))
                    .unwrap()
            }
            "emoji" => {
                let full = "data: {\"choices\":[{\"delta\":{\"content\":\"😀\"}}]}\n\n";
                let bytes = full.as_bytes().to_vec();
                let emoji_bytes: &[u8] = &[0xF0, 0x9F, 0x98, 0x80];
                let pos = bytes
                    .windows(4)
                    .position(|w| w == emoji_bytes)
                    .expect("emoji not found");
                let split_at = pos + 2;
                let chunk1 = Bytes::from(bytes[..split_at].to_vec());
                let chunk2 = Bytes::from(bytes[split_at..].to_vec());
                let stream =
                    futures_util::stream::iter(vec![Ok::<_, Infallible>(chunk1), Ok(chunk2)]);
                Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }
            "invalid" => {
                let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"a\xFFb\"}}]}\n\n";
                Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(body.to_vec()))
                    .unwrap()
            }
            "repetition" => {
                let content = "a".repeat(100);
                let body = format!(
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}}}}]}}\n\n",
                    content
                );
                Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(body))
                    .unwrap()
            }
            _ => (
                StatusCode::NOT_FOUND,
                json!({"error":{"message":"unknown vector"}}).to_string(),
            )
                .into_response(),
        }
    }

    /// Send a streaming chat request to the proxy and return (status, body).
    async fn proxy_stream_request(proxy_addr: &SocketAddr, vector: &str) -> (u16, String) {
        let url = format!("http://{proxy_addr}/v1/chat/completions?vector={vector}");
        let client = reqwest::Client::new();
        let res = client
            .post(&url)
            .header("accept", "text/event-stream")
            .body(
                json!({"model":"test","stream":true,"messages":[{"role":"user","content":"hi"}]})
                    .to_string(),
            )
            .send()
            .await
            .unwrap();
        let status = res.status().as_u16();
        let body = res.text().await.unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn proxy_4xx_passthrough_verbatim() {
        let upstream = start_mock_upstream().await;
        let state = tmp_state();
        let proxy = ProxyServer::new(&state, upstream);
        let proxy_addr = proxy.serve_ephemeral().await.unwrap();

        let (status, body) = proxy_stream_request(&proxy_addr, "4xx").await;
        assert_eq!(status, 400, "4xx must be passed through verbatim");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["message"], "bad request");
        assert_eq!(v["error"]["type"], "invalid_request");
    }

    #[tokio::test]
    async fn proxy_mid_stream_error_no_done() {
        let upstream = start_mock_upstream().await;
        let state = tmp_state();
        let proxy = ProxyServer::new(&state, upstream);
        let proxy_addr = proxy.serve_ephemeral().await.unwrap();

        let (status, body) = proxy_stream_request(&proxy_addr, "error").await;
        assert_eq!(status, 200);
        assert!(body.contains("event: error"), "got: {body}");
        assert!(body.contains("boom"), "got: {body}");
        assert!(
            !body.contains("[DONE]"),
            "error frame must NOT append [DONE]: {body}"
        );
    }

    #[tokio::test]
    async fn proxy_emoji_intact() {
        let upstream = start_mock_upstream().await;
        let state = tmp_state();
        let proxy = ProxyServer::new(&state, upstream);
        let proxy_addr = proxy.serve_ephemeral().await.unwrap();

        let (status, body) = proxy_stream_request(&proxy_addr, "emoji").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\u{1F600}"),
            "emoji must arrive intact, got: {body}"
        );
        assert!(!body.contains('\u{FFFD}'), "no replacement chars: {body}");
    }

    #[tokio::test]
    async fn proxy_invalid_byte_fffd() {
        let upstream = start_mock_upstream().await;
        let state = tmp_state();
        let proxy = ProxyServer::new(&state, upstream);
        let proxy_addr = proxy.serve_ephemeral().await.unwrap();

        let (status, body) = proxy_stream_request(&proxy_addr, "invalid").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\u{FFFD}"),
            "invalid byte must become U+FFFD, got: {body}"
        );
    }

    #[tokio::test]
    async fn proxy_repetition_breaker() {
        let upstream = start_mock_upstream().await;
        let state = tmp_state();
        let proxy = ProxyServer::new(&state, upstream);
        let proxy_addr = proxy.serve_ephemeral().await.unwrap();

        let (status, body) = proxy_stream_request(&proxy_addr, "repetition").await;
        assert_eq!(status, 200);
        assert!(
            body.contains(crate::proxy::sanitize::GUARD_MARKER_PREFIX),
            "got: {body}"
        );
        assert!(body.contains("finish_reason"), "got: {body}");
        assert!(
            body.contains("[DONE]"),
            "breaker must append [DONE]: {body}"
        );
    }

    #[tokio::test]
    async fn proxy_lock_held_second_instance() {
        let state = tmp_state();
        let upstream = SocketAddr::from(([127, 0, 0, 1], 18020));
        let proxy = ProxyServer::new(&state, upstream);
        // First acquire succeeds.
        assert!(proxy.try_acquire_lock());
        // Second server (same state dir) should find the lock held.
        let proxy2 = ProxyServer::new(&state, upstream);
        assert!(
            !proxy2.try_acquire_lock(),
            "second server must see lock held"
        );
        // Release; now the second can acquire.
        proxy.release_lock();
        assert!(proxy2.try_acquire_lock());
        proxy2.release_lock();
    }

    #[tokio::test]
    async fn proxy_lock_reclaims_dead_pid_within_ttl() {
        let state = tmp_state();
        let upstream = SocketAddr::from(([127, 0, 0, 1], 18020));
        let proxy = ProxyServer::new(&state, upstream);

        #[cfg(unix)]
        let mut child = std::process::Command::new("true").spawn().unwrap();
        #[cfg(windows)]
        let mut child = std::process::Command::new("cmd")
            .args(["/c", "exit", "0"])
            .spawn()
            .unwrap();
        let dead_pid = child.id();
        let _ = child.wait();

        let now = crate::platform::now_epoch_ms();
        std::fs::create_dir_all(state.locks()).unwrap();
        std::fs::write(
            state.locks().join("proxy.lock"),
            serde_json::json!({ "pid": dead_pid, "held_at": now, "ttl": LOCK_TTL_MS }).to_string(),
        )
        .unwrap();

        assert!(
            proxy.try_acquire_lock(),
            "in-TTL lock owned by dead pid must be reclaimed"
        );
        proxy.release_lock();
    }

    #[tokio::test]
    async fn proxy_from_upstream_url_with_prefix_and_concurrent_burst() {
        let app = Router::new()
            .route(
                "/api/v1/models",
                routing::get(|| async { json!({"object":"list","data":[]}).to_string() }),
            )
            .route(
                "/api/v1/chat/completions",
                routing::post(|| async {
                    Response::builder()
                        .status(200)
                        .header("content-type", "text/event-stream")
                        .body(Body::from("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"))
                        .unwrap()
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let state = tmp_state();
        let proxy = ProxyServer::from_upstream_url(&state, format!("http://{addr}/api/"));
        let proxy_addr = proxy.serve_ephemeral().await.unwrap();

        let client = reqwest::Client::new();
        let health = client
            .get(format!("http://{proxy_addr}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(health.status(), 200);
        let health_json: serde_json::Value = health.json().await.unwrap();
        assert_eq!(health_json["upstream"], format!("http://{addr}/api"));

        let mut joins = Vec::new();
        for i in 0..16 {
            let c = client.clone();
            joins.push(tokio::spawn(async move {
                if i % 2 == 0 {
                    let r = c
                        .get(format!("http://{proxy_addr}/v1/models"))
                        .send()
                        .await
                        .unwrap();
                    assert_eq!(r.status(), 200);
                } else {
                    let r = c
                        .post(format!("http://{proxy_addr}/v1/chat/completions"))
                        .json(&json!({"model":"m","stream":true,"messages":[]}))
                        .send()
                        .await
                        .unwrap();
                    assert_eq!(r.status(), 200);
                    let body = r.text().await.unwrap();
                    assert!(body.contains("[DONE]"), "missing DONE: {body}");
                }
            }));
        }
        for j in joins {
            j.await.unwrap();
        }
    }

    #[tokio::test]
    async fn proxy_rewrites_content_length_when_sanitizing_body() {
        let app = Router::new().route(
            "/v1/chat/completions",
            routing::post(|body: Bytes| async move {
                let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
                Response::builder()
                    .status(200)
                    .header("content-type", "application/json")
                    .body(Body::from(v.to_string()))
                    .unwrap()
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let state = tmp_state();
        let proxy = ProxyServer::new(&state, addr);
        let proxy_addr = proxy.serve_ephemeral().await.unwrap();

        let big_img = format!("data:image/png;base64,{}", "A".repeat(4096));
        let payload = json!({
            "model": "m",
            "stream": false,
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": big_img}}]
            }]
        })
        .to_string();
        let res = reqwest::Client::new()
            .post(format!("http://{proxy_addr}/v1/chat/completions"))
            .header("content-type", "application/json")
            .header("content-length", payload.len().to_string())
            .body(payload)
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let parsed: serde_json::Value = res.json().await.unwrap();
        let content_str = parsed.to_string();
        assert!(
            content_str.contains("Image file omitted") || content_str.contains("image_url"),
            "unexpected upstream echo: {content_str}"
        );
    }

    #[tokio::test]
    async fn probe_port_requires_http_2xx_health() {
        assert!(!probe_port(0).await);

        let raw_tcp = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let raw_port = raw_tcp.local_addr().unwrap().port();
        let raw_accept = tokio::spawn(async move {
            let _ = raw_tcp.accept().await;
        });
        assert!(
            !probe_port(raw_port).await,
            "raw TCP listener without HTTP /health must not pass proxy probe"
        );
        raw_accept.abort();

        let state = tmp_state();
        let upstream = SocketAddr::from(([127, 0, 0, 1], 18020));
        let proxy_addr = ProxyServer::new(&state, upstream)
            .serve_ephemeral()
            .await
            .unwrap();
        assert!(
            probe_port(proxy_addr.port()).await,
            "live ProxyServer must pass HTTP /health probe"
        );
    }

    #[tokio::test]
    async fn ensure_proxy_cleans_stale_proxy_port_and_finds_default_port() {
        let state = tmp_state();
        let upstream = SocketAddr::from(([127, 0, 0, 1], 18020));
        let live_addr = ProxyServer::new(&state, upstream)
            .serve_ephemeral()
            .await
            .unwrap();
        let live_port = live_addr.port();

        let unused = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let stale_port = unused.local_addr().unwrap().port();
        drop(unused);
        let port_file = state.ports().join("proxy.port");
        std::fs::write(&port_file, stale_port.to_string()).unwrap();

        ensure_proxy_server_for_upstream(&state, live_port, "http://127.0.0.1:18020").await;
        assert!(
            !port_file.exists(),
            "stale proxy.port must be removed when configured proxy port is live"
        );
    }
}
