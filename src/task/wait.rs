//! Status / long-poll HTTP server.
//!
//! Architecture and invariants:
//!
//! - **Singleton**: acquires `Locks::acquire("status")` at startup. If the
//!   lock is held by another live process, the server exits 0 with a single
//!   stderr line `"status server already running"`.
//! - **GET /task/:id/wait?timeout_s=N** — long-poll. Wakes on a terminal
//!   transition (per-task broadcast channel) or a disk-mirror poll (500 ms)
//!   for tasks owned by other instances. Terminal ⇒ ALWAYS HTTP 200 JSON
//!   (success OR failure — never masked). Timeout ⇒ 200 `{status:"executing",
//!   timed_out:true}`. Unknown id ⇒ 404.
//! - **GET /task/:id/status** — computed `elapsed_s` (queued AND executing).
//!   Malformed disk file ⇒ 400 `{error}` (never 500).
//!
//! The `mcp` subcommand spawns this as a detached child (lock → probe →
//! spawn if down → release), best-effort, never blocking the MCP server.

use std::net::SocketAddr;
#[cfg(unix)]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde::Deserialize;
use serde_json::json;

use crate::platform::now_epoch_ms as now_ms;
use crate::state::{AcquireResult, Locks, StateDir};
use crate::task::registry::{DiskRead, TaskRecord, TaskRegistry, TaskStatus};

/// Disk-mirror poll interval for cross-instance waiters (500 ms).
const DISK_POLL_MS: u64 = 500;

/// The status server, cheaply cloneable (all state behind `Arc`).
#[derive(Clone)]
pub struct StatusServer {
    inner: Arc<Inner>,
}

struct Inner {
    registry: TaskRegistry,
    locks: Locks,
}

impl StatusServer {
    pub fn new(state: &StateDir) -> Self {
        let registry = TaskRegistry::new(state);
        let locks = Locks::from_state(state);
        Self {
            inner: Arc::new(Inner { registry, locks }),
        }
    }

    /// Acquire the singleton lock. Returns `true` if we own it, `false` if
    /// another live process holds it (caller should exit 0).
    pub fn try_acquire_lock(&self) -> bool {
        let ttl = 30_000; // 30 s TTL; liveness-checked against active holder pid.
        match self.inner.locks.acquire_live("status", ttl) {
            Ok(AcquireResult::Acquired) => true,
            Ok(AcquireResult::HeldBy { .. }) => false,
            Err(e) => {
                eprintln!("castor: status lock acquire error: {e}");
                false
            }
        }
    }

    /// Acquire the singleton lock, retrying up to `retries` times with `delay_ms`
    /// backoff to avoid TOCTOU startup races when spawned by `ensure_status_server`.
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
        let _ = self.inner.locks.release("status");
    }

    /// Build the axum router.
    fn router(&self) -> Router {
        Router::new()
            .route("/task/{id}/wait", get(wait_handler))
            .route("/task/{id}/status", get(status_handler))
            .with_state(self.clone())
            .merge(super::trace::trace_router(
                self.inner.registry.state_dir().clone(),
            ))
    }

    /// Listen on the given port. Blocks until the server is dropped or the
    /// process is signalled.
    pub async fn serve(self, port: u16) -> std::io::Result<()> {
        // Boot reaper: reconcile orphans left behind by a reboot/crash before
        // serving, guarded by the `task_reap` lock so only one instance does it.
        boot_reap(self.inner.registry.state_dir()).await;

        // Periodic slot/task reaper: while the status server runs, every 30 s
        // reclaim orphaned slot leases (dead owner or bound task terminal) and
        // reconcile orphaned tasks. This is what keeps slots from staying
        // locked indefinitely after a worker that died without releasing them.
        {
            let registry = self.inner.registry.clone();
            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(Duration::from_secs(30));
                ticker.tick().await; // skip the first (immediate) tick
                loop {
                    ticker.tick().await;
                    let _ = registry.reap_orphans().await;
                    let reclaimed = registry.reap_slots().await;
                    if reclaimed > 0 {
                        eprintln!("[castor-status] reaper reclaimed {reclaimed} orphaned slot(s)");
                    }
                }
            });
        }

        let candidate_ports = [port, 18023, 18024, 18025, 0];
        let mut bound_listener = None;
        for &p in &candidate_ports {
            let addr = SocketAddr::from(([127, 0, 0, 1], p));
            match tokio::net::TcpListener::bind(addr).await {
                Ok(l) => {
                    bound_listener = Some(l);
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                    eprintln!("[castor-status] port {p} in use, trying next candidate");
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        let listener = bound_listener.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::AddrInUse,
                "all candidate status ports in use",
            )
        })?;
        let bound_addr = listener.local_addr()?;
        let bound_port = bound_addr.port();
        eprintln!("[castor-status] listening on {bound_addr}");

        // Record active port into <state>/ports/status.port
        let ports_dir = self.inner.registry.state_dir().ports();
        let _ = std::fs::create_dir_all(&ports_dir);
        let port_file = ports_dir.join("status.port");
        let _ = std::fs::write(&port_file, bound_port.to_string());

        axum::serve(listener, self.router()).await
    }

    /// Listen on an ephemeral port (for tests). Returns the bound address.
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
// Query params
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WaitQuery {
    #[serde(default = "default_timeout")]
    timeout_s: u64,
}

fn default_timeout() -> u64 {
    3600
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /task/:id/wait?timeout_s=N
///
/// Long-poll: waits until the task reaches a terminal state or the timeout
/// elapses. Terminal ⇒ 200 JSON (always, even on failure). Timeout ⇒ 200
/// `{status:"executing", timed_out:true}`. Unknown id ⇒ 404.
async fn wait_handler(
    State(server): State<StatusServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Query(q): Query<WaitQuery>,
) -> Response {
    let reg = &server.inner.registry;

    // Orphan reaper before reads: a dead-pid executing task is marked failed
    // (idempotent) so the wait resolves with an honest terminal state.
    let _ = reg.reap_orphans().await;

    // Fast path: already terminal (memory or disk).
    if let Some(rec) = reg.get(&id).await {
        if rec.status.is_terminal() {
            return terminal_response(&rec).into_response();
        }
    } else {
        // Not in memory: check disk.
        match reg.read_disk(&id) {
            DiskRead::Ok(rec) if rec.status.is_terminal() => {
                return terminal_response(&rec).into_response();
            }
            DiskRead::Ok(_) => {}
            DiskRead::NotFound => {
                return (
                    StatusCode::NOT_FOUND,
                    json!({"error": "task not found"}).to_string(),
                )
                    .into_response();
            }
            DiskRead::Corrupt(msg) => {
                return (
                    StatusCode::BAD_REQUEST,
                    json!({"error": format!("malformed task file: {msg}")}).to_string(),
                )
                    .into_response();
            }
        }
    }

    // Not terminal yet: subscribe and wait.
    let rx = reg.subscribe_terminal(&id).await;
    let timeout = Duration::from_secs(q.timeout_s);

    tokio::select! {
        _ = wait_for_terminal(rx, reg, &id, timeout) => {}
    };

    // Re-read after the wait: the task may have gone terminal.
    if let Some(rec) = reg.get(&id).await
        && rec.status.is_terminal()
    {
        return terminal_response(&rec).into_response();
    }
    // Also check disk (cross-instance).
    if let DiskRead::Ok(rec) = reg.read_disk(&id)
        && rec.status.is_terminal()
    {
        return terminal_response(&rec).into_response();
    }

    // Timed out.
    (
        StatusCode::OK,
        json!({"status": "executing", "timed_out": true}).to_string(),
    )
        .into_response()
}

/// Wait for a terminal transition on a task, polling the disk mirror for
/// cross-instance tasks. Returns when the task is terminal or the timeout
/// elapses.
async fn wait_for_terminal(
    mut rx: tokio::sync::broadcast::Receiver<u64>,
    reg: &TaskRegistry,
    id: &str,
    timeout: Duration,
) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return;
        }
        // Wait for either a broadcast wake or the poll interval.
        let poll = Duration::from_millis(DISK_POLL_MS);
        let wait_dur = remaining.min(poll);
        tokio::select! {
            _ = rx.recv() => {
                // Woken by a terminal transition; check state.
                if let Some(rec) = reg.get(id).await
                    && rec.status.is_terminal() {
                        return;
                    }
            }
            _ = tokio::time::sleep(wait_dur) => {
                // Poll the disk mirror (cross-instance fallback).
                if let DiskRead::Ok(rec) = reg.read_disk(id)
                    && rec.status.is_terminal() {
                        return;
                    }
            }
        }
    }
}

/// GET /task/:id/status
///
/// Returns the task's current status with a computed `elapsed_s`.
/// Malformed disk file ⇒ 400 `{error}` (never 500). Unknown id ⇒ 404.
async fn status_handler(
    State(server): State<StatusServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let reg = &server.inner.registry;

    // Orphan reaper before reads.
    let _ = reg.reap_orphans().await;

    // Memory first.
    if let Some(rec) = reg.get(&id).await {
        return status_response(&rec).into_response();
    }

    // Disk fallback.
    match reg.read_disk(&id) {
        DiskRead::Ok(rec) => status_response(&rec).into_response(),
        DiskRead::NotFound => (
            StatusCode::NOT_FOUND,
            json!({"error": "task not found"}).to_string(),
        )
            .into_response(),
        DiskRead::Corrupt(msg) => (
            StatusCode::BAD_REQUEST,
            json!({"error": format!("malformed task file: {msg}")}).to_string(),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Response builders
// ---------------------------------------------------------------------------

/// Build the terminal-response JSON (always 200, never masked).
fn terminal_response(rec: &TaskRecord) -> (StatusCode, String) {
    let now = now_ms();
    let elapsed_s = elapsed_seconds(rec, now);
    let body = json!({
        "found": true,
        "id": rec.id,
        "status": rec.status,
        "reason": rec.reason,
        "session_id": rec.session_id,
        "cwd": rec.cwd,
        "elapsed_s": elapsed_s,
        "created_at": rec.created_at,
        "started_at": rec.started_at,
        "ended_at": rec.ended_at,
        "turns_budget": rec.turns_budget,
    });
    (StatusCode::OK, body.to_string())
}

/// Build the status-response JSON with computed `elapsed_s`.
fn status_response(rec: &TaskRecord) -> (StatusCode, String) {
    let now = now_ms();
    let elapsed_s = elapsed_seconds(rec, now);
    let body = json!({
        "found": true,
        "id": rec.id,
        "status": rec.status,
        "reason": rec.reason,
        "session_id": rec.session_id,
        "cwd": rec.cwd,
        "elapsed_s": elapsed_s,
        "created_at": rec.created_at,
        "started_at": rec.started_at,
        "ended_at": rec.ended_at,
        "turns_budget": rec.turns_budget,
    });
    (StatusCode::OK, body.to_string())
}

/// Compute elapsed seconds from the task's creation (or start) to now.
///
/// - `queued`: elapsed since `created_at`.
/// - `executing`: elapsed since `started_at` (falling back to `created_at`).
/// - terminal: elapsed since `created_at` to `ended_at`.
fn elapsed_seconds(rec: &TaskRecord, now: u64) -> u64 {
    let end = rec.ended_at.unwrap_or(now);
    let start = match rec.status {
        TaskStatus::Executing => rec.started_at.unwrap_or(rec.created_at),
        _ => rec.created_at,
    };
    end.saturating_sub(start) / 1000
}

// ---------------------------------------------------------------------------
// Spawn helper (used by the `mcp` subcommand)
// ---------------------------------------------------------------------------

/// Read the active status port recorded by a running status server, falling back
/// to `default_port` if absent or invalid.
pub fn active_status_port(state: &StateDir, default_port: u16) -> u16 {
    let port_file = state.ports().join("status.port");
    std::fs::read_to_string(&port_file)
        .ok()
        .and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(default_port)
}

/// Best-effort, non-blocking bring-up of the status server, invoked by the
/// `mcp` subcommand before it starts serving.
///
/// The sequence is **lock → probe → release & spawn if down**:
///
/// 1. **lock** — try to acquire the `status` singleton lock. If another live
///    process already holds it, a status server is running; we do not spawn.
/// 2. **probe** — check whether something is already listening on the active status
///    port (a live keeper may hold a stale lock, or a foreign process may own
///    the port). If the port is already serving, we do not spawn.
/// 3. **release & spawn** — release the lock first so the child process can acquire
///    it without a TOCTOU race, then spawn a detached `castor status` child.
///
/// Before any of that, the lock-guarded **boot reaper** runs once (reconcile
/// orphans + reclaim stale slots) so a reboot immediately heals stuck tasks.
///
/// Every step is best-effort: a failure is logged to stderr and swallowed.
/// This never blocks the MCP server from starting.
pub async fn ensure_status_server(state: &StateDir, port: u16) {
    let server = StatusServer::new(state);

    // 1. lock.
    let acquired = server.try_acquire_lock();

    // 2. probe: is a status server already listening on the port?
    let target_port = active_status_port(state, port);
    let already_up = probe_status_port(target_port).await;

    if acquired && !already_up {
        // Release our lock before spawning so the child process can acquire
        // it cleanly without a TOCTOU race.
        server.release_lock();
        // 3. spawn (detached child re-acquires the lock).
        spawn_detached(state, port);
    } else {
        // We either don't own the lock (another live process does) or the
        // port is already serving. Release our lock if we hold it; do not
        // spawn.
        if acquired {
            server.release_lock();
        }
        eprintln!("[castor] status server already up (port {target_port}); not spawning");
    }
}

/// Lock-guarded boot-time orphan reconciliation.
///
/// Acquires the `task_reap` lockfile (60 s TTL) so that at most one process
/// (across all `mcp` / `status` instances sharing a state dir) performs the
/// reconciliation on a given boot. When held, it:
///
/// 1. runs [`TaskRegistry::reap_orphans_disk`] — reaps `executing` tasks whose
///    worker died (reboot / crash) and stale `queued` tasks whose worker never
///    started, persisting honest terminal states to the disk mirror;
/// 2. reclaims orphaned slot leases via [`TaskRegistry::reap_slots`] — deletes
///    any `slot_N.json` whose owner pid is dead (including zombies) or whose
///    bound task is terminal, so a worker that died before releasing its slot
///    cannot keep the slot locked (and deadlocking other tenants) indefinitely.
///
/// All steps are best-effort and never block the caller: a failure is logged
/// to stderr and swallowed.
///
/// Invoked once per process at startup, from the status server's
/// [`StatusServer::serve`] and from the `mcp` subcommand's startup path
/// (`crate::cli`). The `task_reap` lock guarantees at most one instance
/// performs the work per boot, so calling it from more than one site in a
/// single process is harmless (the second call observes the lock as held by
/// itself, or the work is idempotent).
pub async fn boot_reap(state: &StateDir) {
    let locks = Locks::from_state(state);
    match locks.acquire("task_reap", 60_000) {
        Ok(AcquireResult::Acquired) => {
            let registry = TaskRegistry::new(state);
            let reaped = registry.reap_orphans_disk().await;
            if reaped > 0 {
                eprintln!("[castor] boot reaper: reconciled {reaped} orphaned task(s)");
            }
            let slots = registry.reap_slots().await;
            if slots > 0 {
                eprintln!("[castor] boot reaper: reclaimed {slots} orphaned slot(s)");
            }
            let _ = locks.release("task_reap");
        }
        Ok(AcquireResult::HeldBy { .. }) => {
            // Another live instance is reconciling right now; skip.
        }
        Err(e) => {
            eprintln!("[castor] boot reaper lock error: {e}");
        }
    }
}

/// Probe the status port with a short-timeout TCP connect. Returns `true` if
/// something is listening (a live keeper), `false` if the port is dark.
async fn probe_status_port(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let res = tokio::time::timeout(
        Duration::from_millis(500),
        tokio::net::TcpStream::connect(addr),
    )
    .await;
    matches!(res, Ok(Ok(_)))
}

/// Spawn the status server as a fully detached child process (new process
/// group on Unix, stdio nulled). Never waits for the child.
fn spawn_detached(state: &StateDir, port: u16) {
    let Some(exe) = std::env::current_exe().ok() else {
        eprintln!("[castor] cannot determine current exe; skipping status server spawn");
        return;
    };
    let state_str = state.root().to_string_lossy().into_owned();
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("status")
        .arg("--state-dir")
        .arg(&state_str)
        .arg("--port")
        .arg(port.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    // Detach: put the child in its own process group (pgid 0 = the child's
    // own pid) so it survives the parent and is not reaped with it.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW (0x08000000) | CREATE_NEW_PROCESS_GROUP (0x00000200) | DETACHED_PROCESS (0x00000008)
        cmd.creation_flags(0x08000208);
    }
    match cmd.spawn() {
        Ok(_) => {
            eprintln!("[castor] status server spawned (port {port})");
        }
        Err(e) => {
            eprintln!("[castor] failed to spawn status server: {e}");
        }
    }
}

/// Helper to extract the Windows host gateway IP inside WSL from `/etc/resolv.conf`.
pub fn get_wsl_host_ip() -> Option<String> {
    let content = std::fs::read_to_string("/etc/resolv.conf").ok()?;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("nameserver") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                return Some(parts[1].to_string());
            }
        }
    }
    None
}

/// Locate the effective state directory for a task across environment boundaries.
///
/// Under WSL, tasks dispatched by a Windows-host Castor instance reside in
/// `/mnt/c/Users/<user>/.castor`. If the local WSL state dir (`~/.castor`)
/// does not contain the task record or active status port, this probes the
/// Windows user profile state dir so `castor wait` works seamlessly cross-boundary.
///
/// The Windows probe is two-pass so that a foreign profile's `status.port`
/// file can never hijack our state dir:
///
/// - **Pass 1** looks strictly for the task's disk mirror — first under our
///   own `$USER` profile, then across every `/mnt/c/Users/*` profile.
/// - **Pass 2** falls back to a `ports/status.port` check ONLY under our own
///   `$USER` profile (the live Windows server of our account, even when this
///   specific task's file has not been mirrored yet).
pub fn resolve_cross_env_state_dir(state: &StateDir, task_id: &str) -> StateDir {
    // 1. If the task file exists in the current state dir, use it.
    // The on-disk mirror is `{task_id}.json` (see `TaskRegistry::task_path`),
    // where the id itself already carries the `task_` prefix.
    let local_task_file = state.tasks().join(format!("{task_id}.json"));
    if local_task_file.exists() {
        return state.clone();
    }

    // 2. If running inside WSL, probe Windows user profile directories.
    #[cfg(unix)]
    if crate::platform::is_wsl() {
        let user = std::env::var("USER").ok();
        // Pass 1: a task file is definitive proof of ownership. Check our own
        // profile first, then scan every Windows user profile.
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(user) = &user {
            candidates.push(PathBuf::from(format!("/mnt/c/Users/{user}/.castor")));
        }
        if let Ok(entries) = std::fs::read_dir("/mnt/c/Users") {
            for entry in entries.flatten() {
                let candidate = entry.path().join(".castor");
                if !candidates.iter().any(|c| c == &candidate) {
                    candidates.push(candidate);
                }
            }
        }
        // `{task_id}.json` — same convention as `TaskRegistry::task_path`.
        let task_file_name = format!("{task_id}.json");
        if let Some(candidate) = candidates
            .into_iter()
            .find(|c| c.join("tasks").join(&task_file_name).exists())
        {
            return StateDir::new(candidate);
        }
        // Pass 2: only our own profile's active status port qualifies —
        // never another user's `.castor` based on a `status.port` alone.
        if let Some(user) = user
            && PathBuf::from(format!("/mnt/c/Users/{user}/.castor"))
                .join("ports")
                .join("status.port")
                .exists()
        {
            return StateDir::new(PathBuf::from(format!("/mnt/c/Users/{user}/.castor")));
        }
    }

    state.clone()
}

/// Fetch candidate wait URLs in order until one succeeds.
///
/// Returns `Some(body)` for the first candidate that answers 2xx. A 404 (or
/// any other non-2xx) response, a transport error (connection refused,
/// timeout, DNS), or a failed body read on a candidate does **not** end the
/// loop — the *next* candidate is tried. This lets a status server that
/// owns the state dir but not the task (e.g. loopback `:18021` in WSL while
/// the Windows host owns the task) or an unreachable candidate be skipped
/// in favor of a healthy instance on another candidate (the host gateway
/// IP). If every candidate fails, returns `None` and the caller falls back
/// to disk polling.
async fn try_wait_candidate_urls(client: &reqwest::Client, urls: &[String]) -> Option<String> {
    for url in urls {
        // A transport error (refused/timeout/DNS) must continue to the next
        // candidate, NOT abort the loop — hence an explicit match, not `?`.
        let resp = match client.get(url).send().await {
            Ok(resp) => resp,
            Err(_) => continue,
        };
        if !resp.status().is_success() {
            // Non-2xx (notably 404): the probed instance does not serve this
            // task — continue to the next candidate instead of bailing to
            // disk polling.
            continue;
        }
        match resp.text().await {
            Ok(text) => return Some(text),
            Err(_) => continue,
        }
    }
    None
}

/// CLI handler for `castor wait <task_id> [--timeout-s N] [--port P] [--state-dir D]`.
///
/// Seamlessly waits for a task to reach a terminal state whether invoked from native Windows or inside WSL:
/// 1. Inspects the state dir (with cross-environment fallback between WSL and Windows if needed).
/// 2. If the task is already terminal on disk, immediately outputs the JSON response.
/// 3. Attempts HTTP long-poll to the StatusServer (trying 127.0.0.1, then host IP if in WSL).
/// 4. If HTTP is unreachable or refused (due to NAT, firewall, or down server), seamlessly
///    falls back to direct disk-mirror polling until terminal or timeout.
/// 5. Returns the exact JSON response.
pub async fn run_wait_cli(
    state: &StateDir,
    task_id: &str,
    timeout_s: u64,
    override_port: Option<u16>,
) -> Result<String, String> {
    let effective_state = resolve_cross_env_state_dir(state, task_id);
    let reg = TaskRegistry::new(&effective_state);

    // Fast-path: check if task is already terminal on disk.
    match reg.read_disk(task_id) {
        DiskRead::Ok(rec) if rec.status.is_terminal() => {
            let (_, body) = terminal_response(&rec);
            return Ok(body);
        }
        DiskRead::Corrupt(msg) => {
            return Err(format!("malformed task file: {msg}"));
        }
        _ => {}
    }

    let port = override_port.unwrap_or_else(|| active_status_port(&effective_state, 18021));

    // Candidate wait URLs: loopback first, then (under WSL) the host
    // gateway IP from `/etc/resolv.conf`.
    #[cfg(unix)]
    let candidate_urls = {
        let mut urls = vec![format!(
            "http://127.0.0.1:{port}/task/{task_id}/wait?timeout_s={timeout_s}"
        )];
        if crate::platform::is_wsl()
            && let Some(host_ip) = get_wsl_host_ip()
        {
            urls.push(format!(
                "http://{host_ip}:{port}/task/{task_id}/wait?timeout_s={timeout_s}"
            ));
        }
        urls
    };

    #[cfg(not(unix))]
    let candidate_urls = vec![format!(
        "http://127.0.0.1:{port}/task/{task_id}/wait?timeout_s={timeout_s}"
    )];

    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_s.saturating_add(5)))
        .build()
        .map_err(|e| e.to_string())?;

    // Try HTTP long-poll to StatusServer. A 404 or transport error on one
    // candidate does not end the loop — the next candidate is tried before
    // falling back to disk polling (a server that owns the state dir but not
    // this task must not mask a healthy instance on another candidate).
    if let Some(body) = try_wait_candidate_urls(&http_client, &candidate_urls).await {
        return Ok(body);
    }

    // Disk fallback: poll the disk mirror directly until terminal or timeout.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_s);
    let poll_interval = Duration::from_millis(DISK_POLL_MS);

    loop {
        match reg.read_disk(task_id) {
            DiskRead::Ok(rec) if rec.status.is_terminal() => {
                let (_, body) = terminal_response(&rec);
                return Ok(body);
            }
            DiskRead::Corrupt(msg) => {
                return Err(format!("malformed task file: {msg}"));
            }
            _ => {}
        }

        if tokio::time::Instant::now() >= deadline {
            let body = json!({
                "status": "executing",
                "timed_out": true
            });
            return Ok(body.to_string());
        }

        tokio::time::sleep(poll_interval).await;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as Ord};

    static TMP: AtomicUsize = AtomicUsize::new(0);

    fn tmp_state() -> StateDir {
        let n = TMP.fetch_add(1, Ord::SeqCst);
        let p = std::env::temp_dir().join(format!("castor-wait-{}-{}", std::process::id(), n));
        let s = StateDir::new(p);
        s.ensure().unwrap();
        s
    }

    /// A dead pid (spawn a child, wait for it to exit).
    fn dead_pid() -> u32 {
        #[cfg(windows)]
        let mut c = std::process::Command::new("cmd")
            .args(["/c", "exit", "0"])
            .spawn()
            .unwrap();
        #[cfg(not(windows))]
        let mut c = std::process::Command::new("true").spawn().unwrap();
        let pid = c.id();
        let _ = c.wait();
        pid
    }

    /// Make an HTTP GET and return (status_code, body).
    async fn http_get(addr: &SocketAddr, path: &str) -> (u16, String) {
        let url = format!("http://{addr}{path}");
        let client = reqwest::Client::new();
        let res = client.get(&url).send().await.unwrap();
        let status = res.status().as_u16();
        let body = res.text().await.unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn wait_wakes_on_completed() {
        let state = tmp_state();
        let server = StatusServer::new(&state);
        let reg = server.inner.registry.clone();
        let addr = server.serve_ephemeral().await.unwrap();

        let id = reg.create("do it", "/tmp", "s1").await.unwrap();
        let _ = reg
            .transition(&id, TaskStatus::Executing, None)
            .await
            .unwrap();

        // Spawn a wait with a 5s timeout; transition to completed after 100ms.
        let reg2 = reg.clone();
        let id2 = id.clone();
        let handle = tokio::spawn(async move {
            let url = format!("http://{addr}/task/{id2}/wait?timeout_s=5");
            let client = reqwest::Client::new();
            let res = client.get(&url).send().await.unwrap();
            (res.status().as_u16(), res.text().await.unwrap())
        });

        // Give the wait a moment to subscribe, then transition.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = reg2
            .transition(&id, TaskStatus::Completed, None)
            .await
            .unwrap();

        let (status, body) = handle.await.unwrap();
        assert_eq!(status, 200, "wait should return 200 on terminal");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "completed");
        assert_eq!(v["found"], true);
        assert!(
            !v.get("timed_out")
                .and_then(|t| t.as_bool())
                .unwrap_or(false)
        );
    }

    #[tokio::test]
    async fn wait_wakes_on_failed_with_honest_status() {
        let state = tmp_state();
        let server = StatusServer::new(&state);
        let reg = server.inner.registry.clone();
        let addr = server.serve_ephemeral().await.unwrap();

        let id = reg.create("do it", "/tmp", "s2").await.unwrap();
        let _ = reg
            .transition(&id, TaskStatus::Executing, None)
            .await
            .unwrap();

        let reg2 = reg.clone();
        let id2 = id.clone();
        let handle = tokio::spawn(async move {
            let url = format!("http://{addr}/task/{id2}/wait?timeout_s=5");
            let client = reqwest::Client::new();
            let res = client.get(&url).send().await.unwrap();
            (res.status().as_u16(), res.text().await.unwrap())
        });

        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = reg2
            .transition(&id, TaskStatus::Failed, Some("engine error".into()))
            .await
            .unwrap();

        let (status, body) = handle.await.unwrap();
        assert_eq!(status, 200, "failure must be 200, not 500");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "failed");
        assert_eq!(v["reason"], "engine error");
    }

    #[tokio::test]
    async fn wait_timeout_path() {
        let state = tmp_state();
        let server = StatusServer::new(&state);
        let reg = server.inner.registry.clone();
        let addr = server.serve_ephemeral().await.unwrap();

        let id = reg.create("do it", "/tmp", "s3").await.unwrap();
        // Leave it executing; no transition.

        let (status, body) = http_get(&addr, &format!("/task/{id}/wait?timeout_s=1")).await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "executing");
        assert_eq!(v["timed_out"], true);
    }

    #[tokio::test]
    async fn wait_404_unknown_id() {
        let state = tmp_state();
        let server = StatusServer::new(&state);
        let addr = server.serve_ephemeral().await.unwrap();

        let (status, body) = http_get(&addr, "/task/nonexistent_xyz/wait").await;
        assert_eq!(status, 404);
        assert!(
            body.to_lowercase().contains("not found"),
            "body should mention not found: {body}"
        );
    }

    #[tokio::test]
    async fn status_malformed_file_returns_400() {
        let state = tmp_state();
        let server = StatusServer::new(&state);
        let addr = server.serve_ephemeral().await.unwrap();

        // Write a corrupt task file.
        let path = state.tasks().join("task_corrupt.json");
        std::fs::write(&path, "{ not valid json !").unwrap();

        let (status, body) = http_get(&addr, "/task/task_corrupt/status").await;
        assert_eq!(status, 400, "malformed file must be 400, not 500");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            v["error"].as_str().unwrap().contains("malformed"),
            "error should name the problem: {body}"
        );
    }

    #[tokio::test]
    async fn status_elapsed_queued_and_executing() {
        let state = tmp_state();
        let server = StatusServer::new(&state);
        let reg = server.inner.registry.clone();
        let addr = server.serve_ephemeral().await.unwrap();

        // Queued task: created 60s ago.
        let id_q = reg.create("q", "/tmp", "sq").await.unwrap();
        let _ = reg
            .update(&id_q, |r| {
                r.created_at = now_ms() - 60_000;
            })
            .await
            .unwrap();
        let (status, body) = http_get(&addr, &format!("/task/{id_q}/status")).await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "queued");
        let elapsed = v["elapsed_s"].as_u64().unwrap();
        assert!(
            (55..=65).contains(&elapsed),
            "queued elapsed should be ~60s, got {elapsed}"
        );

        // Executing task: started 30s ago.
        let id_e = reg.create("e", "/tmp", "se").await.unwrap();
        let _ = reg
            .transition(&id_e, TaskStatus::Executing, None)
            .await
            .unwrap();
        let _ = reg
            .update(&id_e, |r| {
                r.created_at = now_ms() - 90_000;
                r.started_at = Some(now_ms() - 30_000);
            })
            .await
            .unwrap();
        let (status, body) = http_get(&addr, &format!("/task/{id_e}/status")).await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "executing");
        let elapsed = v["elapsed_s"].as_u64().unwrap();
        assert!(
            (25..=35).contains(&elapsed),
            "executing elapsed should be ~30s, got {elapsed}"
        );
    }

    #[tokio::test]
    async fn orphan_reaper_before_reads() {
        let state = tmp_state();
        let server = StatusServer::new(&state);
        let reg = server.inner.registry.clone();
        let addr = server.serve_ephemeral().await.unwrap();

        // Create an executing task with a dead pid.
        let id = reg.create("orphan", "/tmp", "so").await.unwrap();
        let _ = reg
            .transition(&id, TaskStatus::Executing, None)
            .await
            .unwrap();
        let dead = dead_pid();
        let _ = reg.update(&id, |r| r.pid = Some(dead)).await.unwrap();

        // Wait should resolve with failed (orphan reaper runs before the read).
        let (status, body) = http_get(&addr, &format!("/task/{id}/wait?timeout_s=3")).await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "failed");
        assert_eq!(v["reason"], "orphaned worker died");

        // Idempotent: a second wait also returns failed (not a double-transition).
        let (status2, body2) = http_get(&addr, &format!("/task/{id}/wait?timeout_s=1")).await;
        assert_eq!(status2, 200);
        let v2: serde_json::Value = serde_json::from_str(&body2).unwrap();
        assert_eq!(v2["status"], "failed");
    }

    #[tokio::test]
    async fn singleton_lock_held_exits_cleanly() {
        let state = tmp_state();
        let server = StatusServer::new(&state);
        // First acquire succeeds.
        assert!(server.try_acquire_lock());
        // Second server (same state dir) should find the lock held.
        let server2 = StatusServer::new(&state);
        assert!(
            !server2.try_acquire_lock(),
            "second server must see lock held"
        );
        // Release; now the second can acquire.
        server.release_lock();
        assert!(server2.try_acquire_lock());
        server2.release_lock();
    }

    #[tokio::test]
    async fn status_server_port_fallback_and_active_port_file() {
        let state = tmp_state();
        let server = StatusServer::new(&state);

        // Bind an ephemeral listener to simulate a port conflict.
        let blocker = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let blocked_port = blocker.local_addr().unwrap().port();

        // Spawn server on the blocked port. The server should hit AddrInUse on blocked_port
        // and fall back to one of candidate ports [port, 18023, 18024, 18025, 0],
        // then write the bound port to <state>/ports/status.port.
        let server_clone = server.clone();
        tokio::spawn(async move {
            let _ = server_clone.serve(blocked_port).await;
        });

        // Give server a moment to bind and record port.
        let mut active_port = 0;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let p = active_status_port(&state, 0);
            if p > 0 && p != blocked_port {
                active_port = p;
                break;
            }
        }

        assert_ne!(
            active_port, 0,
            "status server must bind to fallback port and write status.port"
        );
        assert_ne!(
            active_port, blocked_port,
            "status server must have fallen back away from blocked port"
        );

        // Verify the server responds on the active port.
        let reg = server.inner.registry.clone();
        let id = reg.create("status test", "/tmp", "s1").await.unwrap();
        let addr = SocketAddr::from(([127, 0, 0, 1], active_port));
        let (code, _) = http_get(&addr, &format!("/task/{id}/status")).await;
        assert_eq!(code, 200, "status server should respond on active port");
    }

    #[tokio::test]
    async fn test_run_wait_cli_instant_terminal_disk() {
        let state = tmp_state();
        let reg = TaskRegistry::new(&state);
        let id = reg.create("quick", "/tmp", "s_quick").await.unwrap();
        reg.transition(&id, TaskStatus::Completed, Some("all done".into()))
            .await
            .unwrap();

        let output = run_wait_cli(&state, &id, 5, None).await.unwrap();
        let v: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(v["status"], "completed");
        assert_eq!(v["reason"], "all done");
    }

    #[tokio::test]
    async fn test_run_wait_cli_http_long_poll() {
        let state = tmp_state();
        let server = StatusServer::new(&state);
        let reg = server.inner.registry.clone();
        let addr = server.serve_ephemeral().await.unwrap();

        let id = reg.create("http_task", "/tmp", "s_http").await.unwrap();
        reg.transition(&id, TaskStatus::Executing, None)
            .await
            .unwrap();

        let id_clone = id.clone();
        let reg_clone = reg.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            let _ = reg_clone
                .transition(
                    &id_clone,
                    TaskStatus::Completed,
                    Some("http success".into()),
                )
                .await;
        });

        let output = run_wait_cli(&state, &id, 5, Some(addr.port()))
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(v["status"], "completed");
        assert_eq!(v["reason"], "http success");
    }

    #[tokio::test]
    async fn test_run_wait_cli_disk_fallback_offline_server() {
        let state = tmp_state();
        let reg = TaskRegistry::new(&state);
        let id = reg.create("offline_task", "/tmp", "s_off").await.unwrap();
        reg.transition(&id, TaskStatus::Executing, None)
            .await
            .unwrap();

        let id_clone = id.clone();
        let reg_clone = reg.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let _ = reg_clone
                .transition(
                    &id_clone,
                    TaskStatus::Completed,
                    Some("disk rescued".into()),
                )
                .await;
        });

        // Pass a dark port where no server is listening to verify seamless disk fallback.
        let output = run_wait_cli(&state, &id, 5, Some(59999)).await.unwrap();
        let v: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(v["status"], "completed");
        assert_eq!(v["reason"], "disk rescued");
    }

    // ---------------------------------------------------------------------
    // Cross-environment (WSL) resolution fixtures
    //
    // The production lookup probes the *real* Windows mount, so these tests
    // build fixture profiles there under unique names
    // (`castortest-<pid>-<n>`) that cannot collide with real user profiles,
    // and a drop guard removes them. Tests skip when the mount is absent
    // (non-WSL hosts, where the WSL branch is compiled out).
    // ---------------------------------------------------------------------

    #[cfg(unix)]
    fn windows_users_root() -> PathBuf {
        PathBuf::from("/mnt/c/Users")
    }

    /// True only when the Windows user-profile mount is present **and**
    /// writable from this WSL instance, so the fixture tests can actually
    /// create `/mnt/c/Users/<fixture>/.castor`. The mount is absent on
    /// non-WSL hosts (WSL branch compiled out) and is read-only
    /// (`dr-xr-xr-x`) under some WSL permission configurations — in both
    /// cases the fixture tests must *skip*, not fail.
    #[cfg(unix)]
    fn wsl_probe_available() -> bool {
        let root = windows_users_root();
        if !root.is_dir() {
            return false;
        }
        // Probe writability with a throwaway dir; clean it up on success.
        let probe = fixture_name("probe");
        let dir = root.join(&probe);
        match std::fs::create_dir_all(&dir) {
            Ok(()) => {
                let _ = std::fs::remove_dir_all(&dir);
                true
            }
            Err(_) => false,
        }
    }

    /// Remove a fixture profile on drop so the real Windows mount is never
    /// left dirty (also covers panics).
    #[cfg(unix)]
    struct WslFixtureGuard(PathBuf);
    #[cfg(unix)]
    impl Drop for WslFixtureGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Build `/mnt/c/Users/<name>/.castor` with an optional mirrored task
    /// file and an optional `ports/status.port` file. The task file uses the
    /// `{task_id}.json` convention (`TaskRegistry::task_path`).
    #[cfg(unix)]
    fn make_wsl_fixture(
        name: &str,
        task_id: Option<&str>,
        status_port: Option<u16>,
    ) -> WslFixtureGuard {
        let profile = windows_users_root().join(name).join(".castor");
        if let Some(p) = status_port {
            std::fs::create_dir_all(profile.join("ports")).unwrap();
            std::fs::write(profile.join("ports").join("status.port"), p.to_string()).unwrap();
        }
        if let Some(tid) = task_id {
            std::fs::create_dir_all(profile.join("tasks")).unwrap();
            std::fs::write(profile.join("tasks").join(format!("{tid}.json")), "{}").unwrap();
        }
        WslFixtureGuard(profile)
    }

    /// A unique fixture profile name for this test process.
    #[cfg(unix)]
    fn fixture_name(tag: &str) -> String {
        let n = TMP.fetch_add(1, Ord::SeqCst);
        format!("castortest-{}-{}-{tag}", std::process::id(), n)
    }

    /// Local state dir stand-in for the WSL `~/.castor` (no task file).
    #[cfg(unix)]
    fn wsl_local_state() -> StateDir {
        let n = TMP.fetch_add(1, Ord::SeqCst);
        let p =
            std::env::temp_dir().join(format!("castor-wsl-resolve-{}-{}", std::process::id(), n));
        let s = StateDir::new(p);
        let _ = std::fs::create_dir_all(s.tasks());
        s
    }

    /// Serialize `USER` env mutations across tests. In edition 2024
    /// `std::env::set_var`/`remove_var` are `unsafe` (concurrent mutation
    /// from other test threads is UB), so the helper below wraps its
    /// mutation in this lock, mirroring `proxy::sanitize`'s `ENV_LOCK`.
    #[cfg(unix)]
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `resolve_cross_env_state_dir` with `USER` forced to `fake_user`,
    /// restoring the previous value before returning.
    #[cfg(unix)]
    fn resolve_with_user(state: &StateDir, task_id: &str, fake_user: &str) -> StateDir {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let old = std::env::var_os("USER");
        unsafe {
            std::env::set_var("USER", fake_user);
        }
        let out = resolve_cross_env_state_dir(state, task_id);
        unsafe {
            if let Some(old) = old {
                std::env::set_var("USER", old);
            } else {
                std::env::remove_var("USER");
            }
        }
        out
    }

    /// Pass 1: a task file in a foreign profile must be found by scanning
    /// `/mnt/c/Users/*` — even when a *different* foreign profile holds a
    /// `status.port` and our own profile is empty.
    #[test]
    #[cfg(unix)]
    fn cross_env_pass1_foreign_task_file_beats_foreign_status_port() {
        if !wsl_probe_available() {
            eprintln!("skipped: /mnt/c/Users not present/writable (not WSL or read-only mount)");
            return;
        }
        let local = wsl_local_state();
        let _port_only = make_wsl_fixture(&fixture_name("port"), None, Some(18021));
        let owner = make_wsl_fixture(&fixture_name("task"), Some("task_cross_env_a"), None);

        let got = resolve_cross_env_state_dir(&local, "task_cross_env_a");
        assert_eq!(
            got.root(),
            owner.0.as_path(),
            "a task file in a foreign profile must select that profile"
        );
    }

    /// Pass 1: our own `$USER` profile is checked before the directory scan,
    /// so a task file in the own profile wins even when a foreign profile
    /// also has the same task file.
    #[test]
    #[cfg(unix)]
    fn cross_env_pass1_own_profile_preferred_before_scan() {
        if !wsl_probe_available() {
            eprintln!("skipped: /mnt/c/Users not present/writable (not WSL or read-only mount)");
            return;
        }
        let local = wsl_local_state();
        let fake_user = fixture_name("own");
        let _other = make_wsl_fixture(
            &fixture_name("other"),
            Some("task_cross_env_b"),
            Some(18021),
        );
        let own = make_wsl_fixture(&fake_user, Some("task_cross_env_b"), None);

        let got = resolve_with_user(&local, "task_cross_env_b", &fake_user);
        assert_eq!(
            got.root(),
            own.0.as_path(),
            "the $USER profile must be checked before scanning other profiles"
        );
    }

    /// Pass 2: a `status.port` in a foreign profile must never be adopted
    /// (no task file exists anywhere). `USER` is forced to a fixture user
    /// whose profile has no `.castor` at all, so the only `status.port` in
    /// play is the foreign one — resolution must stay on the local dir.
    #[test]
    #[cfg(unix)]
    fn cross_env_pass2_foreign_status_port_alone_not_hijacked() {
        if !wsl_probe_available() {
            eprintln!("skipped: /mnt/c/Users not present/writable (not WSL or read-only mount)");
            return;
        }
        let local = wsl_local_state();
        let fake_user = fixture_name("nope"); // this user has no .castor
        let _other = make_wsl_fixture(&fixture_name("onlyport"), None, Some(18021));

        let got = resolve_with_user(&local, "task_missing_everywhere", &fake_user);
        assert_eq!(
            got.root(),
            local.root(),
            "a foreign profile's status.port alone must not change the state dir"
        );
    }

    /// Pass 2: a `status.port` in our own `$USER` profile (with no task
    /// file anywhere) is a valid cross-boundary target.
    #[test]
    #[cfg(unix)]
    fn cross_env_pass2_own_status_port_only_qualifies() {
        if !wsl_probe_available() {
            eprintln!("skipped: /mnt/c/Users not present/writable (not WSL or read-only mount)");
            return;
        }
        let local = wsl_local_state();
        let fake_user = fixture_name("portown");
        let own = make_wsl_fixture(&fake_user, None, Some(18021));

        let got = resolve_with_user(&local, "task_nowhere_yet", &fake_user);
        assert_eq!(
            got.root(),
            own.0.as_path(),
            "the $USER profile's live status server must be adopted"
        );
    }

    /// A task file present in the *local* state dir short-circuits before
    /// any WSL probe.
    #[test]
    #[cfg(unix)]
    fn cross_env_local_task_file_wins_without_wsl_probe() {
        let local = wsl_local_state();
        let _ = std::fs::write(local.tasks().join("task_local_wins.json"), "{}");
        let got = resolve_cross_env_state_dir(&local, "task_local_wins");
        assert_eq!(got.root(), local.root());
    }

    /// Spin up an axum router on an ephemeral `127.0.0.1:0` and return the
    /// base URL of its single `/wait` route.
    async fn bind_ephemeral_wait(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}/wait")
    }

    async fn candidate_404() -> Response {
        (StatusCode::NOT_FOUND, r#"{"error":"task not found"}"#).into_response()
    }

    async fn candidate_200() -> Response {
        (
            StatusCode::OK,
            r#"{"found":true,"status":"completed","reason":"via second candidate"}"#,
        )
            .into_response()
    }

    /// 404-continue, tested directly against [`try_wait_candidate_urls`] (no
    /// global state, no WSL prerequisite): candidate 1 answers 404,
    /// candidate 2 (a second ephemeral `127.0.0.1:0` server) answers 200.
    /// The helper must *continue* past the 404 and return candidate 2's
    /// body. If the loop had bailed out on the 404 (the old `break`), the
    /// result would be `None` and the assertions fail — this is the
    /// discrimination the old `run_wait_cli`-level test was attempting
    /// through a global fetch log (which deadlocked on the non-reentrant
    /// `Mutex` and contaminated other tests).
    ///
    /// Two other branches are pinned for completeness: every candidate 404
    /// ⇒ `None` (the caller then falls back to disk polling), and a
    /// transport error on one candidate (no listener on the port) must also
    /// continue to the next candidate rather than aborting the loop.
    #[tokio::test]
    async fn try_wait_candidate_urls_404_first_continues_to_second() {
        let url404 = bind_ephemeral_wait(Router::new().route("/wait", get(candidate_404))).await;
        let url200 = bind_ephemeral_wait(Router::new().route("/wait", get(candidate_200))).await;
        let client = reqwest::Client::new();

        // 404 on the first candidate must not end the loop: the second
        // candidate's 200 body is surfaced verbatim.
        let body = try_wait_candidate_urls(&client, &[url404.clone(), url200.clone()])
            .await
            .expect("a 200 candidate must win over a 404 candidate");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "completed");
        assert_eq!(v["reason"], "via second candidate");

        // All candidates 404 ⇒ None (caller falls back to disk polling).
        assert!(
            try_wait_candidate_urls(&client, &[url404.clone(), url404])
                .await
                .is_none(),
            "all-404 candidates must yield None"
        );

        // A transport error (no listener on the port) must also continue to
        // the next candidate rather than aborting the loop.
        let dead = "http://127.0.0.1:1/wait".to_string();
        let body = try_wait_candidate_urls(&client, &[dead, url200])
            .await
            .expect("a transport error on candidate 1 must continue to candidate 2");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "completed");
    }
}
