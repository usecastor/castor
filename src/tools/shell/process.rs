//! Cross-platform command spawning, stream capture, and process tree lifecycle management.

use std::path::Path;
#[cfg(target_os = "linux")]
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use thiserror::Error;
use tokio::io::AsyncReadExt;

use super::scanner::ShellPolicyError;

/// Per-stream capture ceiling (256KB buffer cap).
pub(crate) const MAX_CAPTURE_BYTES: usize = 256 * 1024;

/// Maximum time to wait for stdout/stderr pipe readers to drain after killing a
/// timed-out process tree before aborting the reader tasks.
const POST_KILL_DRAIN_TIMEOUT: Duration = Duration::from_millis(500);

/// Head+tail line budgets for semantic truncation of shell output.
///
/// On success (exit 0): stdout keeps its first `SUCCESS_STDOUT_HEAD_LINES` and last
/// `SUCCESS_STDOUT_TAIL_LINES` lines (eliding the middle above 60); stderr keeps its
/// first `SUCCESS_STDERR_HEAD_LINES` and last `SUCCESS_STDERR_TAIL_LINES` lines
/// (eliding the middle above 100).
///
/// On error (non-zero exit): both streams keep their first 40 and last 80 lines
/// (eliding the middle above 120), so verbose flood at the top is trimmed while the
/// full tail of compiler diagnostics, stack traces, and final errors survives.
pub const SUCCESS_STDOUT_HEAD_LINES: usize = 25;
pub const SUCCESS_STDOUT_TAIL_LINES: usize = 35;
pub const SUCCESS_STDERR_HEAD_LINES: usize = 40;
pub const SUCCESS_STDERR_TAIL_LINES: usize = 60;
pub const ERROR_STDOUT_HEAD_LINES: usize = 40;
pub const ERROR_STDOUT_TAIL_LINES: usize = 80;
pub const ERROR_STDERR_HEAD_LINES: usize = 40;
pub const ERROR_STDERR_TAIL_LINES: usize = 80;

/// Head+tail line truncation shared by all shell output streams.
///
/// Retains the first `head_lines` and last `tail_lines` lines of `text`. When the total
/// line count exceeds `head_lines + tail_lines`, the middle is replaced by a single
/// elision marker `[... N {label} lines omitted ...]` where `N` is the number of elided
/// lines. Text at or below the `head_lines + tail_lines` budget is returned unchanged
/// with `false`.
pub fn truncate_head_tail(
    text: &str,
    head_lines: usize,
    tail_lines: usize,
    label: &str,
) -> (String, bool) {
    let lines: Vec<&str> = text.lines().collect();
    let budget = head_lines + tail_lines;
    if lines.len() <= budget {
        return (text.to_string(), false);
    }
    let elided = lines.len() - budget;
    let head = lines[..head_lines].join("\n");
    let tail = lines[lines.len() - tail_lines..].join("\n");
    let marker = format!("[... {elided} {label} lines omitted ...]");
    (format!("{head}\n{marker}\n{tail}"), true)
}

/// Semantic line truncation for shell execution output (head + tail retention).
///
/// On success (exit 0): stdout keeps its first 25 + last 35 lines (eliding the middle
/// when above 60); stderr keeps its first 40 + last 60 lines (eliding when above 100).
/// On error (non-zero exit): both streams keep their first 40 + last 80 lines (eliding
/// the middle when above 120), preserving both the early failure context and the full
/// tail of compiler diagnostics, failure context, and stack traces.
pub fn semantic_truncate_output(
    stdout: &str,
    stderr: &str,
    exit_code: i32,
) -> (String, String, bool) {
    let (out_head, out_tail, err_head, err_tail) = if exit_code == 0 {
        (
            SUCCESS_STDOUT_HEAD_LINES,
            SUCCESS_STDOUT_TAIL_LINES,
            SUCCESS_STDERR_HEAD_LINES,
            SUCCESS_STDERR_TAIL_LINES,
        )
    } else {
        (
            ERROR_STDOUT_HEAD_LINES,
            ERROR_STDOUT_TAIL_LINES,
            ERROR_STDERR_HEAD_LINES,
            ERROR_STDERR_TAIL_LINES,
        )
    };

    let (clean_stdout, out_trunc) = truncate_head_tail(stdout, out_head, out_tail, "stdout");
    let (clean_stderr, err_trunc) = truncate_head_tail(stderr, err_head, err_tail, "stderr");

    (clean_stdout, clean_stderr, out_trunc || err_trunc)
}

/// Result of a sandboxed shell execution.
#[derive(Debug)]
pub struct ShellOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    /// True when either stream exceeded the capture cap and was truncated.
    pub truncated: bool,
    /// Wall-clock duration in milliseconds (sub-ms precision).
    pub duration_ms: f64,
}

#[derive(Debug, Error)]
pub enum ShellError {
    #[error("policy: {0}")]
    Policy(#[from] ShellPolicyError),
    #[error("cwd: {0}")]
    Cwd(#[from] crate::tools::sandbox::SandboxError),
    #[error("spawn: {0}")]
    Spawn(#[from] std::io::Error),
}

/// Drains an async stream incrementally into a byte-capped shared buffer,
/// continuing to read (and discarding) past the cap so the child's pipe never
/// deadlocks. Writing incrementally preserves already-captured output if the
/// reader task is aborted after [`POST_KILL_DRAIN_TIMEOUT`].
async fn capture_capped<R: tokio::io::AsyncRead + Unpin>(
    mut r: R,
    cap: usize,
    out_buf: Arc<std::sync::Mutex<Vec<u8>>>,
    trunc_flag: Arc<AtomicBool>,
) {
    let mut buf = [0u8; 8192];
    loop {
        let n = match r.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        let mut guard = out_buf.lock().unwrap();
        if guard.len() >= cap {
            trunc_flag.store(true, Ordering::SeqCst);
        } else {
            let room = cap - guard.len();
            guard.extend_from_slice(&buf[..n.min(room)]);
            if n > room {
                trunc_flag.store(true, Ordering::SeqCst);
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn is_unsupported_landlock_fs(path: &Path) -> bool {
    use std::ffi::CString;
    let Ok(c_path) = CString::new(path.to_string_lossy().as_bytes()) else {
        return false;
    };
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statfs(c_path.as_ptr(), &mut stat) } != 0 {
        return false;
    }
    // Landlock PathBeneath rules are not supported on network/virtual filesystems:
    // 0x01021997 = V9FS_MAGIC (9p/drvfs in WSL), 0x53465342 = SMB2/DRVFS,
    // 0xfe534e51 = CIFS_MAGIC_NUMBER, 0x6969 = NFS_SUPER_MAGIC.
    let ft = stat.f_type as u64;
    let fs_name = match ft {
        0x01021997 => "9p/drvfs (WSL)",
        0x53465342 => "SMB2/DRVFS",
        0xfe534e51 => "CIFS/SMB",
        0x6969 => "NFS",
        _ => "",
    };
    if !fs_name.is_empty() {
        tracing::warn!(
            path = %path.display(),
            fs_type = %format!("0x{ft:08X}"),
            fs = fs_name,
            "Landlock PathBeneath rules are unsupported on this filesystem ({fs_name}); degrading to the in-process 5-layer sandbox for the shell child."
        );
        return true;
    }
    false
}

#[cfg(target_os = "linux")]
fn build_landlock_ruleset(cwd: &Path) -> Result<landlock::RulesetCreated, String> {
    if is_unsupported_landlock_fs(cwd) {
        return Err("virtual/network filesystem does not support Landlock".to_string());
    }
    use landlock::{
        ABI, Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
    };
    let abi = ABI::V1;
    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .map_err(|e| format!("Landlock handle_access: {e}"))?
        .create()
        .map_err(|e| format!("Landlock create: {e}"))?;

    // Read-Write for workspace root (cwd)
    let ws_fd = PathFd::new(cwd).map_err(|e| format!("PathFd cwd {}: {e}", cwd.display()))?;
    ruleset = ruleset
        .add_rule(PathBeneath::new(ws_fd, AccessFs::from_all(abi)))
        .map_err(|e| format!("Landlock add_rule cwd: {e}"))?;

    // Read-Write for /tmp and /dev (required for /dev/null, /dev/zero, /dev/urandom)
    for rw_dir in ["/tmp", "/dev"] {
        if let Ok(fd) = PathFd::new(rw_dir) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, AccessFs::from_all(abi)))
                .map_err(|e| format!("Landlock add_rule {rw_dir}: {e}"))?;
        }
    }

    // Read-Only for system paths
    let ro = AccessFs::from_read(abi);
    for dir in ["/usr", "/bin", "/lib", "/lib64", "/etc", "/proc"] {
        if let Ok(fd) = PathFd::new(dir) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, ro))
                .map_err(|e| format!("Landlock add_rule {dir}: {e}"))?;
        }
    }

    // Read-Only for user toolchains, state, and config directories. Includes
    // ~/.gitconfig and ~/.config so git and CLI tools (e.g. cargo, ripgrep,
    // editors) can read their user-level configuration without Permission
    // denied (EACCES) failures from Landlock confinement.
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        for sub in [
            ".castor",
            ".cargo",
            ".rustup",
            ".local",
            ".gitconfig",
            ".config",
        ] {
            let p = home.join(sub);
            if let Ok(fd) = PathFd::new(&p) {
                ruleset = ruleset
                    .add_rule(PathBeneath::new(fd, ro))
                    .map_err(|e| format!("Landlock add_rule {}: {e}", p.display()))?;
            }
        }
    }

    Ok(ruleset)
}

#[cfg(any(target_os = "linux", test))]
fn is_cargo_injected_lib_dir(entry: &str) -> bool {
    if entry.is_empty() {
        return false;
    }
    if entry.contains("/.rustup/toolchains/") {
        return true;
    }
    if let Some((_, after_target)) = entry.rsplit_once("/target/") {
        return after_target.ends_with("/deps")
            || after_target.ends_with("/debug")
            || after_target.ends_with("/release")
            || after_target == "debug"
            || after_target == "release"
            || (after_target.contains("/build/") && after_target.ends_with("/out"));
    }
    false
}

/// Filters Cargo-injected build-output and toolchain directories out of
/// `LD_LIBRARY_PATH` while preserving any user-supplied library paths.
#[cfg(any(target_os = "linux", test))]
fn sanitize_cargo_ld_library_path(raw: &str) -> Option<String> {
    let kept: Vec<&str> = raw
        .split(':')
        .filter(|entry| !entry.is_empty() && !is_cargo_injected_lib_dir(entry))
        .collect();
    if kept.is_empty() {
        None
    } else {
        Some(kept.join(":"))
    }
}

pub async fn execute(cmd: &str, cwd: &Path, timeout: Duration) -> Result<ShellOutput, ShellError> {
    let t0 = Instant::now();
    let bash = crate::platform::bash_path();
    let mut cmd_builder = tokio::process::Command::new(bash);
    cmd_builder
        .arg("-c")
        .arg(cmd)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(unix)]
    cmd_builder.process_group(0);

    #[cfg(target_os = "linux")]
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let cargo_bin = home.join(".cargo/bin");
        let local_bin = home.join(".local/bin");
        let current_path = std::env::var("PATH").unwrap_or_default();
        let mut new_path = String::new();
        if cargo_bin.is_dir() {
            new_path.push_str(&cargo_bin.to_string_lossy());
            new_path.push(':');
        }
        if local_bin.is_dir() {
            new_path.push_str(&local_bin.to_string_lossy());
            new_path.push(':');
        }
        new_path.push_str(&current_path);
        cmd_builder.env("PATH", new_path);
    }

    #[cfg(target_os = "linux")]
    {
        // When executing under Linux or WSL mounts (such as drvfs 9p mounts), compilers
        // writing temporary files to /tmp (ext4) and atomically renaming to target dirs
        // fail with EXDEV (os error 18, Invalid cross-device link). Scoping TMPDIR to
        // the workspace scratchpad ensures all temporary writes and atomic renames
        // remain on the same filesystem and within the workspace Landlock confinement rule.
        let scratch_tmp = cwd.join(".scratch").join("tmp");
        if std::fs::create_dir_all(&scratch_tmp).is_ok() {
            cmd_builder.env("TMPDIR", &scratch_tmp);
        }

        // When invoked from `cargo test` or `cargo run`, Cargo injects dozens of
        // `target/*/build/*/out` and toolchain directories into `LD_LIBRARY_PATH`.
        // Inheriting them into child `bash` processes on 9p/drvfs mounts forces `ld.so`
        // to spend hundreds of milliseconds probing non-existent directories before
        // executing the first command.
        if std::env::var_os("CARGO").is_some()
            && let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH")
        {
            match sanitize_cargo_ld_library_path(&ld_path) {
                Some(cleaned) => {
                    cmd_builder.env("LD_LIBRARY_PATH", cleaned);
                }
                None => {
                    cmd_builder.env_remove("LD_LIBRARY_PATH");
                }
            }
        }

        if let Ok(ruleset) = build_landlock_ruleset(cwd) {
            let mut ruleset_opt = Some(ruleset);
            unsafe {
                cmd_builder.pre_exec(move || {
                    // Best-effort confinement: if the kernel or virtualization layer
                    // rejects restrict_self (e.g. WSL2 hypervisor FD limitations returning EBADF/EPERM),
                    // degrade gracefully to in-process SandboxPolicy path validation rather than
                    // crashing child process spawns.
                    if let Some(r) = ruleset_opt.take() {
                        let _ = r.restrict_self();
                    }
                    Ok(())
                });
            }
        }
    }

    let mut child = cmd_builder.spawn()?;
    #[cfg(windows)]
    let _job = child
        .raw_handle()
        .and_then(crate::platform::ProcessJob::attach);
    let pid = child.id().unwrap_or(0);

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");

    let out_buf = Arc::new(std::sync::Mutex::new(Vec::new()));
    let err_buf = Arc::new(std::sync::Mutex::new(Vec::new()));
    let out_trunc = Arc::new(AtomicBool::new(false));
    let err_trunc = Arc::new(AtomicBool::new(false));

    let mut out_task = {
        let (buf, flag) = (Arc::clone(&out_buf), Arc::clone(&out_trunc));
        tokio::spawn(async move {
            capture_capped(stdout, MAX_CAPTURE_BYTES, buf, flag).await;
        })
    };
    let mut err_task = {
        let (buf, flag) = (Arc::clone(&err_buf), Arc::clone(&err_trunc));
        tokio::spawn(async move {
            capture_capped(stderr, MAX_CAPTURE_BYTES, buf, flag).await;
        })
    };

    // Poll spawned pipe readers once immediately so their reactor registrations
    // are active even on a fresh current-thread runtime before any sleep tick.
    tokio::task::yield_now().await;

    let deadline = tokio::time::Instant::now() + timeout;
    let mut status = None;
    loop {
        match child.try_wait() {
            Ok(Some(s)) => {
                status = Some(s);
                break;
            }
            Ok(None) => {
                if tokio::time::Instant::now() >= deadline {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(e) => {
                return Err(ShellError::Spawn(e));
            }
        }
    }

    let (exit_code, out, err) = match status {
        Some(status) => {
            let _ = tokio::join!(out_task, err_task);
            let out = std::mem::take(&mut *out_buf.lock().unwrap());
            let mut err = String::from_utf8_lossy(&std::mem::take(&mut *err_buf.lock().unwrap()))
                .into_owned();
            if status.code().is_none() {
                err.push_str("\n[Process terminated by signal]");
            }
            (
                if status.success() {
                    0
                } else {
                    status.code().unwrap_or(1)
                },
                out,
                err,
            )
        }
        None => {
            #[cfg(unix)]
            if pid > 0 {
                unsafe {
                    libc::killpg(pid as i32, libc::SIGKILL);
                    let mut st: libc::c_int = 0;
                    libc::waitpid(pid as i32, &mut st, 0);
                }
            }
            #[cfg(windows)]
            {
                if let Some(ref job) = _job {
                    job.terminate(124);
                }
                if pid > 0 {
                    crate::platform::kill_process_tree(pid);
                }
                let _ = child.kill().await;
            }
            #[cfg(not(any(unix, windows)))]
            {
                let _ = child.kill().await;
            }
            let drain_fut = async {
                let _ = tokio::join!(&mut out_task, &mut err_task);
            };
            if tokio::time::timeout(POST_KILL_DRAIN_TIMEOUT, drain_fut)
                .await
                .is_err()
            {
                out_task.abort();
                err_task.abort();
                let _ = out_task.await;
                let _ = err_task.await;
            }
            let out = std::mem::take(&mut *out_buf.lock().unwrap());
            let mut err = String::from_utf8_lossy(&std::mem::take(&mut *err_buf.lock().unwrap()))
                .into_owned();
            use std::fmt::Write;
            let _ = write!(err, "\n[Command timed out after {}ms]", timeout.as_millis());
            (124, out, err)
        }
    };

    let raw_stdout = String::from_utf8_lossy(&out).into_owned();
    let (clean_stdout, clean_stderr, sem_truncated) =
        semantic_truncate_output(&raw_stdout, &err, exit_code);
    let was_truncated =
        out_trunc.load(Ordering::SeqCst) || err_trunc.load(Ordering::SeqCst) || sem_truncated;

    Ok(ShellOutput {
        exit_code,
        stdout: clean_stdout,
        stderr: clean_stderr,
        truncated: was_truncated,
        duration_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn capture_capped_preserves_bytes_when_aborted_on_open_pipe() {
        use tokio::io::AsyncWriteExt;

        let (mut writer, reader) = tokio::io::duplex(4096);
        writer.write_all(b"CAPTURED_BEFORE_HANG\n").await.unwrap();
        writer.flush().await.unwrap();

        let out_buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let trunc_flag = Arc::new(AtomicBool::new(false));
        let mut task = {
            let (buf, flag) = (Arc::clone(&out_buf), Arc::clone(&trunc_flag));
            tokio::spawn(async move {
                capture_capped(reader, MAX_CAPTURE_BYTES, buf, flag).await;
            })
        };

        // Keep `writer` alive (simulating an escaped grandchild holding the pipe open).
        let drain_res = tokio::time::timeout(Duration::from_millis(100), &mut task).await;
        assert!(
            drain_res.is_err(),
            "pipe should remain open while writer lives"
        );
        task.abort();

        let captured = String::from_utf8_lossy(&out_buf.lock().unwrap()).into_owned();
        assert_eq!(captured, "CAPTURED_BEFORE_HANG\n");
        drop(writer);
    }

    #[test]
    fn sanitize_cargo_ld_library_path_strips_cargo_dirs_preserves_user_dirs() {
        let raw = "/opt/cuda/lib64:/mnt/c/repo/target/debug/build/aws-lc-sys-123/out:/mnt/c/repo/target/debug:/mnt/c/repo/target/debug/deps:/home/user/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib:/usr/local/lib";
        assert_eq!(
            sanitize_cargo_ld_library_path(raw).as_deref(),
            Some("/opt/cuda/lib64:/usr/local/lib")
        );

        let cargo_only = "/mnt/c/repo/target/debug/deps:/home/user/.rustup/toolchains/stable/lib";
        assert_eq!(sanitize_cargo_ld_library_path(cargo_only), None);
    }
}
