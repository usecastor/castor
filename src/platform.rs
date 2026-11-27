//! Platform detection and OS-specific process management.

/// Current wall-clock time as Unix-epoch milliseconds.
pub fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Liveness probe for a pid (signal 0 on Unix, PROCESS_QUERY_LIMITED_INFORMATION on Windows).
/// Conservative: a probe that errors with EPERM means the process exists and is treated as alive.
///
/// On Linux a successful `kill(pid, 0)` also reports for **zombie** (`<defunct>`) processes.
/// A zombie has already exited — it cannot execute code, heartbeat, or release its slot —
/// so we re-inspect `/proc/{pid}/status` and treat `State: Z` as dead.
#[cfg(unix)]
pub fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    if pid == std::process::id() as u32 {
        return true;
    }
    let r = unsafe { libc::kill(pid as i32, 0) };
    if r == 0 {
        #[cfg(target_os = "linux")]
        {
            return !is_linux_zombie(pid);
        }
        #[cfg(not(target_os = "linux"))]
        {
            return true;
        }
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(target_os = "linux")]
fn is_linux_zombie(pid: u32) -> bool {
    let path = format!("/proc/{pid}/status");
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    content
        .lines()
        .find(|l| l.starts_with("State:"))
        .map(|l| l.trim_start_matches("State:").trim_start().starts_with('Z'))
        .unwrap_or(false)
}

#[cfg(windows)]
pub fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    if pid == std::process::id() {
        return true;
    }
    unsafe {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        const STILL_ACTIVE: u32 = 259;
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if !handle.is_null() {
            let mut exit_code: u32 = 0;
            let success = GetExitCodeProcess(handle, &mut exit_code);
            CloseHandle(handle);
            success != 0 && exit_code == STILL_ACTIVE
        } else {
            false
        }
    }
}

#[cfg(all(not(unix), not(windows)))]
pub fn pid_alive(_pid: u32) -> bool {
    false
}

/// Terminate a process and its whole process group.
pub fn kill_process_tree(pid: u32) {
    if pid == 0 {
        return;
    }
    #[cfg(unix)]
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
        libc::kill(pid as i32, libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output();
    }
}

/// RAII guard for a Windows Job Object configured with
/// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Assigning a spawned child process
/// ensures the Windows kernel terminates the entire process tree (including
/// MSYS2 / Git Bash pipeline grandchildren) on [`Self::terminate`] or drop.
#[cfg(windows)]
pub struct ProcessJob {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(windows)]
unsafe impl Send for ProcessJob {}
#[cfg(windows)]
unsafe impl Sync for ProcessJob {}

#[cfg(windows)]
impl ProcessJob {
    /// Create a job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and
    /// assign `process_handle` to it. Returns `None` if the OS refuses job
    /// creation or assignment.
    pub fn attach(process_handle: std::os::windows::io::RawHandle) -> Option<Self> {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 || AssignProcessToJobObject(job, process_handle as _) == 0 {
                CloseHandle(job);
                return None;
            }
            Some(Self { handle: job })
        }
    }

    /// Terminate every process currently assigned to this job object.
    pub fn terminate(&self, exit_code: u32) {
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;
        unsafe {
            let _ = TerminateJobObject(self.handle, exit_code);
        }
    }
}

#[cfg(windows)]
impl Drop for ProcessJob {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

/// Translates a path to the host environment's native path format.
///
/// In WSL/Linux:
/// If passed a Windows-style path with a drive letter (e.g. `D:\foo\bar` or `d:/foo/bar`),
/// maps it to `/mnt/<drive>/foo/bar`.
///
/// In Windows:
/// If passed a WSL-style path with a `/mnt/<drive>` prefix (e.g. `/mnt/d/foo/bar`),
/// maps it to `<drive>:\foo\bar`.
pub fn to_host_path(p: impl AsRef<std::path::Path>) -> std::path::PathBuf {
    let path = p.as_ref();
    #[cfg(unix)]
    {
        return std::path::PathBuf::from(to_wsl_path(path));
    }
    #[cfg(windows)]
    {
        return std::path::PathBuf::from(to_windows_path(path));
    }
    #[allow(unreachable_code)]
    path.to_path_buf()
}

/// Translates a WSL /mnt path (e.g. `/mnt/d/foo/bar`) to Windows format (e.g. `D:\foo\bar`).
///
/// Handles:
/// - `/mnt/<drive>/...` -> `<DRIVE>:\...`
/// - `/mnt/<drive>` -> `<DRIVE>:\`
/// - `\mnt\<drive>\...` -> `<DRIVE>:\...`
/// - Preserves Windows drive-letter paths (normalizing to backslashes).
/// - Preserves non-/mnt paths as-is.
pub fn to_windows_path(p: impl AsRef<std::path::Path>) -> String {
    let s = p.as_ref().to_string_lossy();
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let normalized = trimmed.replace('\\', "/");
    if let Some(after_mnt) = normalized.strip_prefix("/mnt/") {
        let bytes = after_mnt.as_bytes();
        if !bytes.is_empty() && bytes[0].is_ascii_alphabetic() {
            let drive = (bytes[0] as char).to_ascii_uppercase();
            if bytes.len() == 1 {
                return format!("{drive}:\\");
            } else if bytes[1] == b'/' {
                let rest = after_mnt[2..].replace('/', "\\");
                let clean_rest = rest.trim_start_matches('\\');
                if clean_rest.is_empty() {
                    return format!("{drive}:\\");
                } else {
                    return format!("{drive}:\\{clean_rest}");
                }
            }
        }
    }
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        let drive = (bytes[0] as char).to_ascii_uppercase();
        let rest = if bytes.len() >= 3 && (bytes[2] == b'\\' || bytes[2] == b'/') {
            &trimmed[3..]
        } else {
            &trimmed[2..]
        };
        let clean_rest = rest.replace('/', "\\");
        let trimmed_rest = clean_rest.trim_start_matches('\\');
        if trimmed_rest.is_empty() {
            format!("{drive}:\\")
        } else {
            format!("{drive}:\\{trimmed_rest}")
        }
    } else {
        trimmed.to_string()
    }
}

/// Translates a Windows path with a drive letter to WSL /mnt format (e.g. `D:\foo\bar` -> `/mnt/d/foo/bar`).
pub fn to_wsl_path(p: impl AsRef<std::path::Path>) -> String {
    let s = p.as_ref().to_string_lossy();
    let trimmed = s.trim();
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        let drive = (bytes[0] as char).to_ascii_lowercase();
        let rest = if bytes.len() >= 3 && (bytes[2] == b'\\' || bytes[2] == b'/') {
            &trimmed[3..]
        } else {
            &trimmed[2..]
        };
        let normalized_rest = rest.replace('\\', "/");
        let clean_rest = normalized_rest.trim_start_matches('/');
        if clean_rest.is_empty() {
            format!("/mnt/{drive}")
        } else {
            format!("/mnt/{drive}/{clean_rest}")
        }
    } else {
        trimmed.replace('\\', "/")
    }
}

#[cfg(any(windows, test))]
const WINDOWS_GIT_BASH_CANDIDATES: [&str; 4] = [
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
    r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
];

/// Resolves `bash.exe` relative to a discovered `git.exe` path
/// (`<git_dir>\<subdir>\git.exe`), preferring `bin\bash.exe` and falling back
/// to `usr\bin\bash.exe`.
#[cfg(any(windows, test))]
fn bash_from_git_exe(git_path: &std::path::Path) -> Option<std::path::PathBuf> {
    let git_dir = git_path.parent()?.parent()?;
    [
        git_dir.join("bin").join("bash.exe"),
        git_dir.join("usr").join("bin").join("bash.exe"),
    ]
    .into_iter()
    .find(|candidate| candidate.is_file())
}

/// Returns the path to the bash executable.
///
/// On Windows, prefers Git for Windows bash (`bin\bash.exe`, falling back to
/// `usr\bin\bash.exe`) over Windows' System32 `bash.exe` stub (which redirects
/// to WSL).
pub fn bash_path() -> std::path::PathBuf {
    #[cfg(windows)]
    {
        for candidate in WINDOWS_GIT_BASH_CANDIDATES {
            let p = std::path::Path::new(candidate);
            if p.is_file() {
                return p.to_path_buf();
            }
        }
        if let Ok(output) = std::process::Command::new("where.exe")
            .arg("git.exe")
            .output()
            && output.status.success()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let git_path = std::path::Path::new(line.trim());
                if let Some(bash_bin) = bash_from_git_exe(git_path) {
                    return bash_bin;
                }
            }
        }
    }
    std::path::PathBuf::from("bash")
}

/// Detects whether the process is running inside WSL.
///
/// Prefers the WSL interop marker
/// (`/proc/sys/fs/binfmt_misc/WSLInterop`); falls back to a substring check
/// for `microsoft` in `/proc/version`. Both artifacts are absent on native
/// Windows or bare Linux, so this returns `false` there.
pub fn is_wsl() -> bool {
    if std::path::Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists() {
        return true;
    }
    std::fs::read_to_string("/proc/version")
        .ok()
        .map(|s| s.contains("microsoft"))
        .unwrap_or(false)
}

/// Strip a leading `wsl`/`wsl.exe` interop wrapper (and its options) from a
/// command, returning the inner command to run directly.
///
/// `wsl` takes options *before* the command it executes: `-d`/`--distribution
/// <name>`, `-u`/`--user <name>`, `-e`/`--exec`, and the `--` separator. This
/// consumes the leading `wsl`/`wsl.exe` token plus any of those options so the
/// inner command can be executed directly:
///
///   `wsl -d Ubuntu -- bash -lc "pkill vllm"` -> `bash -lc "pkill vllm"`
///   `wsl -u root pkill -f vllm`              -> `pkill -f vllm`
///   `wsl pkill -f vllm`                       -> `pkill -f vllm`
///
/// Commands that do not start with `wsl`/`wsl.exe` are returned unchanged.
/// The logic is pure (no WSL detection) so it is testable on every platform.
pub fn strip_wsl_wrapper(cmd: &str) -> String {
    let t = cmd.trim();
    let mut tokens: Vec<&str> = t.split_whitespace().collect();
    let Some(first) = tokens.first().copied() else {
        return cmd.to_string();
    };
    if first != "wsl" && first != "wsl.exe" {
        return cmd.to_string();
    }
    tokens.remove(0);
    while let Some(flag) = tokens.first().copied() {
        match flag {
            // Options that consume the following value.
            "-d" | "--distribution" | "-u" | "--user" => {
                tokens.remove(0);
                if !tokens.is_empty() {
                    tokens.remove(0);
                }
            }
            // Flag-only options and the arg-separator / exec markers.
            "-e" | "--exec" | "--" => {
                tokens.remove(0);
            }
            _ => break,
        }
    }
    tokens.join(" ")
}

/// Normalizes a Windows command for execution from WSL.
///
/// When the process runs under WSL, a leading `cmd` or `taskkill` token is
/// rewritten to its explicit `.exe` form
/// (e.g. `taskkill /F /IM llama-server.exe` -> `taskkill.exe /F /IM llama-server.exe`),
/// and a leading `wsl`/`wsl.exe` interop wrapper is stripped so the inner
/// command runs directly (never `sh -c "wsl ..."`). Otherwise the command is
/// returned unchanged.
pub fn wsl_normalize(cmd: &str) -> String {
    if !is_wsl() {
        return cmd.to_string();
    }
    normalize_launch_command(cmd)
}

/// Normalizes a command string for execution under POSIX `sh -c`.
///
/// Prevents command-not-found failures (exit 127) when commands carry Windows backslashes
/// in path segments (e.g. `scripts\wsl\start.sh` or `D:\models\llama.exe`):
/// - Strips leading `wsl` / `wsl.exe` wrappers.
/// - Rewrites `cmd` / `taskkill` to their `.exe` forms under WSL.
/// - Converts Windows backslashes in path segments to forward slashes.
/// - Translates Windows drive letter paths (e.g. `D:\...`) to `/mnt/d/...` under WSL/Linux.
pub fn normalize_launch_command(cmd: &str) -> String {
    let stripped = strip_wsl_wrapper(cmd);
    let trimmed = stripped.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let first_end = trimmed
        .find(|c: char| c.is_whitespace())
        .unwrap_or(trimmed.len());
    let first = &trimmed[..first_end];
    let rest = trimmed[first_end..].trim_start();
    match first {
        "cmd" | "cmd.exe" => {
            let safe_rest = rest.replace('\\', "/");
            let cmd_bin = if is_wsl() { "cmd.exe" } else { first };
            if safe_rest.is_empty() {
                cmd_bin.to_string()
            } else {
                format!("{cmd_bin} {safe_rest}")
            }
        }
        "taskkill" => {
            let tk_bin = if is_wsl() { "taskkill.exe" } else { first };
            if rest.is_empty() {
                tk_bin.to_string()
            } else {
                format!("{tk_bin} {rest}")
            }
        }
        _ => normalize_command_string(&stripped),
    }
}

/// Helper that converts Windows path separators to forward slashes in command strings
/// while preserving intentional shell escape characters (`\"`, `\'`, `\ `).
pub fn normalize_command_string(cmd: &str) -> String {
    let mut out = String::with_capacity(cmd.len());
    let mut chars = cmd.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.peek() {
                Some(&next)
                    if next == ' ' || next == '"' || next == '\'' || next == '$' || next == '`' =>
                {
                    out.push('\\');
                    out.push(next);
                    chars.next();
                }
                Some(&'\\') => {
                    chars.next();
                    out.push('/');
                }
                Some(_) => {
                    out.push('/');
                }
                None => {
                    out.push('\\');
                }
            }
        } else {
            out.push(ch);
        }
    }
    to_wsl_command_paths(&out)
}

fn to_wsl_command_paths(cmd: &str) -> String {
    let words: Vec<&str> = cmd.split(' ').collect();
    let mut normalized_words = Vec::new();
    for word in words {
        let b = word.as_bytes();
        if b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && b[2] == b'/' {
            let drive = (b[0] as char).to_ascii_lowercase();
            let path_part = &word[3..];
            normalized_words.push(format!("/mnt/{drive}/{path_part}"));
        } else {
            normalized_words.push(word.to_string());
        }
    }
    normalized_words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_host_path() {
        #[cfg(unix)]
        {
            assert_eq!(
                to_host_path("D:\\LLM_Ecosystem"),
                std::path::PathBuf::from("/mnt/d/LLM_Ecosystem")
            );
            assert_eq!(
                to_host_path("c:/users/alice"),
                std::path::PathBuf::from("/mnt/c/users/alice")
            );
            assert_eq!(
                to_host_path("/mnt/d/LLM_Ecosystem"),
                std::path::PathBuf::from("/mnt/d/LLM_Ecosystem")
            );
            assert_eq!(
                to_host_path("relative/path"),
                std::path::PathBuf::from("relative/path")
            );
        }
        #[cfg(windows)]
        {
            assert_eq!(
                to_host_path("D:\\LLM_Ecosystem"),
                std::path::PathBuf::from("D:\\LLM_Ecosystem")
            );
            assert_eq!(
                to_host_path("/mnt/d/LLM_Ecosystem"),
                std::path::PathBuf::from("D:\\LLM_Ecosystem")
            );
            assert_eq!(
                to_host_path("/mnt/c/users/alice"),
                std::path::PathBuf::from("C:\\users\\alice")
            );
            assert_eq!(
                to_host_path("relative/path"),
                std::path::PathBuf::from("relative/path")
            );
        }
    }

    #[test]
    fn test_to_windows_path() {
        assert_eq!(to_windows_path("/mnt/d/LLM_Ecosystem"), "D:\\LLM_Ecosystem");
        assert_eq!(to_windows_path("/mnt/c/Users/Alice"), "C:\\Users\\Alice");
        assert_eq!(to_windows_path("/mnt/d"), "D:\\");
        assert_eq!(to_windows_path("/mnt/d/"), "D:\\");
        assert_eq!(to_windows_path("D:\\LLM_Ecosystem"), "D:\\LLM_Ecosystem");
        assert_eq!(to_windows_path("d:/LLM_Ecosystem"), "D:\\LLM_Ecosystem");
        assert_eq!(to_windows_path("relative/path"), "relative/path");
        assert_eq!(to_windows_path("/tmp/castor-x"), "/tmp/castor-x");
    }

    #[test]
    fn test_to_wsl_path() {
        assert_eq!(to_wsl_path("D:\\LLM_Ecosystem"), "/mnt/d/LLM_Ecosystem");
        assert_eq!(to_wsl_path("c:/users/alice"), "/mnt/c/users/alice");
        assert_eq!(to_wsl_path("D:\\"), "/mnt/d");
        assert_eq!(to_wsl_path("/mnt/d/LLM_Ecosystem"), "/mnt/d/LLM_Ecosystem");
        assert_eq!(to_wsl_path("relative/path"), "relative/path");
    }

    #[cfg(unix)]
    #[test]
    fn test_is_wsl_detection() {
        // On this CI host (WSL2) the marker exists; the assertion is guarded so
        // it remains meaningful only where WSL detection is observable.
        let wsl = is_wsl();
        let has_marker = std::path::Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists();
        // Detection is consistent with at least one artifact.
        assert!(
            wsl || !has_marker,
            "is_wsl() inconsistent with WSLInterop marker"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_wsl_normalize() {
        // Under WSL, bare `cmd`/`taskkill` tokens get an explicit `.exe`.
        if is_wsl() {
            assert_eq!(wsl_normalize("taskkill"), "taskkill.exe");
            assert_eq!(wsl_normalize("cmd"), "cmd.exe");
            assert_eq!(
                wsl_normalize("taskkill /F /T /PID 1234"),
                "taskkill.exe /F /T /PID 1234"
            );
            assert_eq!(wsl_normalize("cmd /c echo hi"), "cmd.exe /c echo hi");
            assert_eq!(
                wsl_normalize(r"cmd /c scripts\bonsai\start.bat"),
                "cmd.exe /c scripts/bonsai/start.bat"
            );
            // A `wsl` interop wrapper is stripped so the inner command runs
            // directly rather than as `sh -c "wsl ..."`.
            assert_eq!(
                wsl_normalize(r#"wsl -d Ubuntu -- bash -lc "pkill vllm""#),
                r#"bash -lc "pkill vllm""#
            );
            assert_eq!(wsl_normalize("wsl pkill -f vllm"), "pkill -f vllm");
            assert_eq!(
                wsl_normalize("wsl.exe -u root pkill -f vllm"),
                "pkill -f vllm"
            );
            // Non-matching commands are untouched.
            assert_eq!(wsl_normalize("nvidia-smi"), "nvidia-smi");
            assert_eq!(
                wsl_normalize("taskkillX /F /IM foo"),
                "taskkillX /F /IM foo"
            );
        } else {
            // Off-WSL, normalization is the identity.
            assert_eq!(wsl_normalize("taskkill"), "taskkill");
            assert_eq!(wsl_normalize("cmd"), "cmd");
        }
    }

    /// `strip_wsl_wrapper` is pure (no WSL detection) so it is testable on any
    /// platform.
    #[test]
    fn test_strip_wsl_wrapper() {
        // Distribution + separator: options before the inner command are dropped.
        assert_eq!(
            strip_wsl_wrapper(r#"wsl -d Ubuntu -- bash -lc "pkill vllm""#),
            r#"bash -lc "pkill vllm""#
        );
        // Bare wrapper, no options.
        assert_eq!(strip_wsl_wrapper("wsl pkill -f vllm"), "pkill -f vllm");
        // `.exe` form plus a `--user <arg>` option.
        assert_eq!(
            strip_wsl_wrapper("wsl.exe -u root pkill -f vllm"),
            "pkill -f vllm"
        );
        // Long option forms and `--exec`.
        assert_eq!(
            strip_wsl_wrapper("wsl --distribution Ubuntu --user root -- pkill -f vllm"),
            "pkill -f vllm"
        );
        assert_eq!(
            strip_wsl_wrapper("wsl --exec pkill -f vllm"),
            "pkill -f vllm"
        );
        // A `wsl` that is merely a prefix of a longer token is left alone.
        assert_eq!(
            strip_wsl_wrapper("wslx -d Ubuntu -- bash -lc hi"),
            "wslx -d Ubuntu -- bash -lc hi"
        );
        // A command with no wrapper is returned unchanged.
        assert_eq!(
            strip_wsl_wrapper("bash -lc \"pkill vllm\""),
            "bash -lc \"pkill vllm\""
        );
        assert_eq!(strip_wsl_wrapper("nvidia-smi"), "nvidia-smi");
    }

    #[test]
    fn test_normalize_launch_command_paths() {
        let cmd = r"bash scripts\wsl\start_swarm.sh";
        let norm = normalize_launch_command(cmd);
        assert_eq!(norm, "bash scripts/wsl/start_swarm.sh");

        let cmd2 = r"scripts\wsl\start_swarm.sh";
        let norm2 = normalize_launch_command(cmd2);
        assert_eq!(norm2, "scripts/wsl/start_swarm.sh");

        let cmd3 = r"cmd.exe /c scripts\bonsai\start.bat";
        let norm3 = normalize_launch_command(cmd3);
        assert_eq!(norm3, "cmd.exe /c scripts/bonsai/start.bat");

        let cmd4 = r#"echo "hello \"world\"""#;
        let norm4 = normalize_launch_command(cmd4);
        assert_eq!(norm4, r#"echo "hello \"world\"""#);

        let cmd5 = r"D:\LLM_Ecosystem\scripts\start.sh";
        let norm5 = normalize_launch_command(cmd5);
        assert!(norm5.contains("LLM_Ecosystem/scripts/start.sh"));
        assert!(!norm5.contains('\\'));
    }

    #[test]
    fn test_windows_git_bash_candidates_and_usr_bin_fallback() {
        assert_eq!(
            WINDOWS_GIT_BASH_CANDIDATES,
            [
                r"C:\Program Files\Git\bin\bash.exe",
                r"C:\Program Files (x86)\Git\bin\bash.exe",
                r"C:\Program Files\Git\usr\bin\bash.exe",
                r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
            ]
        );

        let root = std::env::temp_dir().join(format!(
            "castor_platform_bash_fallback_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let cmd_dir = root.join("cmd");
        let bin_dir = root.join("bin");
        let usr_bin_dir = root.join("usr").join("bin");
        std::fs::create_dir_all(&cmd_dir).unwrap();
        std::fs::create_dir_all(&bin_dir).unwrap();
        std::fs::create_dir_all(&usr_bin_dir).unwrap();

        let git_exe = cmd_dir.join("git.exe");
        std::fs::write(&git_exe, b"").unwrap();
        assert_eq!(bash_from_git_exe(&git_exe), None);

        let usr_bin_bash = usr_bin_dir.join("bash.exe");
        std::fs::write(&usr_bin_bash, b"").unwrap();
        assert_eq!(
            bash_from_git_exe(&git_exe),
            Some(usr_bin_bash.clone()),
            "should fall back to <git_dir>/usr/bin/bash.exe when bin/bash.exe is absent"
        );

        let bin_bash = bin_dir.join("bash.exe");
        std::fs::write(&bin_bash, b"").unwrap();
        assert_eq!(
            bash_from_git_exe(&git_exe),
            Some(bin_bash),
            "should prefer <git_dir>/bin/bash.exe when both exist"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
