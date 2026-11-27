use std::path::{Path, PathBuf};
use std::time::Duration;

use super::process::*;
use super::scanner::*;
use super::*;

#[test]
fn fork_bomb_blocked() {
    assert!(matches!(
        validate(":(){ :|:& };:"),
        Err(ShellPolicyError::ProhibitedPattern(_))
    ));
}

#[test]
fn fork_bomb_no_space_blocked() {
    assert!(matches!(
        validate(":(){ :|:& };"),
        Err(ShellPolicyError::ProhibitedPattern(_))
    ));
}

#[test]
fn fork_bomb_space_before_brace_blocked() {
    assert!(matches!(
        validate(":() { :|:& };"),
        Err(ShellPolicyError::ProhibitedPattern(_))
    ));
}

#[test]
fn mkfs_blocked() {
    assert!(matches!(
        validate("mkfs.ext4 /dev/sda1"),
        Err(ShellPolicyError::ProhibitedPattern(_))
    ));
}

#[test]
fn fdisk_blocked() {
    assert!(matches!(
        validate("fdisk /dev/sda"),
        Err(ShellPolicyError::ProhibitedPattern(_))
    ));
}

#[test]
fn format_drive_blocked() {
    assert!(matches!(
        validate("format C:"),
        Err(ShellPolicyError::ProhibitedPattern(_))
    ));
}

#[test]
fn dd_to_device_blocked() {
    assert!(matches!(
        validate("dd if=/dev/zero of=/dev/sda"),
        Err(ShellPolicyError::ProhibitedPattern(_))
    ));
}

#[test]
fn rm_rf_root_blocked() {
    assert!(matches!(
        validate("rm -rf /"),
        Err(ShellPolicyError::ProtectedRoot(_, _))
    ));
}

#[test]
fn rm_rf_home_blocked() {
    assert!(matches!(
        validate("rm -rf /home/user"),
        Err(ShellPolicyError::ProtectedRoot(_, _))
    ));
}

#[test]
fn sudo_rm_blocked() {
    assert!(matches!(
        validate("sudo rm -rf /"),
        Err(ShellPolicyError::ProtectedRoot(_, _))
    ));
}

#[test]
fn bash_c_rm_blocked() {
    assert!(matches!(
        validate("bash -c \"rm -rf /\""),
        Err(ShellPolicyError::ProtectedRoot(_, _))
    ));
}

#[test]
fn unexpanded_ref_blocked() {
    assert!(matches!(
        validate("rm -rf $HOME"),
        Err(ShellPolicyError::UnexpandedReference(_))
    ));
}

#[test]
fn deadman_fuse_canary() {
    assert!(matches!(
        validate(CANARY_DISASTER_FUSE_TOKEN),
        Err(ShellPolicyError::DeadManFuse)
    ));
}

#[test]
fn command_security_denials_carry_binding_directive() {
    let protected = validate("rm -rf /").unwrap_err().to_string();
    let destructive = validate("mkfs.ext4 /dev/sda1").unwrap_err().to_string();
    for msg in [&protected, &destructive] {
        for needle in [
            "was not run",
            "binding security decision",
            "Do NOT bypass",
            "Lead Architect / user",
        ] {
            assert!(msg.contains(needle), "refusal missing '{needle}': {msg}");
        }
    }
    let unexpanded = validate("rm -rf $HOME").unwrap_err().to_string();
    for needle in [
        "was not run",
        "Expand the reference to a concrete, in-workspace path",
        "do not substitute an absolute out-of-workspace path",
        "Lead Architect / user",
    ] {
        assert!(
            unexpanded.contains(needle),
            "refusal missing '{needle}': {unexpanded}"
        );
    }
    let fuse = validate(CANARY_DISASTER_FUSE_TOKEN)
        .unwrap_err()
        .to_string();
    assert!(fuse.contains("was not run"), "{fuse}");
    assert!(fuse.contains("Do not retry this command"), "{fuse}");
    assert!(fuse.contains("Lead Architect / user"), "{fuse}");
}

#[test]
fn empty_command_rejected() {
    assert!(matches!(
        validate("   "),
        Err(ShellPolicyError::InvalidCommand)
    ));
}

#[test]
fn benign_commands_pass() {
    assert!(validate("ls -la").is_ok());
    assert!(validate("cargo build").is_ok());
    assert!(validate("rm -rf ./build").is_ok());
    assert!(validate("echo hello").is_ok());
    assert!(validate("cat file.txt").is_ok());
    assert!(validate("git status").is_ok());
}

#[test]
fn homoglyph_fullwidth_w_blocked() {
    assert!(matches!(
        validate("rm -rf /mnt/c/\u{FF37}indows"),
        Err(ShellPolicyError::ProtectedRoot(_, _))
    ));
}

// ---------------------------------------------------------------------------
// Bypass-regression tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bypass_tests {
    use super::*;

    #[test]
    fn bypass_single_ampersand_separator_blocked() {
        assert!(matches!(
            validate("echo ok & rm -rf /"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("true & rm -rf /home/user"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(validate("ls & ls").is_ok());
    }

    #[test]
    fn bypass_subshell_group_blocked() {
        assert!(matches!(
            validate("(rm -rf /)"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("( rm -rf /home/user )"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("cd /tmp && (rm -rf /)"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("if true; then (rm -rf /); fi"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(validate("echo \"(rm -rf /)\"").is_ok());
    }

    #[test]
    fn bypass_brace_group_blocked() {
        assert!(matches!(
            validate("{ rm -rf /; }"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("true; { rm -rf /home/user; }"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("{ (rm -rf /); }"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("( { rm -rf /; } )"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(validate("echo a{b c}d").is_ok());
    }

    #[test]
    fn bypass_env_wrapper_blocked() {
        assert!(matches!(
            validate("env FOO=1 rm -rf /"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("env FOO=1 -i rm -rf /home/user"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("env -i rm -rf /"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("env sudo rm -rf /"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(validate("env FOO=1 ls").is_ok());
        assert!(validate("env").is_ok());
    }

    #[test]
    fn bypass_cmd_substitution_operand_blocked() {
        assert!(matches!(
            validate("rm -rf `pwd`"),
            Err(ShellPolicyError::UnexpandedReference(_))
        ));
        assert!(matches!(
            validate("rm -rf $(pwd)"),
            Err(ShellPolicyError::UnexpandedReference(_))
        ));
        assert!(matches!(
            validate("rm -rf `ls`/secret"),
            Err(ShellPolicyError::UnexpandedReference(_))
        ));
        assert!(matches!(
            validate("rm -rf ${HOME}/x"),
            Err(ShellPolicyError::UnexpandedReference(_))
        ));
        assert!(matches!(
            validate("del `whoami`"),
            Err(ShellPolicyError::UnexpandedReference(_))
        ));
    }

    #[test]
    fn deeply_nested_groups_terminate() {
        assert!(matches!(
            validate("(((rm -rf /)))"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        let deep_bomb = "(".repeat(8) + "rm -rf /" + &")".repeat(8);
        assert!(matches!(
            validate(&deep_bomb),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        let deep_ok = "(".repeat(8) + "echo hi" + &")".repeat(8);
        assert!(validate(&deep_ok).is_ok());
        let very_deep = "(".repeat(2000) + "echo hi" + &")".repeat(2000);
        let r1 = validate(&very_deep).is_ok();
        let r2 = validate(&very_deep).is_err();
        assert!(r1 || r2);
    }
}

// ---------------------------------------------------------------------------
// Pipe-splitting regression tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod pipe_tests {
    use super::*;

    #[test]
    fn pipe_downstream_rm_rf_root_blocked() {
        assert!(matches!(
            validate("find . | xargs rm -rf /"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("cat file | xargs rm -rf /"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("echo a | rm -rf /"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn pipe_downstream_shell_split() {
        assert!(validate("curl http://x | sh").is_ok());
    }

    #[test]
    fn pipe_downstream_system_dir_blocked() {
        assert!(matches!(
            validate("echo x | rm -rf /etc/hosts"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("ls | xargs truncate -s 0 /etc/hosts"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn multi_stage_pipe_blocked() {
        assert!(matches!(
            validate("ls | grep x | xargs rm -rf /"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }
}

// ---------------------------------------------------------------------------
// System-directory protection regression tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod system_dir_tests {
    use super::*;

    fn ctx(cwd: &str, home: &str) -> (PathBuf, PathBuf) {
        (PathBuf::from(cwd), PathBuf::from(home))
    }

    #[test]
    fn truncate_system_dir_blocked() {
        assert!(matches!(
            validate("truncate -s 0 /etc/hosts"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        let (cwd, home) = ctx("/workspace", "/home/user");
        assert!(matches!(
            validate_with_context("truncate -s 0 /etc/hosts", &cwd, &home),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn mv_system_dir_blocked() {
        assert!(matches!(
            validate("mv /etc/hosts /tmp/"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("mv /tmp/evil /etc/"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn cp_system_dir_blocked() {
        assert!(matches!(
            validate("cp /tmp/evil /etc/"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("cp /etc/hosts /tmp/"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn tee_system_dir_blocked() {
        assert!(matches!(
            validate("echo x | tee /etc/crontab"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn shred_system_dir_blocked() {
        assert!(matches!(
            validate("shred /usr/lib/libc.so"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn wipefs_system_dir_blocked() {
        assert!(matches!(
            validate("wipefs /dev/sda"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn all_system_dirs_blocked_for_rm() {
        for dir in ["/etc", "/usr", "/var", "/boot", "/bin", "/lib", "/opt"] {
            let cmd = format!("rm -rf {dir}");
            assert!(
                matches!(validate(&cmd), Err(ShellPolicyError::ProtectedRoot(_, _))),
                "expected '{cmd}' to be blocked"
            );
        }
    }

    #[test]
    fn workspace_subpath_still_allowed() {
        assert!(validate("rm -rf ./build").is_ok());
        assert!(validate("mv ./a ./b").is_ok());
        assert!(validate("truncate -s 0 ./log.txt").is_ok());
        assert!(validate("cp ./src/main.rs ./backup/").is_ok());
    }

    #[test]
    fn context_anchors_tilde_to_real_home() {
        let (cwd, home) = ctx("/workspace", "/home/someone");
        assert!(matches!(
            validate_with_context("rm -rf ~", &cwd, &home),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn context_home_overrides_default() {
        assert!(validate("rm -rf ~/x").is_ok());
        let (cwd, home) = ctx("/workspace", "/root");
        assert!(matches!(
            validate_with_context("rm -rf ~", &cwd, &home),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn context_uses_real_cwd_for_relative_paths() {
        let (cwd, home) = ctx("/home/user/project", "/home/user");
        assert!(validate_with_context("rm -rf ./target", &cwd, &home).is_ok());
    }

    #[test]
    fn run_async_system_dir_refusal_never_spawns() {
        let dir =
            std::env::temp_dir().join(format!("castor_shell_run_async_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let res = run("truncate -s 0 /etc/hosts", &dir, Duration::from_secs(10));
        assert!(matches!(
            res,
            Err(ShellError::Policy(ShellPolicyError::ProtectedRoot(_, _)))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_async_benign_with_real_context_runs() {
        let dir =
            std::env::temp_dir().join(format!("castor_shell_run_async_ok_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let out = run("echo hello", &dir, Duration::from_secs(10)).unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout.trim(), "hello");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn single_arg_validate_still_available() {
        assert!(validate("rm -rf /").is_err());
        assert!(validate("ls -la").is_ok());
        assert!(!DEFAULT_CWD.is_empty());
        assert!(!DEFAULT_HOME.is_empty());
    }

    #[test]
    fn validate_with_context_empty_home_falls_back() {
        let cwd = Path::new("/workspace");
        let home = Path::new("");
        assert!(matches!(
            validate_with_context("rm -rf ~", cwd, home),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn windows_tilde_user_expansion_for_rm_and_subpaths() {
        let (cwd, home) = ctx("C:\\work", "C:\\Users\\bob");
        let err = validate_with_context("rm -rf ~alice", &cwd, &home).unwrap_err();
        match err {
            ShellPolicyError::ProtectedRoot(op, norm) => {
                assert_eq!(op, "~alice");
                assert_eq!(norm, "/mnt/c/users/alice");
            }
            other => panic!("expected ProtectedRoot, got {other:?}"),
        }
        assert!(
            validate_with_context("rm -rf ~alice/project/dist", &cwd, &home).is_ok(),
            "deeper subpath under Windows ~alice must remain allowed"
        );
    }

    #[cfg(unix)]
    #[test]
    fn unix_non_rm_destructive_verb_expands_tilde_to_unix_home() {
        let (cwd, home) = ctx("/workspace", "/home/bob");
        let err = validate_with_context("mv ~alice /tmp/x", &cwd, &home).unwrap_err();
        match err {
            ShellPolicyError::ProtectedRoot(op, norm) => {
                assert_eq!(op, "~alice");
                assert_eq!(norm, "/home/alice");
            }
            other => panic!("expected ProtectedRoot, got {other:?}"),
        }
    }

    #[test]
    fn windows_protected_system_subpaths_blocked() {
        for cmd in [
            "rm -rf C:\\Windows\\System32",
            "rm -rf /mnt/c/Windows/System32/drivers",
            "rm -rf \"C:\\Program Files\\Git\"",
            "rm -rf \"/mnt/c/Program Files/Common Files\"",
            "del /f /s /q \"C:\\Program Files (x86)\\App\"",
            "mv C:\\Windows\\notepad.exe /tmp/",
        ] {
            assert!(
                matches!(validate(cmd), Err(ShellPolicyError::ProtectedRoot(_, _))),
                "expected '{cmd}' to be blocked as ProtectedRoot"
            );
        }
    }

    #[test]
    fn windows_user_profile_root_blocked_but_project_subpath_allowed() {
        for cmd in [
            "rm -rf /mnt/c/Users/alice",
            "rm -rf C:\\Users\\alice",
            "rm -rf C:\\Users\\alice\\*",
        ] {
            assert!(
                matches!(validate(cmd), Err(ShellPolicyError::ProtectedRoot(_, _))),
                "expected '{cmd}' to be blocked as ProtectedRoot"
            );
        }
        for cmd in [
            "rm -rf /mnt/c/Users/alice/project/target",
            "rm -rf C:\\Users\\alice\\project\\dist",
        ] {
            assert!(
                validate(cmd).is_ok(),
                "expected project subpath '{cmd}' to be allowed"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Head+tail semantic truncation tests (Stage 1)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod head_tail_tests {
    use super::*;
    use std::collections::HashSet;

    /// Build a numbered one-line-per-iteration text block of exactly `n` lines.
    fn numbered(n: usize, tag: &str) -> String {
        (1..=n)
            .map(|i| format!("{tag} {i}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Collect all lines of a truncated block as a set, so we can prove which
    /// original lines survive and which were elided without substring ambiguity.
    fn line_set(block: &str) -> HashSet<String> {
        block.lines().map(str::to_string).collect()
    }

    /// Shared assertions that a head+tail truncation kept the head and tail while
    /// eliding the middle: `kept_lines` (total lines incl. the marker) == head+1+tail,
    /// the first/last head and first/last tail lines are present verbatim, and a
    /// representative set of middle lines are absent.
    fn assert_head_tail(
        block: &str,
        tag: &str,
        label: &str,
        head: usize,
        tail: usize,
        total_original: usize,
    ) {
        let lines: Vec<&str> = block.lines().collect();
        let elided = total_original - head - tail;
        assert_eq!(
            lines.len(),
            head + 1 + tail,
            "expected head+marker+tail lines for {label}"
        );
        // Head block intact: first and last kept head lines.
        assert_eq!(lines[0], format!("{tag} 1"), "head first line must survive");
        assert_eq!(
            lines[head - 1],
            format!("{tag} {head}"),
            "last head line must survive"
        );
        // Elision marker is exactly where head and tail meet.
        assert_eq!(
            lines[head],
            format!("[... {elided} {label} lines omitted ...]"),
            "elision marker must be correct"
        );
        // Tail block intact: first and last kept tail lines.
        // First kept tail line is original (total_original - tail + 1).
        let first_tail = total_original - tail + 1;
        assert_eq!(
            lines[head + 1],
            format!("{tag} {first_tail}"),
            "first tail line must survive"
        );
        assert_eq!(
            lines[lines.len() - 1],
            format!("{tag} {total_original}"),
            "last tail line must survive"
        );

        // The elided middle (original lines mid_start..=mid_end) is gone.
        let mid_start = head + 1;
        let mid_end = total_original - tail;
        let set = line_set(block);
        for probe in [mid_start, mid_start + 1, (mid_start + mid_end) / 2, mid_end] {
            assert!(
                !set.contains(&format!("{tag} {probe}")),
                "middle line {tag} {probe} must have been elided"
            );
        }
    }

    // -- success (exit 0) ---------------------------------------------------

    #[test]
    fn success_stdout_head_and_tail_survive_middle_omitted() {
        // 100 lines > budget 60 (head 25 + tail 35) → 40 elided.
        let stdout = numbered(100, "out");
        let (out, _, trunc) = semantic_truncate_output(&stdout, "", 0);
        assert!(trunc, "stdout over budget must be truncated");
        assert_head_tail(&out, "out", "stdout", 25, 35, 100);
    }

    #[test]
    fn success_stderr_head_and_tail_survive_middle_omitted() {
        // 200 lines > budget 100 (head 40 + tail 60) → 100 elided.
        let stderr = numbered(200, "err");
        let (_, err, trunc) = semantic_truncate_output("", &stderr, 0);
        assert!(trunc, "stderr over budget must be truncated");
        assert_head_tail(&err, "err", "stderr", 40, 60, 200);
    }

    // -- error (non-zero exit) ----------------------------------------------

    #[test]
    fn error_stdout_head_and_tail_survive_middle_omitted() {
        // 150 lines > budget 120 (head 40 + tail 80) → 30 elided.
        let stdout = numbered(150, "out");
        let (out, _, trunc) = semantic_truncate_output(&stdout, "", 1);
        assert!(trunc, "error stdout over budget must be truncated");
        assert_head_tail(&out, "out", "stdout", 40, 80, 150);
    }

    #[test]
    fn error_stderr_head_and_tail_survive_middle_omitted() {
        // 150 lines > budget 120 (head 40 + tail 80) → 30 elided.
        let stderr = numbered(150, "err");
        let (_, err, trunc) = semantic_truncate_output("", &stderr, 1);
        assert!(trunc, "error stderr over budget must be truncated");
        assert_head_tail(&err, "err", "stderr", 40, 80, 150);
    }

    // -- short / at-budget outputs are untouched ----------------------------

    #[test]
    fn short_outputs_below_threshold_untouched() {
        // Well under every budget → both streams byte-identical, not truncated.
        let (o, e, t) = semantic_truncate_output("a\nb\nc", "x\ny", 0);
        assert!(!t);
        assert_eq!(o, "a\nb\nc");
        assert_eq!(e, "x\ny");

        // Exactly at the success budgets (stdout 25+35=60, stderr 40+60=100):
        // no elision and `truncated == false`.
        let stdout = numbered(60, "o");
        let stderr = numbered(100, "e");
        let (o, e, t) = semantic_truncate_output(&stdout, &stderr, 0);
        assert!(!t, "at-budget success output must not be truncated");
        assert_eq!(o, stdout);
        assert_eq!(e, stderr);

        // Exactly at the error budgets (both 40+80=120) → untouched.
        let stdout = numbered(120, "o");
        let stderr = numbered(120, "e");
        let (o, e, t) = semantic_truncate_output(&stdout, &stderr, 3);
        assert!(!t, "at-budget error output must not be truncated");
        assert_eq!(o, stdout);
        assert_eq!(e, stderr);
    }

    // -- shared helper direct behavior --------------------------------------

    #[test]
    fn truncate_head_tail_helper_omits_middle_only() {
        // Over budget: head + marker + tail, middle elided.
        let text = numbered(100, "l");
        let (out, truncated) = truncate_head_tail(&text, 25, 35, "stdout");
        assert!(truncated);
        assert_head_tail(&out, "l", "stdout", 25, 35, 100);
        assert!(out.starts_with("l 1\n"));
        assert!(out.ends_with("l 100"));

        // At or below the 25 + 35 budget: returned unchanged, not truncated.
        let small = numbered(10, "l");
        let (out, truncated) = truncate_head_tail(&small, 25, 35, "stdout");
        assert!(!truncated);
        assert_eq!(out, small);
    }
}

// ---------------------------------------------------------------------------
// Executor tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod run_tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("castor_shell_test_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn benign_echo_runs_and_captures() {
        let dir = tmp_dir("echo");
        let out = run("echo hello", &dir, Duration::from_secs(10)).unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout.trim(), "hello");
        assert!(out.stderr.is_empty());
        assert!(!out.truncated);
        assert!(out.duration_ms >= 0.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refusal_rm_rf_root_never_spawns() {
        let dir = tmp_dir("refusal");
        let marker = dir.join("marker.txt");
        std::fs::write(&marker, "alive").unwrap();
        let res = run("rm -rf /", &dir, Duration::from_secs(10));
        assert!(matches!(
            res,
            Err(ShellError::Policy(ShellPolicyError::ProtectedRoot(_, _)))
        ));
        assert!(marker.exists(), "refused command must never spawn");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refusal_canary_never_spawns_marker() {
        let dir = tmp_dir("canary");
        let marker = dir.join("marker.txt");
        let cmd = format!("echo {CANARY_DISASTER_FUSE_TOKEN} > {}", marker.display());
        let res = run(&cmd, &dir, Duration::from_secs(10));
        assert!(matches!(
            res,
            Err(ShellError::Policy(ShellPolicyError::DeadManFuse))
        ));
        assert!(!marker.exists(), "refused command must never spawn");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn timeout_kills_process_group() {
        let dir = tmp_dir("timeout");
        let out = run("sleep 30", &dir, Duration::from_millis(300)).unwrap();
        assert_eq!(out.exit_code, 124);
        assert!(out.stderr.contains("timed out"));
        assert!(out.duration_ms < 5000.0, "duration was {}", out.duration_ms);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn timeout_returns_promptly_and_preserves_output_with_grandchild_pipe_holders() {
        let dir = tmp_dir("timeout_pipe_drain");
        let cmd = "echo PRE_TIMEOUT_STDOUT; echo PRE_TIMEOUT_STDERR >&2; sleep 30 | cat";
        let out = run(cmd, &dir, Duration::from_millis(300)).unwrap();
        assert_eq!(out.exit_code, 124);
        assert!(
            out.stdout.contains("PRE_TIMEOUT_STDOUT"),
            "pre-timeout stdout must be preserved: {:?}",
            out.stdout
        );
        assert!(
            out.stderr.contains("PRE_TIMEOUT_STDERR"),
            "pre-timeout stderr must be preserved: {:?}",
            out.stderr
        );
        assert!(
            out.stderr.contains("[Command timed out after 300ms]"),
            "missing timeout marker: {:?}",
            out.stderr
        );
        assert!(
            out.duration_ms < 3000.0,
            "timed-out pipeline must return promptly (<3000ms), took {}ms",
            out.duration_ms
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn output_cap_truncates() {
        let dir = tmp_dir("cap");
        let out = run(
            "head -c 300000 /dev/zero | tr '\\0' 'a'",
            &dir,
            Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(out.exit_code, 0);
        assert!(out.truncated);
        assert_eq!(out.stdout.len(), MAX_CAPTURE_BYTES);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn landlock_confinement_blocks_unauthorized_paths() {
        let dir = tmp_dir("landlock");
        let out_ok = run("echo hello > ok.txt", &dir, Duration::from_secs(10)).unwrap();
        assert_eq!(out_ok.exit_code, 0);
        assert!(dir.join("ok.txt").exists());

        let out_denied = run(
            "touch /etc/test_landlock_fail 2>&1 || ls /root 2>&1",
            &dir,
            Duration::from_secs(10),
        )
        .unwrap();
        let combined = format!("{} {}", out_denied.stdout, out_denied.stderr);
        assert!(
            combined.contains("Permission denied") || combined.contains("denied"),
            "Landlock must deny unauthorized paths: {combined}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn landlock_allows_dev_null_writes_and_toolchains() {
        let dir = tmp_dir("landlock_devnull");
        let out_devnull = run("echo test > /dev/null 2>&1", &dir, Duration::from_secs(10)).unwrap();
        assert_eq!(
            out_devnull.exit_code, 0,
            "Writing to /dev/null must succeed: {out_devnull:?}"
        );

        if std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".cargo/bin/cargo").exists())
            .unwrap_or(false)
        {
            let out_cargo = run("cargo --version", &dir, Duration::from_secs(10)).unwrap();
            assert_eq!(
                out_cargo.exit_code, 0,
                "cargo --version must succeed under Landlock: {out_cargo:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---------------------------------------------------------------------------
// UTF-8 regression tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod utf8_regression {
    use super::*;

    #[test]
    fn subshell_with_multibyte_still_blocked() {
        assert!(matches!(
            validate("(→a) (rm -rf /)"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("echo → (rm -rf /)"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("(rm -rf /) →"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("{ →; rm -rf /; }"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn case_body_with_multibyte_still_blocked() {
        assert!(matches!(
            validate("case $X in →) rm -rf / ;; esac"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(validate("case $X in a) rm -rf 📦 ;; b) echo done ;; esac").is_ok());
        assert!(validate("case $X in →) echo ok ;; esac").is_ok());
    }

    #[test]
    fn substitution_with_multibyte_does_not_panic() {
        assert!(validate("echo $(→a)").is_ok());
        assert!(validate("echo $(📦✨)").is_ok());
        assert!(validate("echo `→→→`").is_ok());
        assert!(matches!(
            validate("echo $(→; rm -rf /)"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn device_operand_with_multibyte_does_not_panic() {
        assert!(validate("rm /dev/s→").is_ok());
        assert!(validate("rm /dev/h→").is_ok());
        assert!(validate("rm /dev/nvme📦").is_ok());
        assert!(matches!(
            validate("rm /dev/sda"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("rm /dev/nvme0n1"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }

    #[test]
    fn benign_multibyte_commands_pass() {
        assert!(validate("echo → 📦 ✨").is_ok());
        assert!(validate("echo \"→→→\"").is_ok());
        assert!(validate("echo 你好 world").is_ok());
        assert!(validate("echo →; echo 📦; echo →").is_ok());
        assert!(validate("echo $(→) && echo 📦").is_ok());
        assert!(validate("ls → && cat 📦").is_ok());
    }

    #[test]
    fn multibyte_neighbor_of_delimiters_still_correct() {
        assert!(validate("echo →{rm").is_ok());
        assert!(validate("echo 📦{x").is_ok());
        assert!(matches!(
            validate("{ rm -rf / →; }"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("→ (rm -rf /)"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(matches!(
            validate("echo →$(rm -rf /)"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
        assert!(validate("echo →(x").is_ok());
        assert!(validate("case $X in a) echo ok ;; esac").is_ok());
        assert!(matches!(
            validate("case $X in →in) rm -rf / ;; esac"),
            Err(ShellPolicyError::ProtectedRoot(_, _))
        ));
    }
}
