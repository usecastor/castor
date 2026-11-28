#![cfg(feature = "ast")]
//! Structural AST search/replace powered by `ast-grep` (tree-sitter).
//!
//! Structural capabilities:
//! - `ast_search` — code-like pattern matching with metavariables (`$VAR`, `$$$BODY`).
//! - `ast_replace` — structural rewrite that preserves formatting, with a
//!   mandatory syntax gate (reparse + ERROR-node check) and per-file rollback.
//!
//! All paths flow through the sandbox layers in [`super::sandbox`].
//!
//! Note on the "parse error" gate: tree-sitter always produces a tree (with
//! error recovery), so a malformed rewrite does not fail to parse — it
//! produces a tree containing `ERROR`/`UNDEFINED` nodes. The syntax gate
//! therefore reparses the rewritten content and treats the presence of any
//! ERROR-kind node as "verified invalid" → rollback.

use std::fs;
use std::path::{Path, PathBuf};

use ast_grep_core::matcher::Pattern;
use ast_grep_core::tree_sitter::LanguageExt;
use ast_grep_core::{Language, Node};
use ast_grep_language::SupportLang;
use thiserror::Error;

use super::sandbox::{self, SandboxError};

/// Directories skipped during recursive walks.
const IGNORED_DIRS: &[&str] = &[
    ".venv",
    "venv",
    "node_modules",
    ".git",
    "__pycache__",
    "target",
    "dist",
    "build",
    "vendor",
    ".idea",
    ".vscode",
];

/// Files larger than this are skipped (implausible for AST).
const MAX_FILE_BYTES: u64 = 1_000_000;

/// Global cap on matches returned by a search (single file or directory scan).
const MAX_MATCHES: usize = 25;

#[derive(Debug, Error)]
pub enum AstError {
    #[error("{0}")]
    Sandbox(#[from] SandboxError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    InvalidArgs(String),
    #[error("{0}")]
    Pattern(String),
    #[error("{0}")]
    Language(String),
    #[error("SyntaxValidationError: {0}")]
    SyntaxFailure(String),
}

/// A single AST match. `line`/`col` are 0-indexed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub text: String,
}

/// Summary of a batch replace.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplaceSummary {
    pub files_scanned: usize,
    pub files_applied: usize,
    pub files_rolled_back: usize,
    pub replacements: usize,
    pub applied: Vec<String>,
    pub rolled_back: Vec<String>,
}

/// Resolve a file path to its canonical `SupportLang` via
/// `SupportLang::from_path` with comprehensive extension fallbacks covering
/// all 27 bundled tree-sitter grammars (including `.htm`, `.yml`, `.bash`,
/// `.mjs`, `.cjs`, `.jsx`, `.mts`, `.cts`, `.pyi`, `.cc`, `.hpp`, `.jsonc`,
/// etc.). Returns `None` for unrecognized extensions (e.g. `.txt`, `.md`, or
/// arbitrary non-code files).
pub fn infer_lang_by_extension(path: &Path) -> Option<SupportLang> {
    if let Some(lang) = SupportLang::from_path(path) {
        return Some(lang);
    }
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some(SupportLang::Rust),
        "py" | "pyi" => Some(SupportLang::Python),
        "js" | "mjs" | "cjs" | "jsx" => Some(SupportLang::JavaScript),
        "ts" | "mts" | "cts" => Some(SupportLang::TypeScript),
        "tsx" => Some(SupportLang::Tsx),
        "go" => Some(SupportLang::Go),
        "c" | "h" => Some(SupportLang::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Some(SupportLang::Cpp),
        "java" => Some(SupportLang::Java),
        "cs" => Some(SupportLang::CSharp),
        "lua" => Some(SupportLang::Lua),
        "html" | "htm" | "xhtml" => Some(SupportLang::Html),
        "css" | "scss" => Some(SupportLang::Css),
        "json" | "jsonc" => Some(SupportLang::Json),
        "yaml" | "yml" => Some(SupportLang::Yaml),
        "sh" | "bash" | "zsh" => Some(SupportLang::Bash),
        "rb" => Some(SupportLang::Ruby),
        "php" => Some(SupportLang::Php),
        "swift" => Some(SupportLang::Swift),
        "kt" | "kts" => Some(SupportLang::Kotlin),
        "scala" => Some(SupportLang::Scala),
        "dart" => Some(SupportLang::Dart),
        "sol" => Some(SupportLang::Solidity),
        "hs" => Some(SupportLang::Haskell),
        "ex" | "exs" => Some(SupportLang::Elixir),
        "nix" => Some(SupportLang::Nix),
        _ => None,
    }
}

/// Parse a language alias (e.g. `"js"`, `"ts"`, `"py"`, `"rs"`) into a
/// `SupportLang`. All 27 canonical aliases from `SupportLang::all_langs` are
/// accepted (case-insensitively), along with standard common aliases.
fn parse_lang(alias: &str) -> Result<SupportLang, AstError> {
    if let Ok(lang) = alias.parse::<SupportLang>() {
        return Ok(lang);
    }
    let lower = alias.to_ascii_lowercase();
    match lower.as_str() {
        "sh" | "shell" => Ok(SupportLang::Bash),
        "yml" => Ok(SupportLang::Yaml),
        "c++" => Ok(SupportLang::Cpp),
        "htm" => Ok(SupportLang::Html),
        "csharp" => Ok(SupportLang::CSharp),
        _ => Err(AstError::Language(format!(
            "unsupported language '{alias}'"
        ))),
    }
}

/// Recursively collect candidate files under `dir`, skipping ignored
/// directories, over-large files, and files whose language cannot be inferred.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if IGNORED_DIRS.contains(&name.as_str()) {
            continue;
        }
        let full = entry.path();
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            collect_files(&full, out);
        } else if ft.is_file() {
            let size = match entry.metadata() {
                Ok(m) => m.len(),
                Err(_) => continue,
            };
            if size > MAX_FILE_BYTES {
                continue;
            }
            if infer_lang_by_extension(&full).is_none() {
                continue;
            }
            out.push(full);
        }
    }
}

/// Resolve a search/replace target to a list of files to process.
/// A file target yields a single-element list; a directory target is walked.
fn resolve_files(target: &Path) -> Result<Vec<PathBuf>, AstError> {
    if !target.exists() {
        return Err(AstError::InvalidArgs(format!(
            "Path does not exist: {}",
            target.display()
        )));
    }
    if target.is_file() {
        return Ok(vec![target.to_path_buf()]);
    }
    let mut files = Vec::new();
    collect_files(target, &mut files);
    files.sort();
    Ok(files)
}

/// Run the sandbox containment layers for an optional raw target path string,
/// returning the resolved in-tree path. When `path` is `None` or blank,
/// returns the resolved workspace root directly.
fn resolve_target(root: &Path, path: Option<&str>) -> Result<PathBuf, AstError> {
    let resolved_root = sandbox::resolve_workspace_root(root)?;
    let Some(raw) = path.map(str::trim).filter(|p| !p.is_empty()) else {
        return Ok(resolved_root);
    };
    let normalized = sandbox::normalize_traversal(&resolved_root, raw)?;
    sandbox::refuse_out_of_tree(&resolved_root, &normalized)?;
    sandbox::verify_symlink_containment(&normalized, &resolved_root)?;
    Ok(normalized)
}

/// Search for a syntactic pattern under `root` (or within `path` if specified).
///
/// `lang` is a target language alias from the 27-language `SupportLang` set
/// (e.g. `"ts"`, `"js"`, `"py"`, `"go"`, `"rs"`). Files whose extension maps to
/// a different language are skipped.
pub fn ast_search(
    root: &Path,
    pattern: &str,
    lang: &str,
    path: Option<&str>,
) -> Result<Vec<Match>, AstError> {
    if pattern.trim().is_empty() {
        return Err(AstError::InvalidArgs(
            "AST search requires a non-empty pattern".into(),
        ));
    }
    let target = resolve_target(root, path)?;
    let files = resolve_files(&target)?;

    let target_lang = parse_lang(lang)?;
    let pat =
        Pattern::try_new(pattern, target_lang).map_err(|e| AstError::Pattern(e.to_string()))?;

    let mut matches = Vec::new();
    for file in &files {
        if matches.len() >= MAX_MATCHES {
            break;
        }
        let file_lang = infer_lang_by_extension(file).unwrap_or(target_lang);
        if file_lang != target_lang {
            continue;
        }
        let content = match fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let doc = target_lang.ast_grep(&content);
        for m in doc.root().find_all(&pat) {
            if matches.len() >= MAX_MATCHES {
                break;
            }
            let pos = m.start_pos();
            matches.push(Match {
                file: file.to_string_lossy().to_string(),
                line: pos.line(),
                col: pos.column(m.get_node()),
                text: m.text().to_string(),
            });
        }
    }
    Ok(matches)
}

/// Replace a syntactic pattern under `root` (or within `path` if specified), applying
/// the rewrite per file with a syntax gate (reparse + ERROR-node check) and per-file rollback.
///
/// `lang` is the target language alias. Files whose language does not match
/// `lang` are skipped.
pub fn ast_replace(
    root: &Path,
    pattern: &str,
    replacement: &str,
    lang: &str,
    path: Option<&str>,
) -> Result<ReplaceSummary, AstError> {
    if pattern.trim().is_empty() {
        return Err(AstError::InvalidArgs(
            "AST replace requires a non-empty pattern".into(),
        ));
    }
    let target = resolve_target(root, path)?;
    let files = resolve_files(&target)?;
    let target_lang = parse_lang(lang)?;

    // Validate pattern syntax upfront.
    Pattern::try_new(pattern, target_lang).map_err(|e| AstError::Pattern(e.to_string()))?;

    let mut summary = ReplaceSummary {
        files_scanned: files.len(),
        ..Default::default()
    };

    for file in &files {
        let file_lang = infer_lang_by_extension(file).unwrap_or(target_lang);
        if file_lang != target_lang {
            continue;
        }

        let original = match fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let is_pure_crlf =
            original.contains("\r\n") && !original.replace("\r\n", "").contains('\n');
        let result = replace_in_memory(&original, pattern, replacement, target_lang);
        match result {
            Ok((mut new_content, replacements)) => {
                if is_pure_crlf {
                    new_content = new_content.replace("\r\n", "\n").replace('\n', "\r\n");
                }
                if new_content == original {
                    continue; // no match -> not counted
                }
                // Syntax gate: reparse the new content; any ERROR node -> rollback.
                if syntax_gate(&new_content, target_lang) {
                    // Verified invalid: leave the file unchanged (never written).
                    summary.files_rolled_back += 1;
                    summary.rolled_back.push(file.to_string_lossy().to_string());
                    continue;
                }
                if let Err(e) = fs::write(file, &new_content) {
                    summary.files_rolled_back += 1;
                    summary.rolled_back.push(file.to_string_lossy().to_string());
                    let _ = e;
                    continue;
                }
                summary.files_applied += 1;
                summary.replacements += replacements;
                summary.applied.push(file.to_string_lossy().to_string());
            }
            Err(e) => {
                // Parse/pattern failure on this file -> skip, continue the batch.
                let _ = e;
                continue;
            }
        }
    }

    Ok(summary)
}

/// Compute the rewrite fully in memory. Returns `(new_content, replacements)`.
fn replace_in_memory(
    original: &str,
    pattern: &str,
    replacement: &str,
    lang: SupportLang,
) -> Result<(String, usize), AstError> {
    let doc = lang.ast_grep(original);
    let pat = Pattern::try_new(pattern, lang).map_err(|e| AstError::Pattern(e.to_string()))?;

    // Collect the edits as owned values (they do not borrow the doc), so the
    // matches can be dropped before we mutate the source.
    let edits: Vec<ast_grep_core::source::Edit<String>> = doc
        .root()
        .find_all(pat)
        .map(|m| m.replace_by(replacement))
        .collect();
    if edits.is_empty() {
        return Ok((original.to_string(), 0));
    }

    // Splice the edits in reverse source order so earlier offsets stay valid.
    let mut src = original.to_string();
    for edit in edits.iter().rev() {
        let end = edit.position + edit.deleted_length;
        src.replace_range(
            edit.position..end,
            &String::from_utf8_lossy(&edit.inserted_text),
        );
    }

    Ok((src, edits.len()))
}

/// Produce an HTML-validation copy of `content` in which *bare* ampersands
/// (those not already part of a valid character reference) are escaped to
/// `&amp;`.
///
/// The grammar this exists for is [`SupportLang::Html`]: the tree-sitter HTML
/// scanner treats an `&` not followed by a valid character reference (e.g.
/// `&amp;`, `&#65;`, `&nbsp;`) as malformed and recovers an `ERROR` node. That
/// is a real validation concern, but *bare* ampersands are pervasive and valid
/// in real-world HTML (e.g. the headline "Model Rationale & Intent"), so
/// escaping them here lets the gate accept HTML that a browser would parse
/// fine while still flagging a genuinely broken reference.
///
/// A character reference is `&[name];` (1-30 name chars: letters, digits, `_`)
/// or `&#digits;` / `&#x hex;`. References *without* a closing `;` (e.g.
/// `&amp`) and a trailing, incomplete `&` are NOT treated as valid, so both
/// are escaped. The escape is idempotent: a `&amp;` already in the text stays
/// `&amp;` (its `&` is a valid reference and is left alone), so it never
/// double-escapes to `&amp;amp;`.
pub fn html_validation_text(content: &str) -> String {
    let bytes = content.as_bytes();
    let n = bytes.len();

    // Returns the index just past the valid character reference starting at
    // `i` (`bytes[i] == b'&'`), or `None` when the `&` is not the start of a
    // valid reference (so it must be escaped).
    fn reference_end(bytes: &[u8], i: usize) -> Option<usize> {
        let n = bytes.len();
        if matches!(bytes.get(i + 1), Some(b'#')) {
            // Numeric reference: `&#digits;` or `&#x hex;`.
            let mut k = i + 2;
            let hex = matches!(bytes.get(k), Some(b'x') | Some(b'X'));
            if hex {
                k += 1;
            }
            let start = k;
            while k < n
                && (if hex {
                    bytes[k].is_ascii_hexdigit()
                } else {
                    bytes[k].is_ascii_digit()
                })
            {
                k += 1;
            }
            (k > start && matches!(bytes.get(k), Some(b';'))).then_some(k + 1)
        } else {
            // Named reference: `&[1-30 alphanumeric / '_'];`.
            let mut k = i + 1;
            while k < n && (bytes[k].is_ascii_alphanumeric() || bytes[k] == b'_') {
                k += 1;
            }
            let name_len = k - i - 1;
            ((1..=30).contains(&name_len) && matches!(bytes.get(k), Some(b';'))).then_some(k + 1)
        }
    }

    // `&` is a single ASCII byte (0x26) and can never appear as a continuation
    // byte of a multi-byte UTF-8 sequence, so scanning for it byte-by-byte is
    // safe: every `bytes[i] == b'&'` sits on a char boundary. Everything else is
    // flushed in whole char-boundary slices (`&content[last..]`) so multi-byte
    // sequences (accented, CJK, emoji) are copied intact and never mangled into
    // per-byte Latin-1 mojibake (the old `out.push(bytes[i] as char)` did
    // exactly that, turning e.g. `Café` into `CafÃ©`).
    let mut out = String::with_capacity(content.len());
    let mut i = 0usize;
    let mut last = 0usize;
    while i < n {
        if bytes[i] != b'&' {
            i += 1;
            continue;
        }
        // Flush the run of non-`&` bytes preceding this `&` as a UTF-8 slice.
        if last < i {
            out.push_str(&content[last..i]);
        }
        match reference_end(bytes, i) {
            Some(end) => {
                // A valid character reference: keep it verbatim (idempotent).
                out.push_str(&content[i..end]);
                last = end;
                i = end;
            }
            None => {
                // A bare `&`: escape it, then continue scanning past that byte.
                out.push_str("&amp;");
                last = i + 1;
                i += 1;
            }
        }
    }
    // Trailing run of non-`&` bytes after the last `&`.
    if last < n {
        out.push_str(&content[last..n]);
    }
    out
}

/// Languages whose declaration syntax the outline can index. Prose and
/// markup/config formats (Markdown, JSON, YAML, TOML, HTML, CSS, …) resolve
/// to `None` here and produce no outline.
const OUTLINE_LANGS: &[SupportLang] = &[
    SupportLang::Rust,
    SupportLang::Python,
    SupportLang::JavaScript,
    SupportLang::Tsx,
    SupportLang::TypeScript,
    SupportLang::Go,
    SupportLang::C,
    SupportLang::Cpp,
    SupportLang::Java,
    SupportLang::CSharp,
    SupportLang::Lua,
    SupportLang::Bash,
    SupportLang::Ruby,
    SupportLang::Php,
    SupportLang::Swift,
    SupportLang::Kotlin,
    SupportLang::Scala,
    SupportLang::Dart,
    SupportLang::Solidity,
    SupportLang::Haskell,
    SupportLang::Elixir,
    SupportLang::Nix,
];

/// Declaration node kinds recognized by [`outline_after_line`], spanning the
/// Rust, Python, JS/TS, C-family, and JVM grammars bundled with ast-grep.
pub(crate) const OUTLINE_DECL_KINDS: &[&str] = &[
    "function_item",
    "fn_item",
    "struct_item",
    "enum_item",
    "impl_item",
    "trait_item",
    "mod_item",
    "type_item",
    "function_definition",
    "class_definition",
    "function_declaration",
    "class_declaration",
    "method_definition",
    "method_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "type_declaration",
    "struct_specifier",
    "class_specifier",
];

/// Maximum number of outline items emitted for a single unread tail.
const MAX_OUTLINE_ITEMS: usize = 40;

/// Maximum tree depth (root = 0) the outline DFS descends to. The budget of
/// three edges is what lets the walk reach methods nested one scope below a
/// top-level type: `module → class_definition → block → function_definition`
/// (Python) and `program → class_declaration → class_body → method_definition`
/// (JS/TS, C#, Java) both put the method at depth 3, while a top-level
/// function's own body stays at depth ≥ 3 and is never expanded.
const MAX_OUTLINE_DEPTH: usize = 3;

/// Container declaration kinds the walk descends *into* (beyond just
/// recording them): the enclosing scopes whose bodies hold the methods the
/// outline is meant to surface.
const OUTLINE_CONTAINER_KINDS: &[&str] = &[
    "impl_item",
    "trait_item",
    "class_definition",
    "class_declaration",
    "mod_item",
    "namespace_definition",
];

/// A declaration that carries a body of statements rather than nested
/// declarations (a function or method). The walk records these but does not
/// descend into them, so local/nested function bodies stay out of the
/// outline. Non-declaration wrapper nodes (`source_file`, `module`,
/// `program`, `block`, `class_body`, …) and the container kinds above are
/// always descended into, within [`MAX_OUTLINE_DEPTH`].
fn is_leaf_declaration(kind: &str) -> bool {
    OUTLINE_DECL_KINDS.contains(&kind) && !OUTLINE_CONTAINER_KINDS.contains(&kind)
}

/// Build a symbol outline of the unread tail of `content` (the lines whose
/// 1-based line number is `>= after_line_1based`).
///
/// The path is resolved to a language via [`infer_lang_by_extension`];
/// extensions that are not in [`OUTLINE_LANGS`] (prose and config formats
/// such as Markdown, JSON, YAML, TOML, HTML, CSS) return `None`.
///
/// The tree is parsed with `lang.ast_grep(content)` and walked DFS up to
/// [`MAX_OUTLINE_DEPTH`]. Nodes that are `is_error()`/`is_missing()` or of
/// kind `ERROR`/`UNDEFINED`/`MISSING` are skipped entirely (tree-sitter error
/// recovery still yields the real declarations around a broken region, so
/// they are outlined while the error nodes themselves never appear). Each
/// declaration-kind node whose 1-based start line is `>= after_line_1based`
/// becomes an item: its first source line trimmed, whitespace collapsed,
/// trailing `{`/`:` stripped, and truncated at 120 chars, prefixed by
/// `L{1-based line}:` and indented two spaces per nesting depth.
///
/// At most [`MAX_OUTLINE_ITEMS`] items are emitted; the remainder is
/// summarized with a single `... (N more declarations)` line. Returns
/// `None` when the unread tail holds no declarations.
pub fn outline_after_line(path: &Path, content: &str, after_line_1based: usize) -> Option<String> {
    let lang = match infer_lang_by_extension(path) {
        Some(l) if OUTLINE_LANGS.contains(&l) => l,
        _ => return None,
    };
    let doc = lang.ast_grep(content);
    let mut items: Vec<String> = Vec::new();
    let mut total = 0usize;

    fn first_line_signature(text: &str) -> String {
        let first = text.split('\n').next().unwrap_or("");
        let collapsed: String = first.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut sig = collapsed
            .trim()
            .trim_end_matches(['{', ':'])
            .trim()
            .to_string();
        if sig.is_empty() {
            sig = "<declaration>".to_string();
        }
        if sig.chars().count() > 120 {
            sig = format!("{}…", sig.chars().take(119).collect::<String>().trim_end());
        }
        sig
    }

    fn visit(
        node: &Node<ast_grep_core::tree_sitter::StrDoc<SupportLang>>,
        depth: usize,
        after_line_1based: usize,
        items: &mut Vec<String>,
        total: &mut usize,
    ) {
        if node.is_error()
            || node.is_missing()
            || node.kind() == "ERROR"
            || node.kind() == "UNDEFINED"
            || node.kind() == "MISSING"
        {
            return;
        }
        let kind = node.kind();
        let kind = kind.as_ref();
        if OUTLINE_DECL_KINDS.contains(&kind) {
            let line_1based = node.start_pos().line() + 1;
            if line_1based >= after_line_1based {
                *total += 1;
                if items.len() < MAX_OUTLINE_ITEMS {
                    items.push(format!(
                        "{}L{}: {}",
                        "  ".repeat(depth),
                        line_1based,
                        first_line_signature(node.text().as_ref())
                    ));
                }
            }
            // A function/method body holds statements, not nested
            // declarations of interest, so do not descend into it.
            if is_leaf_declaration(kind) {
                return;
            }
        }
        if depth < MAX_OUTLINE_DEPTH {
            for child in node.children() {
                visit(&child, depth + 1, after_line_1based, items, total);
            }
        }
    }

    visit(&doc.root(), 0, after_line_1based, &mut items, &mut total);
    if total == 0 {
        return None;
    }
    if total > MAX_OUTLINE_ITEMS {
        items.push(format!(
            "  ... ({} more declarations)",
            total - MAX_OUTLINE_ITEMS
        ));
    }
    Some(format!(
        "\n[Outline of unread lines {}.. (index only — read target lines before editing):\n{}\n]",
        after_line_1based,
        items.join("\n")
    ))
}

/// Diagnostic details for a detected syntax error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxErrorDetail {
    pub line: usize,
    pub col: usize,
    pub snippet: String,
}

/// Convert a byte offset in `source` to a 1-indexed (line, column) tuple.
/// Safely handles UTF-8 multi-byte boundaries.
pub fn byte_to_line_col(source: &str, mut byte_offset: usize) -> (usize, usize) {
    byte_offset = byte_offset.min(source.len());
    while byte_offset > 0 && !source.is_char_boundary(byte_offset) {
        byte_offset -= 1;
    }
    let prefix = &source[..byte_offset];
    let line = prefix.matches('\n').count() + 1;
    let col = match prefix.rfind('\n') {
        Some(idx) => byte_offset - idx,
        None => byte_offset + 1,
    };
    (line, col)
}

/// Iteratively walk the AST to collect syntax errors (`ERROR`, `UNDEFINED`, `MISSING` nodes).
/// Does not descend into child nodes of an error node to avoid duplicate cascading reports.
fn collect_syntax_errors<'a>(
    node: &Node<'a, ast_grep_core::tree_sitter::StrDoc<SupportLang>>,
    source: &str,
    errors: &mut Vec<SyntaxErrorDetail>,
    limit: usize,
) {
    if errors.len() >= limit {
        return;
    }
    let mut stack = vec![node.clone()];
    while let Some(current) = stack.pop() {
        if errors.len() >= limit {
            break;
        }
        let kind = current.kind();
        if current.is_error()
            || current.is_missing()
            || kind == "ERROR"
            || kind == "UNDEFINED"
            || kind == "MISSING"
        {
            let range = current.range();
            let (line, col) = byte_to_line_col(source, range.start);
            let line_content = source.lines().nth(line.saturating_sub(1)).unwrap_or("");
            let text = current.text();
            let trimmed_text = text.trim();
            let raw_snippet = if !trimmed_text.is_empty() {
                trimmed_text
            } else if !line_content.trim().is_empty() {
                line_content.trim()
            } else {
                "<end of file>"
            };
            let snippet = if raw_snippet.chars().count() > 60 {
                let s: String = raw_snippet.chars().take(60).collect();
                format!("{s}...")
            } else {
                raw_snippet.to_string()
            };
            errors.push(SyntaxErrorDetail { line, col, snippet });
            if errors.len() >= limit {
                break;
            }
            continue;
        }
        let children: Vec<_> = current.children().collect();
        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }
}

/// Syntax gate: reparse `content` and report whether any ERROR-kind node is
/// present (tree-sitter error recovery). Presence means the content is
/// malformed. Used by [`ast_replace`] for its rollback decision.
///
/// For HTML, the content is validated through [`html_validation_text`], which
/// escapes bare `&` to `&amp;` first, so valid HTML containing an unescaped
/// ampersand is not rejected as malformed.
pub fn syntax_gate(content: &str, lang: SupportLang) -> bool {
    if content.trim().is_empty() {
        return false;
    }
    let source = if lang == SupportLang::Html {
        html_validation_text(content)
    } else {
        content.to_string()
    };
    let doc = lang.ast_grep(&source);
    let mut has_error = false;
    walk_for_errors(&doc.root(), &mut has_error);
    has_error
}

/// Validate that `content` parses cleanly as the language that
/// [`infer_lang_by_extension`] resolves for `path`.
///
/// Returns `Ok(())` when:
/// - The content is empty (e.g. creating blank files), or
/// - The extension does not map to any supported grammar (non-code files like `.txt`/`.md`), or
/// - The content parses without any ERROR/UNDEFINED/MISSING nodes.
///
/// Returns [`AstError::SyntaxFailure`] with descriptive line, column, and snippet
/// diagnostics when the resolved grammar flags the content as malformed.
/// This acts as an invisible barrier preventing `edit_file` and `write_file`
/// from committing syntax errors to disk.
pub fn validate_file_syntax(path: &Path, content: &str) -> Result<(), AstError> {
    if content.trim().is_empty() {
        return Ok(());
    }
    let Some(lang) = infer_lang_by_extension(path) else {
        // Unrecognized extension: no bundled grammar, so there is no syntax to
        // validate. (This also keeps non-code edits like `.txt`/`.md` free.)
        return Ok(());
    };
    let source = if lang == SupportLang::Html {
        html_validation_text(content)
    } else {
        content.to_string()
    };
    let doc = lang.ast_grep(&source);
    let mut errors = Vec::new();
    collect_syntax_errors(&doc.root(), &source, &mut errors, 3);
    if errors.is_empty() {
        return Ok(());
    }

    let mut msg = format!(
        "in-memory {lang} parse of {} found ERROR/UNDEFINED nodes ({} syntax error(s)):",
        path.display(),
        errors.len()
    );
    for err in &errors {
        msg.push_str(&format!(
            "\n  - line {}, col {}: syntax error near `{}`",
            err.line, err.col, err.snippet
        ));
    }
    msg.push_str("\nPlease verify syntax, matching delimiters, and brackets before saving.");
    Err(AstError::SyntaxFailure(msg))
}

/// Count the number of syntax errors (ERROR/UNDEFINED/MISSING nodes) in
/// `content` when parsed in memory as the language that
/// [`infer_lang_by_extension`] resolves for `path`.
///
/// Returns `None` when `content` is empty/whitespace or when the extension does
/// not map to any bundled grammar (there is nothing to parse). Otherwise
/// returns `Some(count)` — the number of distinct error nodes, collected up to
/// a high limit (`256`) so a badly broken file does not cost an unbounded scan
/// while still distinguishing "a few" from "many".
///
/// This is the counting companion to [`validate_file_syntax`]: it lets
/// `edit_file` decide whether an edit *worsens* an already-broken file (more
/// error nodes than the original) instead of failing outright on the
/// original's pre-existing errors.
pub fn count_syntax_errors(path: &Path, content: &str) -> Option<usize> {
    if content.trim().is_empty() {
        return None;
    }
    let lang = infer_lang_by_extension(path)?;
    let source = if lang == SupportLang::Html {
        html_validation_text(content)
    } else {
        content.to_string()
    };
    let doc = lang.ast_grep(&source);
    let mut errors = Vec::new();
    collect_syntax_errors(&doc.root(), &source, &mut errors, 256);
    Some(errors.len())
}

fn walk_for_errors<'a>(
    node: &Node<'a, ast_grep_core::tree_sitter::StrDoc<SupportLang>>,
    out: &mut bool,
) {
    if *out {
        return;
    }
    let mut stack = vec![node.clone()];
    while let Some(current) = stack.pop() {
        let kind = current.kind();
        if current.is_error()
            || current.is_missing()
            || kind == "ERROR"
            || kind == "UNDEFINED"
            || kind == "MISSING"
        {
            *out = true;
            return;
        }
        let children: Vec<_> = current.children().collect();
        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("castor_ast_{}_{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        dunce::canonicalize(&root).unwrap_or(root)
    }

    #[test]
    fn search_finds_pattern_in_fixture() {
        let root = test_root("search");
        let file = root.join("src").join("app.ts");
        fs::write(&file, "function greet(name) {\n  return `hi ${name}`;\n}\n").unwrap();

        let matches = ast_search(&root, "function $NAME($$$ARGS) { $$$BODY }", "ts", None).unwrap();
        assert!(!matches.is_empty(), "expected at least one match");
        let m = &matches[0];
        assert_eq!(m.file, file.to_string_lossy());
        assert_eq!(m.line, 0);
        assert_eq!(m.col, 0);
        assert!(m.text.contains("greet"), "match text: {}", m.text);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn search_finds_pattern_in_rust_fixture() {
        let root = test_root("search_rust");
        let file = root.join("src").join("lib.rs");
        fs::write(
            &file,
            "fn compute_sum(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
        )
        .unwrap();

        let matches =
            ast_search(&root, "fn $NAME($$$ARGS) -> $RET { $$$BODY }", "rs", None).unwrap();
        assert!(!matches.is_empty(), "expected at least one match");
        let m = &matches[0];
        assert_eq!(m.file, file.to_string_lossy());
        assert!(m.text.contains("compute_sum"), "match text: {}", m.text);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn search_skips_unrelated_language_files() {
        let root = test_root("mixed_langs");
        let ts_file = root.join("src").join("app.ts");
        let rs_file = root.join("src").join("lib.rs");
        fs::write(
            &ts_file,
            "function greet(name: string): string {\n  return `hi ${name}`;\n}\n",
        )
        .unwrap();
        fs::write(
            &rs_file,
            "fn compute_sum(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
        )
        .unwrap();

        // Searching Rust pattern must succeed across mixed dir, ignoring the .ts file without error.
        let rs_matches =
            ast_search(&root, "fn $NAME($$$ARGS) -> $RET { $$$BODY }", "rs", None).unwrap();
        assert_eq!(rs_matches.len(), 1);
        assert_eq!(rs_matches[0].file, rs_file.to_string_lossy());

        // Searching TS pattern must succeed across mixed dir, ignoring the .rs file without error.
        let ts_matches = ast_search(
            &root,
            "function $NAME($$$ARGS): $RET { $$$BODY }",
            "ts",
            None,
        )
        .unwrap();
        assert_eq!(ts_matches.len(), 1);
        assert_eq!(ts_matches[0].file, ts_file.to_string_lossy());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn search_resolves_representative_languages() {
        // Prove extension→SupportLang resolution + parsing across a
        // representative spread of the bundled grammars.
        let root = test_root("representative");
        // (relative fixture path, language alias, ast-grep pattern, expected match text, source)
        let cases: &[(&str, &str, &str, &str, &str)] = &[
            (
                "src/greet.go",
                "go",
                "return a + b",
                "a + b",
                "package main\n\nfunc add(a int, b int) int {\n\treturn a + b\n}\n",
            ),
            (
                "src/add.c",
                "c",
                "return a + b;",
                "a + b",
                "int add(int a, int b) {\n    return a + b;\n}\n",
            ),
            (
                "src/add.cpp",
                "cpp",
                "return a + b;",
                "a + b",
                "int add(int a, int b) {\n    return a + b;\n}\n",
            ),
            (
                "src/Program.cs",
                "cs",
                "System.Console.WriteLine($MSG);",
                "hi",
                "class Program {\n    void Main() {\n        System.Console.WriteLine(\"hi\");\n    }\n}\n",
            ),
            (
                "src/App.java",
                "java",
                "System.out.println($MSG);",
                "hi",
                "class App {\n    public void run() {\n        System.out.println(\"hi\");\n    }\n}\n",
            ),
            (
                "src/greet.sh",
                "bash",
                "echo $MSG",
                "hi",
                "#!/bin/sh\ngreet() {\n    echo \"hi\"\n}\n",
            ),
            (
                "src/meta.json",
                "json",
                "{ $$$ }",
                "\"name\"",
                "{\n    \"name\": \"castor\",\n    \"version\": \"1.0.2\"\n}\n",
            ),
            (
                "src/add.py",
                "py",
                "return a + b",
                "a + b",
                "def add(a, b):\n    return a + b\n",
            ),
            (
                "src/add.rs",
                "rs",
                "a + b",
                "a + b",
                "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
            ),
        ];

        for (rel, _, _, _, src) in cases {
            let file = root.join(rel);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(&file, src).unwrap();
        }

        for (rel, lang, pattern, expected_text, _) in cases {
            let expected = root.join(rel);
            let matches = ast_search(&root, pattern, lang, None).unwrap();
            assert!(
                !matches.is_empty(),
                "language '{lang}' ({rel}) should match pattern: {pattern}"
            );
            assert_eq!(
                std::path::Path::new(&matches[0].file),
                expected.as_path(),
                "language '{lang}' resolved to wrong file: {}",
                matches[0].file
            );
            assert!(
                matches[0].text.contains(expected_text),
                "language '{lang}' match text should contain '{expected_text}', got: {}",
                matches[0].text
            );
        }

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn infer_lang_by_extension_maps_representative_extensions() {
        // Direct check of the canonical extension→SupportLang mapping.
        let table: &[(&str, SupportLang)] = &[
            ("a.go", SupportLang::Go),
            ("a.c", SupportLang::C),
            ("a.cpp", SupportLang::Cpp),
            ("a.cs", SupportLang::CSharp),
            ("a.java", SupportLang::Java),
            ("a.sh", SupportLang::Bash),
            ("a.json", SupportLang::Json),
            ("a.py", SupportLang::Python),
            ("a.rs", SupportLang::Rust),
            ("a.ts", SupportLang::TypeScript),
            ("a.tsx", SupportLang::Tsx),
            ("a.md", SupportLang::Markdown),
            ("a.yaml", SupportLang::Yaml),
        ];
        for (name, expected) in table {
            let path = std::path::Path::new(name);
            let lang = infer_lang_by_extension(path)
                .unwrap_or_else(|| panic!("extension '{name}' should resolve"));
            assert_eq!(lang, *expected, "extension '{name}' mis-mapped");
        }

        // Unknown extension stays out of the supported set.
        assert_eq!(
            infer_lang_by_extension(std::path::Path::new("a.xyz_unknown")),
            None
        );
    }

    #[test]
    fn search_targets_specific_file_path() {
        let root = test_root("search_path");
        let file_a = root.join("src").join("a.ts");
        let file_b = root.join("src").join("b.ts");
        fs::write(&file_a, "function foo() { return 1; }\n").unwrap();
        fs::write(&file_b, "function bar() { return 2; }\n").unwrap();

        // Target ONLY a.ts
        let matches = ast_search(
            &root,
            "function $NAME() { $$$BODY }",
            "ts",
            Some("src/a.ts"),
        )
        .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].file, file_a.to_string_lossy());
        assert!(matches[0].text.contains("foo"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn replace_applies_and_reparses_clean() {
        let root = test_root("replace_ok");
        let file = root.join("src").join("app.ts");
        let original = "function greet(name) {\n  return `hi ${name}`;\n}\n";
        fs::write(&file, original).unwrap();

        let summary = ast_replace(
            &root,
            "function $NAME($$$ARGS) { $$$BODY }",
            "function renamed($$$ARGS) { $$$BODY }",
            "ts",
            None,
        )
        .unwrap();

        assert_eq!(summary.files_applied, 1, "expected one applied file");
        assert_eq!(summary.files_rolled_back, 0, "expected no rollbacks");
        assert_eq!(summary.replacements, 1);
        assert_eq!(summary.applied, vec![file.to_string_lossy().to_string()]);

        let new_content = fs::read_to_string(&file).unwrap();
        assert!(new_content.contains("renamed"), "content: {new_content}");
        assert!(!new_content.contains("greet"), "content: {new_content}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn replace_targets_specific_file_path() {
        let root = test_root("replace_path");
        let file_a = root.join("src").join("a.ts");
        let file_b = root.join("src").join("b.ts");
        fs::write(&file_a, "function target() {}\n").unwrap();
        fs::write(&file_b, "function target() {}\n").unwrap();

        // Replace ONLY in a.ts
        let summary = ast_replace(
            &root,
            "function target() {}",
            "function updated() {}",
            "ts",
            Some("src/a.ts"),
        )
        .unwrap();

        assert_eq!(summary.files_applied, 1);
        assert_eq!(summary.applied, vec![file_a.to_string_lossy().to_string()]);
        assert_eq!(
            fs::read_to_string(&file_a).unwrap(),
            "function updated() {}\n"
        );
        assert_eq!(
            fs::read_to_string(&file_b).unwrap(),
            "function target() {}\n"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn invalid_replacement_rolls_back() {
        let root = test_root("replace_bad");
        let file = root.join("src").join("app.ts");
        let original = "function greet(name) {\n  return `hi ${name}`;\n}\n";
        fs::write(&file, original).unwrap();

        // A replacement that produces malformed syntax (a dangling `function`
        // with no body) must be rolled back: the file stays unchanged and the
        // file is named in the rolled-back list.
        let summary = ast_replace(
            &root,
            "function $NAME($$$ARGS) { $$$BODY }",
            "function broken(",
            "ts",
            None,
        )
        .unwrap();

        assert_eq!(
            summary.files_rolled_back, 1,
            "expected one rolled-back file"
        );
        assert_eq!(summary.files_applied, 0);
        assert_eq!(
            summary.rolled_back,
            vec![file.to_string_lossy().to_string()],
            "rolled-back list must name the file"
        );

        let after = fs::read_to_string(&file).unwrap();
        assert_eq!(after, original, "file must be unchanged after rollback");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn out_of_tree_root_refused() {
        // A non-existent absolute root is refused by the sandbox before any
        // AST work happens.
        let missing = std::env::temp_dir().join("castor_ast_missing_never");
        let err = ast_search(&missing, "function $N() {}", "ts", None).unwrap_err();
        assert!(
            matches!(err, AstError::Sandbox(SandboxError::WorkspaceRoot(_))),
            "expected a sandbox refusal, got {err:?}"
        );
    }

    #[test]
    fn validate_file_syntax_accepts_valid_rust() {
        // A syntactically valid Rust string parses cleanly with no ERROR node.
        let path = std::path::Path::new("src/lib.rs");
        let content = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        assert!(validate_file_syntax(path, content).is_ok());
    }

    #[test]
    fn validate_file_syntax_accepts_valid_typescript() {
        let path = std::path::Path::new("src/app.ts");
        let content = "function greet(name: string): string {\n  return `hi ${name}`;\n}\n";
        assert!(validate_file_syntax(path, content).is_ok());
    }

    #[test]
    fn validate_file_syntax_rejects_invalid_rust() {
        // `let x = ;;;` is malformed Rust: the missing initializer produces an
        // ERROR node, so the gate must report a syntax failure.
        let path = std::path::Path::new("src/lib.rs");
        let content = "fn main() {\n    let x = ;;;\n}\n";
        let err = validate_file_syntax(path, content).unwrap_err();
        assert!(
            matches!(err, AstError::SyntaxFailure(_)),
            "expected a SyntaxFailure, got {err:?}"
        );
        assert!(
            err.to_string().contains("SyntaxValidationError"),
            "error message must be labelled SyntaxValidationError, got: {err}"
        );
    }

    #[test]
    fn validate_file_syntax_rejects_invalid_typescript() {
        // A dangling `function` with no body is malformed TS.
        let path = std::path::Path::new("src/app.ts");
        let content = "function broken(\n";
        let err = validate_file_syntax(path, content).unwrap_err();
        assert!(matches!(err, AstError::SyntaxFailure(_)));
    }

    #[test]
    fn validate_file_syntax_ignores_unsupported_extension() {
        // An unrecognized extension has no bundled grammar, so validation is a
        // no-op `Ok` (there is no language to assume) even for "nonsense" text.
        let path = std::path::Path::new("notes.txt");
        let content = "let x = ;;;\n";
        assert!(validate_file_syntax(path, content).is_ok());
    }

    #[test]
    fn count_syntax_errors_counts_rust_errors() {
        // Valid Rust -> zero errors.
        let fixed = "fn a() {\n    let x = 1;\n}\nfn b() {\n    let y = 2;\n}\n";
        assert_eq!(
            count_syntax_errors(std::path::Path::new("lib.rs"), fixed),
            Some(0)
        );
        // One missing initializer -> one error.
        let broken = "fn a() {\n    let x = 1;\n}\nfn b() {\n    let y = ;\n}\n";
        assert_eq!(
            count_syntax_errors(std::path::Path::new("lib.rs"), broken),
            Some(1)
        );
        // Two missing initializers -> two errors (the fix/worsen discriminator
        // used by `edit_file`).
        let worse = "fn a() {\n    let x = ;\n}\nfn b() {\n    let y = ;\n}\n";
        assert_eq!(
            count_syntax_errors(std::path::Path::new("lib.rs"), worse),
            Some(2)
        );
    }

    #[test]
    fn count_syntax_errors_returns_none_for_unsupported_and_empty() {
        // No bundled grammar for `.txt` -> None, regardless of content.
        assert_eq!(
            count_syntax_errors(std::path::Path::new("notes.txt"), "let x = ;;;"),
            None
        );
        // Empty / whitespace content -> None (nothing to parse).
        assert_eq!(
            count_syntax_errors(std::path::Path::new("lib.rs"), ""),
            None
        );
        assert_eq!(
            count_syntax_errors(std::path::Path::new("lib.rs"), "   \n\t"),
            None
        );
    }

    #[test]
    fn infer_lang_by_extension_maps_html() {
        assert_eq!(
            infer_lang_by_extension(std::path::Path::new("a.html")),
            Some(SupportLang::Html)
        );
        assert_eq!(
            infer_lang_by_extension(std::path::Path::new("a.htm")),
            Some(SupportLang::Html)
        );
        assert_eq!(
            infer_lang_by_extension(std::path::Path::new("a.xhtml")),
            Some(SupportLang::Html)
        );
    }

    #[test]
    fn validate_file_syntax_allows_empty_files() {
        assert!(validate_file_syntax(std::path::Path::new("empty.rs"), "").is_ok());
        assert!(validate_file_syntax(std::path::Path::new("empty.py"), "   \n\t").is_ok());
        assert!(validate_file_syntax(std::path::Path::new("empty.json"), "").is_ok());
    }

    #[test]
    fn validate_file_syntax_across_all_mainstream_languages() {
        // Pairs of (file_path, valid_content, invalid_content)
        let cases: &[(&str, &str, &str)] = &[
            (
                "app.rs",
                "fn main() { println!(\"ok\"); }",
                "fn main() { let x = ;;; }",
            ),
            (
                "script.py",
                "def greet(name):\n    return f\"hi {name}\"\n",
                "def broken(x y):\n    return x + y\n",
            ),
            (
                "web.js",
                "function add(a, b) { return a + b; }",
                "function broken( { return; }",
            ),
            (
                "component.jsx",
                "function Comp() { return <div>Hello</div>; }",
                "function Comp() { return <div><span></div>; }",
            ),
            ("module.mjs", "export const x = 42;", "export const x = ;"),
            (
                "legacy.cjs",
                "module.exports = { x: 42 };",
                "module.exports = { x: };",
            ),
            (
                "types.ts",
                "interface User { name: string; age: number; }",
                "interface User { name: string age: number; }",
            ),
            (
                "view.tsx",
                "export const View = (): JSX.Element => <div>Ok</div>;",
                "export const View = (): JSX.Element => <div><span></div>;",
            ),
            (
                "server.go",
                "package main\nfunc main() {}\n",
                "package main\nfunc broken( { return }\n",
            ),
            (
                "native.c",
                "int add(int a, int b) { return a + b; }",
                "int broken( { return 0; }",
            ),
            ("native.h", "int add(int a, int b);", "int broken( ;"),
            (
                "native.cpp",
                "int add(int a, int b) { return a + b; }",
                "int broken( { return 0; }",
            ),
            (
                "native.cc",
                "int add(int a, int b) { return a + b; }",
                "int broken( { return 0; }",
            ),
            (
                "native.hpp",
                "template<typename T> T add(T a, T b) { return a + b; }",
                "template<typename T> T broken( { return 0; }",
            ),
            (
                "App.java",
                "public class App { public static void main(String[] args) {} }",
                "public class App { public static void broken( {} }",
            ),
            (
                "Program.cs",
                "class Program { static void Main() {} }",
                "class Program { static void Broken( {} }",
            ),
            (
                "script.lua",
                "function add(a, b) return a + b end",
                "function broken( return end",
            ),
            (
                "page.html",
                "<!DOCTYPE html><html><body><h1>Title</h1></body></html>",
                "<<<\n",
            ),
            (
                "page.htm",
                "<!DOCTYPE html><html><body><h1>Title</h1></body></html>",
                "<<<\n",
            ),
            (
                "style.css",
                "body { margin: 0; padding: 0; color: #fff; }",
                "body { margin: ;;; }",
            ),
            (
                "data.json",
                "{\"name\": \"castor\", \"valid\": true}",
                "{\"name\": \"castor\", \"broken\": ,}",
            ),
            (
                "config.yaml",
                "server:\n  port: 8080\n  host: 127.0.0.1\n",
                "server: [unclosed list\n",
            ),
            (
                "config.yml",
                "server:\n  port: 8080\n  host: 127.0.0.1\n",
                "server: [unclosed list\n",
            ),
            (
                "run.sh",
                "#!/bin/bash\necho \"hello\"\n",
                "if [ -f file ]; then\n  echo \"no fi\"\n",
            ),
            (
                "run.bash",
                "#!/bin/bash\necho \"hello\"\n",
                "if [ -f file ]; then\n  echo \"no fi\"\n",
            ),
        ];

        for (file_name, valid, invalid) in cases {
            let path = std::path::Path::new(file_name);
            // 1. Valid syntax must pass cleanly
            assert!(
                validate_file_syntax(path, valid).is_ok(),
                "language for '{file_name}' should accept valid syntax:\n{valid}"
            );

            // 2. Invalid syntax must fail and return descriptive diagnostics
            let err = validate_file_syntax(path, invalid).unwrap_err();
            assert!(
                matches!(err, AstError::SyntaxFailure(_)),
                "language for '{file_name}' should return SyntaxFailure for invalid syntax"
            );
            let msg = err.to_string();
            assert!(
                msg.contains("syntax error"),
                "error message for '{file_name}' should mention syntax error, got: {msg}"
            );
            assert!(
                msg.contains("line ") && msg.contains("col "),
                "error message for '{file_name}' should include line and column numbers, got: {msg}"
            );
        }
    }

    #[test]
    fn html_validation_text_escapes_bare_ampersand_only() {
        // A bare `&` (not part of a character reference) becomes `&amp;`.
        assert_eq!(
            html_validation_text("Model Rationale & Intent"),
            "Model Rationale &amp; Intent"
        );
        // Multiple bare `&` are all escaped.
        assert_eq!(html_validation_text("A & B & C"), "A &amp; B &amp; C");
    }

    #[test]
    fn html_validation_text_preserves_valid_references() {
        // Existing character references (named and numeric) are left intact, so
        // the escape is idempotent and never double-escapes a valid reference.
        assert_eq!(html_validation_text("A &amp; B"), "A &amp; B");
        assert_eq!(html_validation_text("C &nbsp; &copy;"), "C &nbsp; &copy;");
        assert_eq!(html_validation_text("D &#65; &#x41;"), "D &#65; &#x41;");
    }

    #[test]
    fn html_validation_text_preserves_multibyte_utf8() {
        // The non-`&` runs are flushed as whole char-boundary slices, so
        // multi-byte UTF-8 (accented, CJK, emoji) is copied intact — the old
        // byte-by-byte `out.push(bytes[i] as char)` mangled each of these into
        // per-byte Latin-1 mojibake (e.g. `Café` → `CafÃ©`). A multi-byte char
        // can never equal the ASCII `&` byte, so the ampersand scan stays
        // boundary-safe and the `&` handling is unchanged.
        //
        // (a) A bare `&` next to multi-byte text: the accented `é` (2 bytes),
        //     the CJK chars (3 bytes each), and the emoji (4 bytes each) all
        //     survive byte-for-byte while the bare `&` is still escaped.
        assert_eq!(
            html_validation_text("Café & 中文 & 测试 & 😀 & 😃"),
            "Café &amp; 中文 &amp; 测试 &amp; 😀 &amp; 😃"
        );
        // (b) A *valid* reference adjacent to multi-byte text: the reference is
        //     kept verbatim and the surrounding non-ASCII bytes stay intact.
        assert_eq!(
            html_validation_text("Café &amp; 😀 &copy; 中文"),
            "Café &amp; 😀 &copy; 中文"
        );
        // (c) Pure multi-byte text with no `&` at all is a no-op identity —
        //     the whole string round-trips exactly (this is the mojibake case).
        assert_eq!(html_validation_text("Café 😀 中文"), "Café 😀 中文");
        assert_eq!(html_validation_text("测试"), "测试");
    }

    #[test]
    fn validate_file_syntax_accepts_valid_html_with_bare_ampersand() {
        // The motivating case: valid HTML with a bare `&` (no character
        // reference) must NOT be rejected as malformed. The gate escapes the
        // bare `&` before parsing, so the HTML grammar sees a valid reference.
        let path = std::path::Path::new("index.html");
        let content = "<!DOCTYPE html>\n<html><head><title>T</title></head>\
                       <body><h1>Model Rationale &amp; Intent</h1>\
                       <p>Model Rationale &amp; Intent</p></body></html>\n";
        assert!(validate_file_syntax(path, content).is_ok());

        // The exact headline from the bug report, with a *bare* ampersand in
        // the source, is what this fix is for.
        let bare = "<!DOCTYPE html>\n<html><body><h1>Model Rationale & Intent</h1></body></html>\n";
        assert!(
            validate_file_syntax(path, bare).is_ok(),
            "a bare & in valid HTML must not be a SyntaxFailure"
        );
    }

    #[test]
    fn validate_file_syntax_still_rejects_invalid_html() {
        // Genuinely malformed HTML must still fail the gate — the bare-`&`
        // escape must not mask a real structural error. `<<<` is a construct the
        // tree-sitter HTML grammar recovers as an ERROR node (verified against
        // the grammar; note that a bare `&` in *text* is accepted, so the
        // escape cannot turn a valid doc into an invalid one or vice versa).
        let path = std::path::Path::new("bad.html");
        let content = "<<<\n";
        let err = validate_file_syntax(path, content).unwrap_err();
        assert!(matches!(err, AstError::SyntaxFailure(_)), "{err}");
    }

    #[test]
    fn outline_after_line_extracts_nested_declarations_ignoring_errors() {
        // Lines 1-10 are a clean `struct` + `impl` block; the region after
        // line 10 contains a broken statement (`let broken = ;;;`) that
        // tree-sitter recovers as an ERROR node. The outline must report the
        // `impl` and its nested method with exact 1-based line numbers, must
        // outline the valid declaration that follows the broken region, and
        // must never emit the ERROR node itself.
        let content = concat!(
            "pub struct Foo {\n",               // 1
            "    pub field: i32,\n",            // 2
            "}\n",                              // 3
            "\n",                               // 4
            "impl Foo {\n",                     // 5
            "    pub fn bar(&self) -> i32 {\n", // 6
            "        self.field\n",             // 7
            "    }\n",                          // 8
            "}\n",                              // 9
            "\n",                               // 10
            "let broken = ;;;\n",               // 11 (broken region → ERROR node)
            "fn inner_after_error() {}\n",      // 12
            "\n",                               // 13
            "pub mod inner {\n",                // 14
            "    pub fn deep() {}\n",           // 15
            "}\n",                              // 16
        );
        let path = Path::new("src/foo.rs");
        let out = outline_after_line(path, content, 1)
            .expect("a Rust file with declarations in the tail must outline");
        assert!(out.contains("[Outline of unread lines 1.."), "{out}");
        // Exact 1-based line numbers for the top-level items, the method
        // nested one scope deeper, and the declaration after the error.
        assert!(out.contains("  L1: pub struct Foo"), "{out}");
        assert!(out.contains("  L5: impl Foo"), "{out}");
        assert!(out.contains("    L6: pub fn bar(&self) -> i32"), "{out}");
        assert!(out.contains("  L12: fn inner_after_error()"), "{out}");
        assert!(out.contains("  L14: pub mod inner"), "{out}");
        assert!(out.contains("    L15: pub fn deep()"), "{out}");
        // The broken statement is an ERROR node: it must not be outlined.
        assert!(!out.contains("L11"), "ERROR node must be skipped: {out}");
        for bad in ["ERROR", "UNDEFINED", "MISSING"] {
            assert!(!out.contains(bad), "outline must skip error nodes: {out}");
        }
    }

    #[test]
    fn outline_after_line_caps_at_max_items_with_remainder_summary() {
        // 100 function declarations after the read window: the outline must
        // stop at 40 items and summarize the remaining 60.
        let content = (0..100)
            .map(|i| format!("fn f_{i:02}() {{ }}\n"))
            .collect::<Vec<_>>()
            .join("");
        let path = Path::new("src/many.rs");
        let out = outline_after_line(path, &content, 1).expect("outline");
        assert_eq!(
            out.lines()
                .filter(|l| l.trim_start().starts_with("L"))
                .count(),
            MAX_OUTLINE_ITEMS,
            "exactly {MAX_OUTLINE_ITEMS} L-items expected: {out}"
        );
        assert!(
            out.contains("... (60 more declarations)"),
            "remaining summary expected: {out}"
        );
        assert!(out.contains("L1: fn f_00()"), "{out}");
        assert!(out.contains("L40: fn f_39()"), "{out}");
    }

    #[test]
    fn outline_after_line_returns_none_for_unsupported_extensions() {
        // Prose and config formats produce no outline, no error.
        let md = (1..=50)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(outline_after_line(Path::new("notes.md"), &md, 1), None);
        assert_eq!(outline_after_line(Path::new("notes.txt"), &md, 1), None);
        let json = (1..=50)
            .map(|i| format!("\"k{i}\": {i},"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(outline_after_line(Path::new("data.json"), &json, 1), None);
        assert_eq!(outline_after_line(Path::new("data.yaml"), &json, 1), None);
        assert_eq!(outline_after_line(Path::new("data.toml"), &json, 1), None);
        assert_eq!(outline_after_line(Path::new("page.html"), &json, 1), None);
        assert_eq!(outline_after_line(Path::new("page.css"), &json, 1), None);
    }

    #[test]
    fn outline_after_line_returns_none_when_tail_has_no_declarations() {
        // The unread tail covers only the body and closing brace of the last
        // function — no declaration starts within it, so there is nothing to
        // outline.
        let content = concat!(
            "fn first() {\n",   // 1
            "    let a = 1;\n", // 2
            "    let b = 2;\n", // 3
            "    a + b\n",      // 4
            "}\n",              // 5
            "fn second() {\n",  // 6
            "    0\n",          // 7
            "}\n",              // 8
        );
        assert_eq!(
            outline_after_line(Path::new("src/tail.rs"), content, 7),
            None
        );
        assert_eq!(
            outline_after_line(Path::new("src/tail.rs"), content, 9),
            None
        );

        // Sanity: starting the tail at the declaration itself outlines it.
        let out = outline_after_line(Path::new("src/tail.rs"), content, 6)
            .expect("a declaration starting in the tail must be outlined");
        assert!(out.contains("L6: fn second()"), "{out}");
    }

    #[test]
    fn outline_after_line_collapses_whitespace_and_strips_opening_brace() {
        // A multi-space-separated signature is collapsed to single spaces and
        // loses its trailing `{`.
        let content = concat!(
            "pub fn   spaced(a:   i32,   b:   i32)   ->   i32  {\n", // 1
            "    a + b\n",                                           // 2
            "}\n",                                                   // 3
            "\n",                                                    // 4
            "fn after() {}\n",                                       // 5
        );
        let out = outline_after_line(Path::new("src/sp.rs"), content, 1).expect("outline");
        assert!(
            out.contains("L1: pub fn spaced(a: i32, b: i32) -> i32"),
            "{out}"
        );
        assert!(out.contains("L5: fn after()"), "{out}");
    }

    #[test]
    fn outline_after_line_truncates_oversized_signatures() {
        let long_param = "a: i32".repeat(40);
        let content = format!("fn long_one({long_param}) {{}}\n");
        assert!(content.len() > 120);
        let out = outline_after_line(Path::new("src/long.rs"), &content, 1).expect("outline");
        let sig_line = out
            .lines()
            .find(|l| l.trim_start().starts_with("L1:"))
            .unwrap();
        // The item is a top-level declaration (depth 1): 2-space indent, then
        // the `L1: ` marker, then the signature.
        assert!(
            sig_line.starts_with("  L1: "),
            "expected depth-1 indent, got: {sig_line}"
        );
        let sig = sig_line.trim_start_matches("  L1: ");
        // The raw signature far exceeds 120 chars, so it must be truncated to
        // at most 120 and flagged with an ellipsis.
        assert!(
            sig.chars().count() <= 120,
            "signature must be truncated to 120 chars (got {}): {sig_line}",
            sig.chars().count()
        );
        assert!(
            sig.ends_with('…'),
            "truncated signature must end with ellipsis: {sig_line}"
        );
    }

    #[test]
    fn search_and_replace_blank_path_use_resolved_root() {
        let root = test_root("blank_path");
        let file = root.join("src").join("app.ts");
        fs::write(&file, "function target() {}\n").unwrap();

        let matches = ast_search(&root, "function target() {}", "ts", Some("   ")).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].file, file.to_string_lossy());

        let summary = ast_replace(
            &root,
            "function target() {}",
            "function renamed() {}",
            "ts",
            Some(""),
        )
        .unwrap();
        assert_eq!(summary.files_applied, 1);
        assert_eq!(
            fs::read_to_string(&file).unwrap(),
            "function renamed() {}\n"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn replace_preserves_pure_crlf_line_endings() {
        let root = test_root("replace_crlf");
        let file = root.join("src").join("app.ts");
        let original = "function greet(name) {\r\n  return name;\r\n}\r\n";
        fs::write(&file, original).unwrap();

        let summary = ast_replace(
            &root,
            "function greet($NAME) { $$$BODY }",
            "function greet($NAME) {\n  console.log($NAME);\n  $$$BODY\n}",
            "ts",
            None,
        )
        .unwrap();
        assert_eq!(summary.files_applied, 1);

        let updated = fs::read_to_string(&file).unwrap();
        assert!(updated.contains("\r\n"), "updated file should use CRLF");
        assert!(
            !updated.replace("\r\n", "").contains('\n'),
            "pure CRLF file must not introduce bare LF: {updated:?}"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn search_caps_at_max_matches_25() {
        let root = test_root("search_cap_25");
        let file = root.join("src").join("many.rs");
        let src = (0..40)
            .map(|i| format!("fn f_{i:02}() {{}}\n"))
            .collect::<Vec<_>>()
            .join("");
        fs::write(&file, src).unwrap();

        let matches = ast_search(&root, "fn $NAME() {}", "rs", None).unwrap();
        assert_eq!(matches.len(), 25, "ast_search must cap matches at 25");

        let _ = fs::remove_dir_all(&root);
    }
}
