#![cfg(feature = "ast")]
//! In-memory Code Property Graph (CPG) & Seeded Personalized PageRank (PPR) retrieval.
//!
//! Inspired by Patchwork M1 (`m1_graph`) and M2 (`m2_retrieval`):
//! - M1: Tree-sitter AST property graph extracting definitions (functions, structs, classes,
//!   impls), module structure, calls, and symbol references across workspace source files.
//! - M2: Seeded Personalized PageRank (Haveliwala 2003 / Page 1999) + BM25F lexical
//!   symbol scoring (Cormack 2009 RRF, k=60) to rank top relevant code symbols for a prompt.
//!
//! When enabled, top-ranked symbols are packed into a bounded context block
//! (<=2,500 tokens / ~10,000 characters) in the Turn-1 workspace skeleton,
//! eliminating blind grepping and preventing partial-fix regressions.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use ast_grep_core::Node;
use ast_grep_core::tree_sitter::LanguageExt;
use ast_grep_language::SupportLang;

use super::ast::{OUTLINE_DECL_KINDS, infer_lang_by_extension};

/// Supported symbol declaration kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Struct,
    Class,
    Enum,
    Trait,
    Impl,
    Module,
    Other,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Struct => "struct",
            Self::Class => "class",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Module => "module",
            Self::Other => "symbol",
        }
    }
}

/// A code symbol node in the property graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolNode {
    pub id: String,
    pub file: PathBuf,
    pub name: String,
    pub kind: SymbolKind,
    pub start_line: usize,
    pub end_line: usize,
    pub signature: String,
}

/// An in-memory Code Property Graph over a workspace.
#[derive(Debug, Clone, Default)]
pub struct CodePropertyGraph {
    pub nodes: Vec<SymbolNode>,
    /// Adjacency list: from_node_index -> Vec<(to_node_index, weight)>.
    pub adj: Vec<Vec<(usize, f64)>>,
    /// Reverse adjacency list: to_node_index -> Vec<(from_node_index, weight)>.
    pub rev_adj: Vec<Vec<(usize, f64)>>,
    /// Lookup from lowercased symbol name to node indices.
    name_to_nodes: HashMap<String, Vec<usize>>,
}

impl CodePropertyGraph {
    /// Create an empty code property graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a single source file to the graph, extracting symbol definitions
    /// and cross-references. Equivalent to indexing the file and immediately
    /// linking its references against the currently-known symbol table; for
    /// single-file callers and unit tests the file is already complete, so
    /// same-file references resolve correctly.
    pub fn add_file(&mut self, path: &Path, content: &str) {
        let range = self.index_file(path, content);
        self.link_range(range, content);
    }

    /// Pass 1 of the two-pass build: parse one source file, append its
    /// declaration nodes to `nodes`, and record them in the
    /// `name_to_nodes` symbol table. Returns the node-index range
    /// `[start_idx..end_idx)` occupied by this file's symbols. Unsupported
    /// (non-code) files yield an empty range and index nothing.
    ///
    /// Keeping indexing and reference-linking separate lets
    /// [`build_from_workspace`] index every file first — so the complete
    /// `name_to_nodes` table is available — and only then link references.
    pub fn index_file(&mut self, path: &Path, content: &str) -> Range<usize> {
        let lang = match infer_lang_by_extension(path) {
            Some(l) => l,
            None => return 0..0,
        };

        let doc = lang.ast_grep(content);
        let root = doc.root();

        let mut file_symbols = Vec::new();
        Self::collect_declarations(&root, path, &mut file_symbols);

        let start_idx = self.nodes.len();
        let new_total = start_idx + file_symbols.len();
        self.adj.resize_with(new_total, Vec::new);
        self.rev_adj.resize_with(new_total, Vec::new);
        self.nodes.extend(file_symbols);

        // Record each symbol in the name table at its absolute index. A
        // positional enumerate (not an `iter().position()` scan) keeps this
        // O(n) per file instead of O(n^2).
        for (offset, sym) in self.nodes[start_idx..].iter().enumerate() {
            let idx = start_idx + offset;
            self.name_to_nodes
                .entry(sym.name.to_ascii_lowercase())
                .or_default()
                .push(idx);
        }

        start_idx..self.nodes.len()
    }

    /// Pass 2 of the two-pass build: link references for the symbols in
    /// `range` against the **complete** `name_to_nodes` symbol table. For each
    /// symbol, its body lines are scanned for identifiers that name other known
    /// symbols, producing directed edges to them (a self-reference — the
    /// symbol's own name — is never an edge).
    ///
    /// `content.lines()` is split exactly once per file (the caller supplies it)
    /// rather than once per symbol, and each symbol's body is located by a
    /// positional offset (`start_idx + offset`) rather than an `iter().position()`
    /// scan.
    ///
    /// In [`build_from_workspace`] this runs *after* every file has been
    /// indexed, so a symbol in file A correctly links to a symbol defined in a
    /// file Z that is enumerated later — the complete symbol table is present.
    /// For a single file (`add_file`) the table holds exactly this file's
    /// symbols, so it links same-file references.
    pub fn link_range(&mut self, range: Range<usize>, content: &str) {
        if range.is_empty() {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        let lines_len = lines.len();

        // Copy the per-symbol fields needed for scanning so the loop can
        // mutably borrow `self` for `add_edge` without holding an immutable
        // borrow on `self.nodes`.
        let entries: Vec<(usize, String, usize, usize)> = self.nodes[range.clone()]
            .iter()
            .enumerate()
            .map(|(offset, n)| {
                (
                    range.start + offset,
                    n.name.to_ascii_lowercase(),
                    n.start_line,
                    n.end_line,
                )
            })
            .collect();

        for (node_idx, sym_lower, start_line, end_line) in entries {
            if start_line == 0
                || start_line > end_line
                || start_line > lines_len
                || end_line > lines_len
            {
                continue;
            }
            // Body slice: lines from `start_line` (1-based) up to `end_line`,
            // inclusive of both.
            let body_slice = &lines[start_line - 1..end_line];
            let mut mentioned = HashSet::new();
            for line in body_slice {
                for word in extract_identifiers(line) {
                    let lower = word.to_ascii_lowercase();
                    if lower != sym_lower {
                        mentioned.insert(lower);
                    }
                }
            }
            for word in mentioned {
                // Copy the target indices out of the map (dropping the
                // immutable borrow) before adding edges, so the loop can
                // mutably borrow `self`.
                let targets: Vec<usize> = self
                    .name_to_nodes
                    .get(&word)
                    .map(|v| v.to_vec())
                    .unwrap_or_default();
                for target_idx in targets {
                    if target_idx != node_idx {
                        self.add_edge(node_idx, target_idx, 1.0);
                    }
                }
            }
        }
    }

    /// Add a directed weighted edge between two nodes.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.nodes.len()
            && to < self.nodes.len()
            && !self.adj[from].iter().any(|(dst, _)| *dst == to)
        {
            self.adj[from].push((to, weight));
            self.rev_adj[to].push((from, weight));
        }
    }

    fn collect_declarations(
        node: &Node<ast_grep_core::tree_sitter::StrDoc<SupportLang>>,
        file_path: &Path,
        out: &mut Vec<SymbolNode>,
    ) {
        if node.is_error()
            || node.is_missing()
            || node.kind() == "ERROR"
            || node.kind() == "UNDEFINED"
            || node.kind() == "MISSING"
        {
            return;
        }

        let kind_str = node.kind();
        let kind_str = kind_str.as_ref();
        if OUTLINE_DECL_KINDS.contains(&kind_str) {
            let start_line = node.start_pos().line() + 1;
            let end_line = node.end_pos().line() + 1;

            let sym_kind = match kind_str {
                "function_item"
                | "fn_item"
                | "function_definition"
                | "function_declaration"
                | "method_definition" => SymbolKind::Function,
                "struct_item" => SymbolKind::Struct,
                "class_definition" | "class_declaration" => SymbolKind::Class,
                "enum_item" => SymbolKind::Enum,
                "trait_item" => SymbolKind::Trait,
                "impl_item" => SymbolKind::Impl,
                "mod_item" => SymbolKind::Module,
                _ => SymbolKind::Other,
            };

            // Extract name from child identifier node or signature line.
            let name = Self::extract_symbol_name(node, kind_str);
            let signature = Self::first_line_signature(node.text().as_ref());

            let rel_str = file_path.to_string_lossy().replace('\\', "/");
            let id = format!("{rel_str}#{name}");

            out.push(SymbolNode {
                id,
                file: file_path.to_path_buf(),
                name,
                kind: sym_kind,
                start_line,
                end_line,
                signature,
            });

            // For leaf declarations (functions/methods), stop recursing into statements.
            if kind_str == "function_item"
                || kind_str == "fn_item"
                || kind_str == "function_definition"
                || kind_str == "function_declaration"
                || kind_str == "method_definition"
            {
                return;
            }
        }

        for child in node.children() {
            Self::collect_declarations(&child, file_path, out);
        }
    }

    fn extract_symbol_name(
        node: &Node<ast_grep_core::tree_sitter::StrDoc<SupportLang>>,
        kind: &str,
    ) -> String {
        for child in node.children() {
            let ck = child.kind();
            let ck = ck.as_ref();
            if ck == "identifier" || ck == "type_identifier" || ck == "name" {
                let text = child.text().trim().to_string();
                if !text.is_empty() {
                    return text;
                }
            }
        }
        // Fallback: parse from signature text.
        let raw = node.text();
        let first_line = raw.lines().next().unwrap_or("").trim();
        let tokens: Vec<&str> = first_line.split_whitespace().collect();
        for (i, tok) in tokens.iter().enumerate() {
            if [
                "fn", "def", "struct", "class", "impl", "trait", "mod", "enum", "function",
            ]
            .contains(tok)
                && i + 1 < tokens.len()
            {
                let candidate = tokens[i + 1].trim_matches(['(', '{', ':', '<', '>']).trim();
                if !candidate.is_empty() {
                    return candidate.to_string();
                }
            }
        }
        format!("{kind}_{}", node.start_pos().line() + 1)
    }

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

    /// Build a CodePropertyGraph by scanning a workspace root directory.
    ///
    /// Two passes over the collected files:
    ///   - **Pass 1** ([`index_file`]) parses every file, appending its symbol
    ///     nodes to `nodes` and populating the `name_to_nodes` symbol table, so
    ///     after the pass the table is *complete*.
    ///   - **Pass 2** ([`link_range`]) links references for every file against
    ///     that complete table — so a symbol in file A correctly links to a
    ///     symbol defined in a file Z that was enumerated later.
    ///
    /// `workspace_root` is canonicalized once (falling back to the raw path if
    /// canonicalization fails); [`collect_source_files`] then skips symlinks and
    /// any entry whose canonical path escapes the canonical root, so traversal
    /// can never read files outside the workspace.
    pub fn build_from_workspace(workspace_root: &Path, max_files: usize) -> Self {
        let mut graph = Self::new();

        // Canonicalize the root once so symlink containment is checked against a
        // fully resolved base; fall back to the raw path if it cannot be resolved.
        let canon_root =
            dunce::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());

        let mut files = Vec::new();
        collect_source_files(&canon_root, &mut files, max_files, &canon_root);

        // Pass 1: index every file first so the symbol table is complete.
        let mut pending: Vec<(PathBuf, Range<usize>, String)> = Vec::new();
        for path in files {
            if let Ok(content) = fs::read_to_string(&path) {
                let rel = path
                    .strip_prefix(&canon_root)
                    .unwrap_or(&path)
                    .to_path_buf();
                let range = graph.index_file(&rel, &content);
                if !range.is_empty() {
                    pending.push((rel, range, content));
                }
            }
        }

        // Pass 2: link references across all files using the complete table.
        for (_rel, range, content) in pending {
            graph.link_range(range, &content);
        }
        graph
    }

    /// Extract prompt seed nodes and compute initial BM25F-style lexical weights.
    pub fn extract_prompt_seeds(&self, prompt: &str) -> Vec<(usize, f64)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let prompt_lower = prompt.to_ascii_lowercase();
        let prompt_tokens: HashSet<String> = extract_identifiers(&prompt_lower)
            .into_iter()
            .map(|s| s.to_ascii_lowercase())
            .collect();

        let mut seeds = Vec::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            let mut weight = 0.0;
            let name_lower = node.name.to_ascii_lowercase();
            let file_str = node.file.to_string_lossy().to_ascii_lowercase();

            // 1. Exact symbol name in prompt (highest confidence).
            if prompt_tokens.contains(&name_lower) {
                weight += 3.0;
            } else if prompt_lower.contains(&name_lower) && name_lower.len() >= 3 {
                weight += 2.0;
            }

            // 2. File path match.
            if let Some(file_name) = node.file.file_name().and_then(|f| f.to_str()) {
                let fname_lower = file_name.to_ascii_lowercase();
                if prompt_lower.contains(&fname_lower) {
                    weight += 2.0;
                }
            } else if prompt_lower.contains(&file_str) {
                weight += 2.0;
            }

            // 3. Signature keyword overlap.
            for sig_word in extract_identifiers(&node.signature.to_ascii_lowercase()) {
                if sig_word.len() >= 4 && prompt_tokens.contains(&sig_word) {
                    weight += 0.5;
                }
            }

            if weight > 0.0 {
                seeds.push((idx, weight));
            }
        }
        seeds
    }

    /// Run Seeded Personalized PageRank (PPR) over the graph.
    ///
    /// Implements power iteration with damping factor alpha (default 0.85)
    /// and teleportation personalized on prompt seed weights.
    pub fn personalized_pagerank(
        &self,
        seeds: &[(usize, f64)],
        damping: f64,
        max_iters: usize,
    ) -> Vec<f64> {
        let n = self.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        // Construct teleportation vector p.
        let mut p = vec![0.0; n];
        let total_seed_weight: f64 = seeds.iter().map(|(_, w)| *w).sum();
        if total_seed_weight > 0.0 {
            for &(idx, w) in seeds {
                p[idx] = w / total_seed_weight;
            }
        } else {
            let uniform = 1.0 / (n as f64);
            p.fill(uniform);
        }

        // Initialize state r.
        let mut r = p.clone();
        let mut r_next = vec![0.0; n];

        // Precompute total out-weights (symmetrizing forward + backward flow).
        let mut out_weights = vec![0.0; n];
        for (u, weight) in out_weights.iter_mut().enumerate() {
            let fwd_sum: f64 = self.adj[u].iter().map(|(_, w)| *w).sum();
            let bwd_sum: f64 = self.rev_adj[u].iter().map(|(_, w)| *w * 0.5).sum();
            *weight = fwd_sum + bwd_sum;
        }

        for _ in 0..max_iters {
            let mut dangling_sum = 0.0;
            r_next.fill(0.0);

            for u in 0..n {
                let r_u = r[u];
                let out_w = out_weights[u];
                if out_w > 0.0 {
                    // Push mass forward to callees/dependencies.
                    for &(v, w) in &self.adj[u] {
                        r_next[v] += damping * r_u * (w / out_w);
                    }
                    // Push partial mass backward to callers/parents.
                    for &(v, w) in &self.rev_adj[u] {
                        r_next[v] += damping * r_u * ((w * 0.5) / out_w);
                    }
                } else {
                    dangling_sum += r_u;
                }
            }

            // Distribute teleportation and dangling mass.
            for i in 0..n {
                r_next[i] += (1.0 - damping) * p[i] + damping * dangling_sum * p[i];
            }

            // Check convergence (L1 norm).
            let diff: f64 = r.iter().zip(&r_next).map(|(a, b)| (a - b).abs()).sum();
            r.copy_from_slice(&r_next);
            if diff < 1e-5 {
                break;
            }
        }

        r
    }

    /// Retrieve the top-k most relevant symbols for a user prompt, combining
    /// Seeded Personalized PageRank and BM25F lexical scoring via Reciprocal
    /// Rank Fusion (RRF, k=60).
    pub fn retrieve_relevant_symbols(&self, prompt: &str, top_k: usize) -> Vec<&SymbolNode> {
        let n = self.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        let seeds = self.extract_prompt_seeds(prompt);
        let ppr_scores = self.personalized_pagerank(&seeds, 0.85, 20);

        // Lexical score array.
        let mut lex_scores = vec![0.0; n];
        for &(idx, w) in &seeds {
            lex_scores[idx] = w;
        }

        // Rank indices by PPR score.
        let mut ppr_ranked: Vec<usize> = (0..n).collect();
        ppr_ranked.sort_by(|&a, &b| {
            ppr_scores[b]
                .partial_cmp(&ppr_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Rank indices by lexical score.
        let mut lex_ranked: Vec<usize> = (0..n).collect();
        lex_ranked.sort_by(|&a, &b| {
            lex_scores[b]
                .partial_cmp(&lex_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Map node_idx -> rank (1-based).
        let mut ppr_rank_map = vec![n; n];
        for (rank, &idx) in ppr_ranked.iter().enumerate() {
            ppr_rank_map[idx] = rank + 1;
        }

        let mut lex_rank_map = vec![n; n];
        for (rank, &idx) in lex_ranked.iter().enumerate() {
            lex_rank_map[idx] = rank + 1;
        }

        // Compute RRF(i) = 1/(60 + rank_PPR) + 1/(60 + rank_Lex).
        const RRF_K: f64 = 60.0;
        let mut rrf_scores: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let score = (1.0 / (RRF_K + ppr_rank_map[i] as f64))
                    + (1.0 / (RRF_K + lex_rank_map[i] as f64));
                (i, score)
            })
            .collect();

        rrf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Filter out zero-relevance symbols if seeds were present.
        let top_indices: Vec<usize> = rrf_scores
            .into_iter()
            .filter(|&(idx, _)| {
                if seeds.is_empty() {
                    true
                } else {
                    ppr_scores[idx] > 0.0 || lex_scores[idx] > 0.0
                }
            })
            .take(top_k)
            .map(|(idx, _)| idx)
            .collect();

        top_indices
            .into_iter()
            .map(|idx| &self.nodes[idx])
            .collect()
    }
}

/// Render top-k retrieved symbols into a bounded, clean markdown context block.
/// Bounded to `max_chars` (default ~10,000 chars <= 2,500 tokens).
pub fn render_graph_context(symbols: &[&SymbolNode], max_chars: usize) -> String {
    if symbols.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "[Retrieved Code Context]\n\
         The following code symbols were identified via Code Property Graph PageRank as relevant to this task:\n",
    );

    for (i, sym) in symbols.iter().enumerate() {
        let entry = format!(
            "- `{}` (L{}-L{}, {}): {}\n",
            sym.id,
            sym.start_line,
            sym.end_line,
            sym.kind.as_str(),
            sym.signature
        );
        if i > 0 && out.len() + entry.len() > max_chars {
            out.push_str("... [remaining retrieved symbols elided to preserve context budget]\n");
            break;
        }
        out.push_str(&entry);
    }
    out
}

/// Extract clean alphanumeric identifier tokens from a string.
fn extract_identifiers(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            current.push(c);
        } else if !current.is_empty() {
            if current.len() >= 2 {
                words.push(current.clone());
            }
            current.clear();
        }
    }
    if current.len() >= 2 {
        words.push(current);
    }
    words
}

/// Recursively collect source files for graph indexing, excluding build/cache
/// dirs, symlinks, and any path whose canonical form escapes the canonical
/// workspace root.
///
/// `canon_root` is the canonicalized workspace root (resolved once by
/// [`build_from_workspace`]). Each directory entry is classified by its raw
/// `file_type()` (lstat semantics): symlinked directories are never descended
/// into, and symlinked files are never indexed. Every remaining file is
/// canonicalized and admitted only when its canonical path stays under
/// `canon_root`, so CPG traversal can never read a file outside the workspace
/// even through a non-symlink path that resolves externally.
fn collect_source_files(dir: &Path, out: &mut Vec<PathBuf>, max_files: usize, canon_root: &Path) {
    if out.len() >= max_files {
        return;
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    const EXCLUDED: &[&str] = &[
        ".git",
        "target",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".scratch",
        "dist",
        "build",
    ];

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if EXCLUDED.contains(&name.as_str()) || name.starts_with('.') {
            continue;
        }
        // lstat file type (does not follow symlinks).
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if file_type.is_symlink() {
            // Never descend into or read through a symlink — it could point
            // outside the workspace root.
            continue;
        }
        if file_type.is_dir() {
            collect_source_files(&path, out, max_files, canon_root);
        } else if file_type.is_file() {
            if infer_lang_by_extension(&path).is_none() {
                continue;
            }
            // Canonicalize and enforce containment under the workspace root.
            let Ok(canon) = dunce::canonicalize(&path) else {
                continue;
            };
            if !canon.starts_with(canon_root) {
                continue;
            }
            out.push(path);
            if out.len() >= max_files {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_add_file_rust() {
        let source = r#"
pub struct UserConfig {
    pub name: String,
}

impl UserConfig {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }
}

pub fn create_default_user() -> UserConfig {
    UserConfig::new("guest")
}
"#;
        let mut graph = CodePropertyGraph::new();
        graph.add_file(Path::new("src/user.rs"), source);

        assert_eq!(graph.nodes.len(), 5);
        let names: Vec<&str> = graph.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"UserConfig"));
        assert!(names.contains(&"new"));
        assert!(names.contains(&"get_name"));
        assert!(names.contains(&"create_default_user"));

        // Prove reference linking: `create_default_user` references `UserConfig` / `new`.
        let create_idx = graph
            .nodes
            .iter()
            .position(|n| n.name == "create_default_user")
            .unwrap();
        let user_idx = graph
            .nodes
            .iter()
            .position(|n| n.name == "UserConfig")
            .unwrap();
        assert!(
            graph.adj[create_idx]
                .iter()
                .any(|(dst, _)| *dst == user_idx),
            "create_default_user must have an edge to UserConfig"
        );
    }

    #[test]
    fn test_seeded_pagerank_ranks_relevant_symbol_highest() {
        let source = r#"
pub fn parse_http_header(raw: &str) -> Option<String> {
    Some(raw.trim().to_string())
}

pub fn handle_connection(stream: &str) {
    let header = parse_http_header(stream);
    println!("{:?}", header);
}

pub fn unrelated_math_function(x: i32) -> i32 {
    x * 42
}
"#;
        let mut graph = CodePropertyGraph::new();
        graph.add_file(Path::new("src/net.rs"), source);

        let prompt =
            "Please investigate a bug in parse_http_header where whitespace trimming fails.";
        let relevant = graph.retrieve_relevant_symbols(prompt, 2);

        assert!(!relevant.is_empty());
        assert_eq!(relevant[0].name, "parse_http_header");
        // Second symbol should be connected caller `handle_connection`.
        if relevant.len() > 1 {
            assert_eq!(relevant[1].name, "handle_connection");
        }
    }

    #[test]
    fn test_render_graph_context_respects_char_limit() {
        let nodes = [
            SymbolNode {
                id: "src/net.rs#parse_http_header".to_string(),
                file: PathBuf::from("src/net.rs"),
                name: "parse_http_header".to_string(),
                kind: SymbolKind::Function,
                start_line: 2,
                end_line: 4,
                signature: "pub fn parse_http_header(raw: &str) -> Option<String>".to_string(),
            },
            SymbolNode {
                id: "src/net.rs#handle_connection".to_string(),
                file: PathBuf::from("src/net.rs"),
                name: "handle_connection".to_string(),
                kind: SymbolKind::Function,
                start_line: 6,
                end_line: 9,
                signature: "pub fn handle_connection(stream: &str)".to_string(),
            },
        ];
        let refs: Vec<&SymbolNode> = nodes.iter().collect();
        let rendered = render_graph_context(&refs, 200);
        assert!(rendered.contains("[Retrieved Code Context]"));
        assert!(rendered.contains("parse_http_header"));
        assert!(rendered.len() <= 500);
    }

    /// Pass 1 (`index_file`) must complete for every file before Pass 2
    /// (`link_range`) runs, so a symbol in file A links to a symbol defined in a
    /// file Z that was *added after* A. Here A is indexed first and Z second —
    /// the case single-pass linking (index-and-link per file) would miss — yet
    /// the A -> Z edge is still created when A's range is linked in Pass 2
    /// against the complete symbol table.
    #[test]
    fn forward_cross_file_edge_created_in_two_pass() {
        let a_src = "pub fn run() {\n    zfn();\n}\n";
        let z_src = "pub fn zfn() {\n    0\n}\n";

        // Pass 1: index A first, then Z (Z added after A).
        let mut graph = CodePropertyGraph::new();
        let a_range = graph.index_file(Path::new("src/a.rs"), a_src);
        let z_range = graph.index_file(Path::new("src/z.rs"), z_src);

        // Pass 2: link both ranges against the now-complete symbol table.
        graph.link_range(a_range, a_src);
        graph.link_range(z_range, z_src);

        let run_idx = graph
            .nodes
            .iter()
            .position(|n| n.name == "run")
            .expect("run() indexed");
        let zfn_idx = graph
            .nodes
            .iter()
            .position(|n| n.name == "zfn")
            .expect("zfn() indexed");
        assert!(
            graph.adj[run_idx].iter().any(|(dst, _)| *dst == zfn_idx),
            "run() in file A must link to zfn() defined in the later-added file Z"
        );
    }

    /// Platform symlink creation for the containment tests (unix: a single
    /// `symlink` covers both file and directory links; windows splits them).
    #[cfg(unix)]
    fn symlink(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(windows)]
    fn symlink(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
        // File targets need symlink_file; the dir-symlink test passes a
        // directory target and is gated separately below.
        if target.is_dir() {
            std::os::windows::fs::symlink_dir(target, link)
        } else {
            std::os::windows::fs::symlink_file(target, link)
        }
    }

    /// A symlink inside the workspace pointing to a file outside the workspace
    /// root must be ignored by `build_from_workspace` — the external symbol is
    /// never indexed, even though a real in-workspace file references it.
    #[test]
    fn symlink_file_outside_workspace_ignored() {
        let base = tempfile::tempdir().unwrap();
        let root = base.path().join("ws");
        let outside = base.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();

        fs::write(
            root.join("src.rs"),
            "pub fn real() {\n    external_fn();\n}\n",
        )
        .unwrap();
        fs::write(outside.join("ext.rs"), "pub fn external_fn() {}\n").unwrap();

        // A symlinked file inside `root` pointing at the external file.
        symlink(&outside.join("ext.rs"), &root.join("link_file.rs")).unwrap();

        let graph = CodePropertyGraph::build_from_workspace(&root, 1000);

        assert!(
            graph.nodes.iter().any(|n| n.name == "real"),
            "real in-workspace symbol must be indexed"
        );
        assert!(
            !graph.nodes.iter().any(|n| n.name == "external_fn"),
            "external symbol behind a file symlink must NOT be indexed: {:?}",
            graph.nodes
        );
        let _ = base;
    }

    /// A symlinked *directory* inside the workspace pointing outside must not be
    /// descended into, so source files it contains are never indexed.
    #[test]
    fn symlink_dir_outside_workspace_not_descended() {
        let base = tempfile::tempdir().unwrap();
        let root = base.path().join("ws");
        let outside = base.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();

        fs::write(
            root.join("src.rs"),
            "pub fn real() {\n    external_fn();\n}\n",
        )
        .unwrap();
        fs::write(outside.join("ext.rs"), "pub fn external_fn() {}\n").unwrap();

        // A symlinked directory inside `root` pointing at the external dir.
        symlink(&outside, &root.join("link_dir")).unwrap();

        let graph = CodePropertyGraph::build_from_workspace(&root, 1000);

        assert!(
            graph.nodes.iter().any(|n| n.name == "real"),
            "real in-workspace symbol must be indexed"
        );
        assert!(
            !graph.nodes.iter().any(|n| n.name == "external_fn"),
            "external symbol behind a directory symlink must NOT be indexed: {:?}",
            graph.nodes
        );
        let _ = base;
    }

    /// Measure `build_from_workspace` latency on this repository (files indexed,
    /// nodes, edges). Emitted on stderr only (hidden by the default harness,
    /// shown under `--nocapture`) so the normal gate stays quiet.
    #[test]
    fn build_from_workspace_repo_timing() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let start = std::time::Instant::now();
        let graph = CodePropertyGraph::build_from_workspace(repo, 100_000);
        let elapsed_ms = start.elapsed().as_millis();

        let files: std::collections::HashSet<&std::path::Path> =
            graph.nodes.iter().map(|n| n.file.as_path()).collect();
        let edges: usize = graph.adj.iter().map(|v| v.len()).sum();

        eprintln!(
            "build_from_workspace timing: {elapsed_ms} ms, {} files (with symbols), {} nodes, {} edges",
            files.len(),
            graph.nodes.len(),
            edges
        );

        assert!(
            !graph.nodes.is_empty(),
            "the repository must yield at least one indexed symbol"
        );
    }
}
