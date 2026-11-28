#![cfg(feature = "ast")]
//! In-memory Code Property Graph (CPG) & Seeded Personalized PageRank (PPR) retrieval.
//!
//! Architecture:
//! - AST Property Graph: Tree-sitter extraction of definitions (functions, structs, classes,
//!   impls), module structure, calls, and symbol references across workspace source files.
//! - Retrieval & Ranking: Seeded Personalized PageRank (Haveliwala 2003 / Page 1999) combined
//!   with path-aware lexical symbol scoring via Reciprocal Rank Fusion (Cormack 2009 RRF, k=60)
//!   to rank top relevant code symbols for a prompt.
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
        let lang = match infer_lang_by_extension(path).filter(|&l| is_cpg_lang(l)) {
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

        let mut seen_targets: HashSet<usize> = HashSet::new();
        for (node_idx, sym_lower, start_line, end_line) in entries {
            if start_line == 0
                || start_line > end_line
                || start_line > lines_len
                || end_line > lines_len
            {
                continue;
            }
            // Body slice: lines from `start_line` (1-based) up to `end_line`,
            // inclusive of both. Scan identifiers zero-allocation and only track
            // targets that actually exist in `name_to_nodes`.
            let body_slice = &lines[start_line - 1..end_line];
            seen_targets.clear();
            for line in body_slice {
                for_each_lower_identifier(line, |lower| {
                    if lower != sym_lower
                        && let Some(targets) = self.name_to_nodes.get(lower)
                    {
                        for &target_idx in targets {
                            if target_idx != node_idx {
                                seen_targets.insert(target_idx);
                            }
                        }
                    }
                });
            }
            for &target_idx in &seen_targets {
                self.add_edge(node_idx, target_idx, 1.0);
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

    fn collect_declarations<'a>(
        node: &Node<'a, ast_grep_core::tree_sitter::StrDoc<SupportLang>>,
        file_path: &Path,
        out: &mut Vec<SymbolNode>,
    ) {
        let rel_str = file_path.to_string_lossy().replace('\\', "/");
        let mut stack: Vec<Node<'a, ast_grep_core::tree_sitter::StrDoc<SupportLang>>> =
            vec![node.clone()];
        while let Some(current) = stack.pop() {
            if current.is_error()
                || current.is_missing()
                || current.kind() == "ERROR"
                || current.kind() == "UNDEFINED"
                || current.kind() == "MISSING"
            {
                continue;
            }

            let kind_str = current.kind();
            let kind_str = kind_str.as_ref();
            if OUTLINE_DECL_KINDS.contains(&kind_str) {
                let start_line = current.start_pos().line() + 1;
                let end_line = current.end_pos().line() + 1;

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
                let name = Self::extract_symbol_name(&current, kind_str);
                let signature = Self::first_line_signature(current.text().as_ref());
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

                // For leaf declarations (functions/methods), stop descending into statements.
                if kind_str == "function_item"
                    || kind_str == "fn_item"
                    || kind_str == "function_definition"
                    || kind_str == "function_declaration"
                    || kind_str == "method_definition"
                {
                    continue;
                }
            }

            let mut pending_test_attr = false;
            let mut children_to_visit = Vec::new();
            for child in current.children() {
                let ck = child.kind();
                let ck = ck.as_ref();
                if ck == "attribute_item" {
                    if is_test_attribute(child.text().as_ref()) {
                        pending_test_attr = true;
                    }
                    continue;
                }
                if ck == "line_comment" || ck == "block_comment" {
                    continue;
                }
                let skip_test_item = pending_test_attr;
                pending_test_attr = false;
                if skip_test_item {
                    continue;
                }
                children_to_visit.push(child);
            }
            // Push in reverse order so the first kept sibling is popped first (DFS pre-order).
            for child in children_to_visit.into_iter().rev() {
                stack.push(child);
            }
        }
    }

    fn extract_symbol_name(
        node: &Node<ast_grep_core::tree_sitter::StrDoc<SupportLang>>,
        kind: &str,
    ) -> String {
        for child in node.children() {
            let ck = child.kind();
            let ck = ck.as_ref();
            if ck == "identifier"
                || ck == "type_identifier"
                || ck == "property_identifier"
                || ck == "field_identifier"
                || ck == "name"
            {
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
    /// Excludes non-production eval/benchmark/fixture directories by default and
    /// visits production source roots (`src/`, `lib/`, `crates/`, etc.) before
    /// secondary directories.
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
    /// any directory whose canonical path escapes the canonical root, so
    /// traversal can never read files outside the workspace.
    pub fn build_from_workspace(workspace_root: &Path, max_files: usize) -> Self {
        Self::build_with_options(
            workspace_root,
            max_files,
            CollectOptions {
                include_evals: false,
                include_tests: true,
            },
        )
    }

    /// Build a CodePropertyGraph tuned to a specific prompt, skipping
    /// non-production eval/benchmark and test directories unless the prompt
    /// explicitly references them.
    pub fn build_from_workspace_for_prompt(
        workspace_root: &Path,
        max_files: usize,
        prompt: &str,
    ) -> Self {
        let query = PromptQuery::parse(prompt);
        Self::build_with_options(
            workspace_root,
            max_files,
            CollectOptions {
                include_evals: query.wants_evals,
                include_tests: query.wants_tests,
            },
        )
    }

    fn build_with_options(
        workspace_root: &Path,
        max_files: usize,
        options: CollectOptions,
    ) -> Self {
        let mut graph = Self::new();

        // Canonicalize the root once so symlink containment is checked against a
        // fully resolved base; fall back to the raw path if it cannot be resolved.
        let canon_root =
            dunce::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());

        let mut files = Vec::new();
        collect_source_files(&canon_root, &mut files, max_files, &canon_root, options);

        // Pass 1: index every file first so the symbol table is complete.
        let mut pending: Vec<(Range<usize>, String)> = Vec::new();
        for path in files {
            if let Ok(content) = fs::read_to_string(&path) {
                let rel = path
                    .strip_prefix(&canon_root)
                    .unwrap_or(&path)
                    .to_path_buf();
                let range = graph.index_file(&rel, &content);
                if !range.is_empty() {
                    pending.push((range, content));
                }
            }
        }

        // Pass 2: link references across all files using the complete table.
        for (range, content) in pending {
            graph.link_range(range, &content);
        }
        graph
    }

    /// Extract prompt seed nodes and compute initial path-aware lexical weights.
    pub fn extract_prompt_seeds(&self, prompt: &str) -> Vec<(usize, f64)> {
        self.compute_query_matches(prompt).0
    }

    fn compute_query_matches(&self, prompt: &str) -> (Vec<(usize, f64)>, Vec<f64>) {
        let n = self.nodes.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let query = PromptQuery::parse(prompt);
        let mut seeds = Vec::new();
        let mut prod_factors = vec![1.0; n];

        for (idx, node) in self.nodes.iter().enumerate() {
            let rel_path = normalize_rel_path(&node.file);
            let file_name = rel_path.rsplit('/').next().unwrap_or(rel_path.as_str());
            let file_stem = strip_extension(file_name);
            let rel_stem = strip_extension(&rel_path);
            let name_lower = node.name.to_ascii_lowercase();

            // 1. Explicit file path / filename / stem matching.
            let mut explicit_file_match = false;
            let mut file_weight: f64 = 0.0;

            for pt in &query.path_tokens {
                if rel_path == *pt || rel_path.ends_with(&format!("/{pt}")) {
                    file_weight = file_weight.max(24.0);
                    explicit_file_match = true;
                } else if rel_stem == *pt || rel_stem.ends_with(&format!("/{pt}")) {
                    file_weight = file_weight.max(16.0);
                    explicit_file_match = true;
                } else if rel_path.starts_with(&format!("{pt}/")) {
                    file_weight = file_weight.max(4.0);
                }
            }

            for ft in &query.file_tokens {
                if file_name == *ft {
                    file_weight = file_weight.max(16.0);
                    explicit_file_match = true;
                }
            }

            if !explicit_file_match {
                if file_stem.len() >= 4
                    && !is_generic_file_stem(file_stem)
                    && query.tokens.contains(file_stem)
                {
                    file_weight += 4.0;
                }
                let matched_segments = rel_path
                    .split('/')
                    .filter(|seg| {
                        let s = strip_extension(seg);
                        s.len() >= 4 && !is_generic_file_stem(s) && query.tokens.contains(s)
                    })
                    .count();
                if matched_segments >= 2 {
                    file_weight += 4.0;
                }
            }

            // 2. Symbol name & qualified id matching.
            let mut sym_weight = 0.0;
            let id_lower = node.id.to_ascii_lowercase();
            if query.lower.contains(&id_lower) {
                sym_weight += 20.0;
                explicit_file_match = true;
            }
            if query.tokens.contains(&name_lower) {
                if is_generic_symbol(&name_lower) || name_lower.len() < 4 {
                    sym_weight += 1.0;
                } else if name_lower.contains('_') || name_lower.len() >= 7 {
                    sym_weight += 6.0;
                } else {
                    sym_weight += 3.5;
                }
            } else if name_lower.len() >= 6
                && !is_generic_symbol(&name_lower)
                && query.lower.contains(&name_lower)
            {
                sym_weight += 2.5;
            }

            // 3. Signature keyword overlap (capped to avoid common-word inflation).
            let mut sig_weight: f64 = 0.0;
            for_each_lower_identifier(&node.signature, |sig_word| {
                if sig_word.len() >= 4
                    && !is_generic_symbol(sig_word)
                    && query.tokens.contains(sig_word)
                {
                    sig_weight += 0.4;
                }
            });
            sig_weight = sig_weight.min(2.0);

            // 4. Non-production noise down-weighting unless explicitly requested.
            let prod_factor =
                compute_prod_factor(&rel_path, &name_lower, explicit_file_match, &query);
            prod_factors[idx] = prod_factor;

            let total_weight = (file_weight + sym_weight + sig_weight) * prod_factor;
            if total_weight > 0.0 {
                seeds.push((idx, total_weight));
            }
        }
        (seeds, prod_factors)
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
    /// Seeded Personalized PageRank and path-aware lexical scoring via
    /// Reciprocal Rank Fusion (RRF, k=60) with normalized lexical affinity and
    /// non-production noise suppression.
    pub fn retrieve_relevant_symbols(&self, prompt: &str, top_k: usize) -> Vec<&SymbolNode> {
        let n = self.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        let (seeds, prod_factors) = self.compute_query_matches(prompt);
        let ppr_scores = self.personalized_pagerank(&seeds, 0.85, 20);

        // Lexical score array.
        let mut lex_scores = vec![0.0; n];
        for &(idx, w) in &seeds {
            lex_scores[idx] = w;
        }
        let max_lex = lex_scores.iter().copied().fold(0.0_f64, f64::max);

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

        // Compute RRF(i) = ppr_rrf + lex_rrf + normalized lexical boost, scaled
        // by production preference. Nodes with zero lexical match receive no
        // lexical RRF contribution so unseeded PageRank hubs cannot outrank
        // explicit file or symbol matches.
        const RRF_K: f64 = 60.0;
        let mut rrf_scores: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let ppr_rrf = if ppr_scores[i] > 0.0 {
                    1.0 / (RRF_K + ppr_rank_map[i] as f64)
                } else {
                    0.0
                };
                let lex_rrf = if lex_scores[i] > 0.0 {
                    1.0 / (RRF_K + lex_rank_map[i] as f64)
                } else {
                    0.0
                };
                let lex_norm = if max_lex > 0.0 {
                    lex_scores[i] / max_lex
                } else {
                    0.0
                };
                let raw = ppr_rrf + lex_rrf + 0.06 * lex_norm;
                (i, raw * prod_factors[i])
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

#[derive(Debug, Clone, Copy)]
struct CollectOptions {
    include_evals: bool,
    include_tests: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathTier {
    Prod,
    Test,
    EvalOrBench,
}

struct PromptQuery {
    lower: String,
    tokens: HashSet<String>,
    path_tokens: Vec<String>,
    file_tokens: Vec<String>,
    wants_evals: bool,
    wants_tests: bool,
    wants_mocks: bool,
}

impl PromptQuery {
    fn parse(prompt: &str) -> Self {
        let lower = prompt.to_ascii_lowercase();
        let mut tokens = HashSet::new();
        for_each_lower_identifier(prompt, |tok| {
            tokens.insert(tok.to_string());
        });

        let mut path_tokens = Vec::new();
        let mut file_tokens = Vec::new();

        for raw in prompt.split(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | '"' | '\'' | '`' | '<' | '>'
                )
        }) {
            if raw.is_empty() {
                continue;
            }
            let mut normalized = raw.replace('\\', "/").to_ascii_lowercase();
            // Skip URLs so host/path components do not register as file paths.
            if normalized.contains("://") {
                continue;
            }
            // Strip trailing line/fragment suffixes (`src/task/wait.rs:199`, `wait.rs#L10`).
            if let Some((before_hash, after_hash)) = normalized.split_once('#') {
                if !after_hash.is_empty() {
                    for_each_lower_identifier(after_hash, |tok| {
                        tokens.insert(tok.to_string());
                    });
                }
                normalized = before_hash.to_string();
            }
            while let Some((head, tail)) = normalized.rsplit_once(':') {
                if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) {
                    normalized = head.to_string();
                } else {
                    break;
                }
            }
            let cleaned = normalized
                .trim_start_matches("./")
                .trim_matches(|c: char| matches!(c, '/' | '.' | ':' | '?' | '!'));
            if cleaned.is_empty() {
                continue;
            }
            if cleaned.contains('/') {
                if !path_tokens.iter().any(|p| p == cleaned) {
                    path_tokens.push(cleaned.to_string());
                }
            } else if let Some((_, ext)) = cleaned.rsplit_once('.')
                && !ext.is_empty()
                && ext.len() <= 8
                && ext.chars().all(|c| c.is_ascii_alphanumeric())
                && !file_tokens.iter().any(|f| f == cleaned)
            {
                file_tokens.push(cleaned.to_string());
            }
        }

        let wants_evals = tokens.iter().any(|t| {
            matches!(
                t.as_str(),
                "eval" | "evals" | "benchmark" | "benchmarks" | "benches" | "fixture" | "fixtures"
            )
        }) || lower.contains("known-fail")
            || path_tokens.iter().any(|p| {
                p.split('/').any(|seg| {
                    matches!(
                        seg,
                        "evals" | "eval" | "benchmarks" | "benches" | "fixtures" | "fixture"
                    )
                })
            });

        let wants_tests = tokens.iter().any(|t| {
            matches!(
                t.as_str(),
                "test" | "tests" | "testing" | "unittest" | "unittests" | "spec"
            )
        }) || path_tokens
            .iter()
            .any(|p| classify_non_prod_path(p) == PathTier::Test)
            || file_tokens
                .iter()
                .any(|f| classify_non_prod_path(f) == PathTier::Test);

        let wants_mocks = tokens.iter().any(|t| {
            matches!(t.as_str(), "mock" | "mocks" | "mocking" | "fake" | "stub")
                || t.starts_with("mock_")
                || t.starts_with("fake_")
                || t.starts_with("stub_")
        });

        Self {
            lower,
            tokens,
            path_tokens,
            file_tokens,
            wants_evals,
            wants_tests,
            wants_mocks,
        }
    }
}

fn is_cpg_lang(lang: SupportLang) -> bool {
    !matches!(
        lang,
        SupportLang::Html
            | SupportLang::Css
            | SupportLang::Json
            | SupportLang::Yaml
            | SupportLang::Markdown
    )
}

fn is_test_attribute(attr_text: &str) -> bool {
    let compact: String = attr_text.chars().filter(|c| !c.is_whitespace()).collect();
    compact == "#[cfg(test)]"
        || compact == "#[test]"
        || compact == "#[tokio::test]"
        || compact.starts_with("#[tokio::test(")
        || compact == "#[rstest]"
        || compact.starts_with("#[rstest(")
}

#[inline]
fn for_each_lower_identifier<F>(text: &str, mut f: F)
where
    F: FnMut(&str),
{
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut buf = [0u8; 128];
    while i < bytes.len() {
        if bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let len = i - start;
            if len >= 2 {
                if len <= buf.len() {
                    for (dst, &src) in buf[..len].iter_mut().zip(&bytes[start..i]) {
                        *dst = src.to_ascii_lowercase();
                    }
                    let s = std::str::from_utf8(&buf[..len]).expect("ascii identifier");
                    f(s);
                } else {
                    let owned = text[start..i].to_ascii_lowercase();
                    f(&owned);
                }
            }
        } else {
            i += 1;
        }
    }
}

fn normalize_rel_path(path: &Path) -> String {
    let s = path
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    s.trim_start_matches("./").to_string()
}

fn strip_extension(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && !ext.contains('/') => stem,
        _ => name,
    }
}

fn is_generic_symbol(name: &str) -> bool {
    matches!(
        name,
        "new"
            | "default"
            | "from"
            | "into"
            | "clone"
            | "fmt"
            | "drop"
            | "run"
            | "init"
            | "main"
            | "get"
            | "set"
            | "add"
            | "put"
            | "del"
            | "len"
            | "is_empty"
            | "as_str"
            | "process"
            | "handle"
            | "execute"
            | "build"
            | "start"
            | "stop"
            | "test"
            | "tests"
            | "setup"
            | "teardown"
            | "helper"
            | "mock"
            | "self"
            | "true"
            | "false"
            | "none"
            | "some"
            | "result"
            | "option"
            | "string"
            | "error"
            | "async"
            | "pub"
    )
}

fn is_generic_file_stem(stem: &str) -> bool {
    matches!(
        stem,
        "mod"
            | "lib"
            | "main"
            | "test"
            | "tests"
            | "index"
            | "init"
            | "utils"
            | "util"
            | "helpers"
            | "src"
    )
}

fn classify_non_prod_path(rel_path: &str) -> PathTier {
    for seg in rel_path.split('/') {
        if matches!(
            seg,
            "evals"
                | "eval"
                | "benchmarks"
                | "benchmark"
                | "benches"
                | "fixtures"
                | "fixture"
                | "known-fail"
        ) {
            return PathTier::EvalOrBench;
        }
        if matches!(seg, "tests" | "test" | "__tests__") {
            return PathTier::Test;
        }
    }
    let file_name = rel_path.rsplit('/').next().unwrap_or(rel_path);
    if file_name.starts_with("test_")
        || file_name.ends_with("_test.rs")
        || file_name.ends_with("_test.py")
        || file_name.ends_with("_test.go")
        || file_name.ends_with(".test.ts")
        || file_name.ends_with(".test.js")
        || file_name.ends_with(".spec.ts")
        || file_name.ends_with(".spec.js")
    {
        return PathTier::Test;
    }
    PathTier::Prod
}

fn is_mock_symbol(name_lower: &str) -> bool {
    name_lower == "mock"
        || name_lower == "fake"
        || name_lower == "stub"
        || name_lower.starts_with("mock_")
        || name_lower.starts_with("fake_")
        || name_lower.starts_with("stub_")
        || name_lower.starts_with("dummy_")
        || name_lower.contains("_mock")
}

fn compute_prod_factor(
    rel_path: &str,
    name_lower: &str,
    explicit_file_match: bool,
    query: &PromptQuery,
) -> f64 {
    if explicit_file_match {
        return 1.0;
    }
    match classify_non_prod_path(rel_path) {
        PathTier::EvalOrBench if !query.wants_evals => 0.02,
        PathTier::Test if !query.wants_tests => 0.05,
        _ if is_mock_symbol(name_lower) && !query.wants_mocks => 0.05,
        _ => 1.0,
    }
}

fn dir_priority(name: &str) -> u8 {
    match name {
        "src" | "lib" | "crates" | "app" | "pkg" | "packages" => 0,
        "bin" => 1,
        "tests" | "test" | "examples" | "docs" => 3,
        _ => 2,
    }
}

/// Recursively collect source files for graph indexing, excluding build/cache
/// dirs, symlinks, and any directory whose canonical form escapes the
/// canonical workspace root.
///
/// Production source directories (`src/`, `lib/`, `crates/`, etc.) are visited
/// before secondary directories so bounded `max_files` budgets always prioritize
/// production source code.
fn collect_source_files(
    dir: &Path,
    out: &mut Vec<PathBuf>,
    max_files: usize,
    canon_root: &Path,
    options: CollectOptions,
) {
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
    const EVAL_DIRS: &[&str] = &[
        "evals",
        "benchmarks",
        "benches",
        "fixtures",
        "fixture",
        "known-fail",
    ];
    const TEST_DIRS: &[&str] = &["tests", "__tests__"];

    let mut local_files: Vec<PathBuf> = Vec::new();
    let mut subdirs: Vec<(u8, String, PathBuf)> = Vec::new();

    for entry in entries.filter_map(|e| e.ok()) {
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
        let path = entry.path();
        if file_type.is_dir() {
            if !options.include_evals && EVAL_DIRS.contains(&name.as_str()) {
                continue;
            }
            if !options.include_tests && TEST_DIRS.contains(&name.as_str()) {
                continue;
            }
            let Ok(canon_sub) = dunce::canonicalize(&path) else {
                continue;
            };
            if !canon_sub.starts_with(canon_root) {
                continue;
            }
            subdirs.push((dir_priority(&name), name, canon_sub));
        } else if file_type.is_file() {
            if infer_lang_by_extension(&path)
                .filter(|&l| is_cpg_lang(l))
                .is_none()
            {
                continue;
            }
            local_files.push(path);
        }
    }

    local_files.sort();
    for path in local_files {
        out.push(path);
        if out.len() >= max_files {
            return;
        }
    }

    subdirs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    for (_, _, sub_path) in subdirs {
        collect_source_files(&sub_path, out, max_files, canon_root, options);
        if out.len() >= max_files {
            return;
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

    #[test]
    fn cfg_test_items_excluded_while_cfg_not_test_preserved() {
        let source = r#"
pub fn handle_proxy() {}

#[cfg(not(test))]
pub fn prod_only() {}

#[cfg(test)]
pub fn serve_ephemeral() {}

#[cfg(test)]
mod tests {
    pub fn mock_upstream_handler() {}
}
"#;
        let mut graph = CodePropertyGraph::new();
        graph.add_file(Path::new("src/proxy/mod.rs"), source);

        let names: Vec<&str> = graph.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"handle_proxy"));
        assert!(names.contains(&"prod_only"));
        assert!(!names.contains(&"serve_ephemeral"));
        assert!(!names.contains(&"mock_upstream_handler"));
    }

    #[test]
    fn explicit_file_path_ranks_target_file_above_evals_and_hubs() {
        let mut graph = CodePropertyGraph::new();
        graph.add_file(
            Path::new("evals/tasks/canary/fixture/legacy.js"),
            "function process() {}\nfunction caller1() { process(); }\nfunction caller2() { process(); }\nfunction caller3() { process(); }\n",
        );
        graph.add_file(
            Path::new("src/proxy/mod.rs"),
            "pub fn proxy_hub() {}\npub fn call_a() { proxy_hub(); }\npub fn call_b() { proxy_hub(); }\n#[cfg(test)]\nmod tests {\n    fn mock_upstream_handler() { proxy_hub(); }\n}\n",
        );
        graph.add_file(
            Path::new("src/task/mod.rs"),
            "pub fn task_entry() { proxy_hub(); }\n",
        );
        graph.add_file(
            Path::new("src/task/wait.rs"),
            "pub struct StatusServer {}\npub async fn wait_handler() {}\npub async fn run_wait_cli() {}\n",
        );

        let by_full_path =
            graph.retrieve_relevant_symbols("Fix timeout bug in src/task/wait.rs", 3);
        assert!(!by_full_path.is_empty());
        assert!(
            by_full_path
                .iter()
                .all(|n| n.file == Path::new("src/task/wait.rs")),
            "expected all top results from src/task/wait.rs, got: {:?}",
            by_full_path.iter().map(|n| &n.id).collect::<Vec<_>>()
        );

        let by_file_name = graph.retrieve_relevant_symbols("Inspect wait.rs", 3);
        assert!(
            by_file_name
                .iter()
                .all(|n| n.file == Path::new("src/task/wait.rs")),
            "expected all top results from wait.rs, got: {:?}",
            by_file_name.iter().map(|n| &n.id).collect::<Vec<_>>()
        );

        // Path-qualified `src/task/mod.rs` must not match `src/proxy/mod.rs`.
        let by_mod_path = graph.retrieve_relevant_symbols("Update src/task/mod.rs", 1);
        assert_eq!(by_mod_path[0].id, "src/task/mod.rs#task_entry");
    }

    #[test]
    fn evals_and_mock_noise_downweighted_on_normal_prompts() {
        let mut graph = CodePropertyGraph::new();
        graph.add_file(
            Path::new("evals/tasks/canary/fixture/legacy.js"),
            "function process() {}\nfunction caller_a() { process(); }\nfunction caller_b() { process(); }\n",
        );
        graph.add_file(
            Path::new("src/runner/engine.rs"),
            "pub fn process_turn() {}\n",
        );
        graph.add_file(
            Path::new("src/proxy/mock_helper.rs"),
            "pub fn mock_server() { process_turn(); }\n",
        );

        let normal = graph.retrieve_relevant_symbols("Investigate process_turn handling", 3);
        assert!(!normal.is_empty());
        assert_eq!(normal[0].id, "src/runner/engine.rs#process_turn");
        assert!(
            !normal
                .iter()
                .any(|n| n.file.to_string_lossy().contains("evals")),
            "evals noise must not appear for normal prompt: {:?}",
            normal.iter().map(|n| &n.id).collect::<Vec<_>>()
        );

        let explicit_eval =
            graph.retrieve_relevant_symbols("Fix evals/tasks/canary/fixture/legacy.js", 2);
        assert!(!explicit_eval.is_empty());
        assert_eq!(
            explicit_eval[0].file,
            Path::new("evals/tasks/canary/fixture/legacy.js")
        );
    }

    #[test]
    fn markdown_skipped_and_deep_containers_do_not_overflow() {
        assert!(!is_cpg_lang(SupportLang::Markdown));

        let base = tempfile::tempdir().unwrap();
        let root = base.path();
        fs::write(
            root.join("README.md"),
            "# Documentation\n\n```rust\nfn markdown_fake() {}\n```\n",
        )
        .unwrap();
        fs::write(root.join("lib.rs"), "pub fn real_code() {}\n").unwrap();

        let graph = CodePropertyGraph::build_from_workspace(root, 100);
        assert!(graph.nodes.iter().any(|n| n.name == "real_code"));
        assert!(
            !graph
                .nodes
                .iter()
                .any(|n| n.file.extension().and_then(|e| e.to_str()) == Some("md")),
            "Markdown files must not be indexed: {:?}",
            graph.nodes
        );

        let depth = 2500;
        let mut deep_src = String::new();
        for i in 0..depth {
            deep_src.push_str(&format!("mod m{i} {{\n"));
        }
        deep_src.push_str("pub fn deep_leaf() {}\n");
        for _ in 0..depth {
            deep_src.push_str("}\n");
        }

        let mut deep_graph = CodePropertyGraph::new();
        deep_graph.add_file(Path::new("src/deep.rs"), &deep_src);
        assert!(
            deep_graph.nodes.iter().any(|n| n.name == "deep_leaf"),
            "deeply nested function should still be collected without stack overflow"
        );
        assert_eq!(deep_graph.nodes.len(), depth + 1);
    }
}
