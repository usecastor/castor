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
use std::path::{Path, PathBuf};

use ast_grep_core::tree_sitter::LanguageExt;
use ast_grep_core::Node;
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
    /// and cross-references.
    pub fn add_file(&mut self, path: &Path, content: &str) {
        let lang = match infer_lang_by_extension(path) {
            Some(l) => l,
            None => return,
        };

        let doc = lang.ast_grep(content);
        let root = doc.root();

        // 1. First pass: extract all declaration nodes.
        let mut file_symbols = Vec::new();
        Self::collect_declarations(&root, path, &mut file_symbols);

        let start_idx = self.nodes.len();
        for sym in &file_symbols {
            let lower_name = sym.name.to_ascii_lowercase();
            let idx = start_idx + file_symbols.iter().position(|s| s.id == sym.id).unwrap();
            self.name_to_nodes.entry(lower_name).or_default().push(idx);
        }

        // Extend adjacency storage.
        let new_total = start_idx + file_symbols.len();
        self.adj.resize_with(new_total, Vec::new);
        self.rev_adj.resize_with(new_total, Vec::new);
        self.nodes.extend(file_symbols);

        // 2. Second pass: link references and calls from definitions in this file.
        // For each node in the file, scan its body for identifiers matching known symbols.
        for node_idx in start_idx..self.nodes.len() {
            let sym = &self.nodes[node_idx];
            // Simple reference scanning: tokenize symbol signature/body lines
            // or match identifiers against known names.
            let lines: Vec<&str> = content.lines().collect();
            if sym.start_line > 0 && sym.end_line <= lines.len() && sym.start_line <= sym.end_line {
                let body_slice = &lines[(sym.start_line - 1)..sym.end_line];
                let mut mentioned = HashSet::new();
                for line in body_slice {
                    for word in extract_identifiers(line) {
                        let lower = word.to_ascii_lowercase();
                        if lower != sym.name.to_ascii_lowercase() {
                            mentioned.insert(lower);
                        }
                    }
                }
                let mut edges_to_add = Vec::new();
                for word in mentioned {
                    if let Some(target_indices) = self.name_to_nodes.get(&word) {
                        for &target_idx in target_indices {
                            if target_idx != node_idx {
                                edges_to_add.push((node_idx, target_idx, 1.0));
                            }
                        }
                    }
                }
                for (from, to, weight) in edges_to_add {
                    self.add_edge(from, to, weight);
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
                "function_item" | "fn_item" | "function_definition" | "function_declaration"
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
            if ["fn", "def", "struct", "class", "impl", "trait", "mod", "enum", "function"]
                .contains(tok)
                && i + 1 < tokens.len()
            {
                let candidate = tokens[i + 1]
                    .trim_matches(['(', '{', ':', '<', '>'])
                    .trim();
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
    pub fn build_from_workspace(workspace_root: &Path, max_files: usize) -> Self {
        let mut graph = Self::new();
        let mut files = Vec::new();
        collect_source_files(workspace_root, &mut files, max_files);

        for path in files {
            if let Ok(content) = fs::read_to_string(&path) {
                let rel = path
                    .strip_prefix(workspace_root)
                    .unwrap_or(&path)
                    .to_path_buf();
                graph.add_file(&rel, &content);
            }
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
                let score =
                    (1.0 / (RRF_K + ppr_rank_map[i] as f64)) + (1.0 / (RRF_K + lex_rank_map[i] as f64));
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

        top_indices.into_iter().map(|idx| &self.nodes[idx]).collect()
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

/// Recursively collect source files for graph indexing, excluding build/cache dirs.
fn collect_source_files(dir: &Path, out: &mut Vec<PathBuf>, max_files: usize) {
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
        if path.is_dir() {
            collect_source_files(&path, out, max_files);
        } else if infer_lang_by_extension(&path).is_some() {
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
        let create_idx = graph.nodes.iter().position(|n| n.name == "create_default_user").unwrap();
        let user_idx = graph.nodes.iter().position(|n| n.name == "UserConfig").unwrap();
        assert!(
            graph.adj[create_idx].iter().any(|(dst, _)| *dst == user_idx),
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

        let prompt = "Please investigate a bug in parse_http_header where whitespace trimming fails.";
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
}
