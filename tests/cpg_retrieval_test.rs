#![cfg(feature = "ast")]

use castor::runner::skeleton::{
    build_prompt_retrieved_context, build_workspace_skeleton_with_prompt,
};
use castor::tools::graph::CodePropertyGraph;
use std::path::Path;

#[test]
fn test_cpg_multi_language_ingestion_and_call_resolution() {
    let rust_code = r#"
pub struct OrderProcessor {
    id: u64,
}

impl OrderProcessor {
    pub fn process_payment(&self, amount: f64) -> bool {
        verify_funds(amount)
    }
}

pub fn verify_funds(amount: f64) -> bool {
    amount > 0.0
}
"#;

    let python_code = r#"
class PaymentGateway:
    def charge(self, amount: float) -> bool:
        return self.verify_token(amount)

    def verify_token(self, amount: float) -> bool:
        return True
"#;

    let mut graph = CodePropertyGraph::new();
    graph.add_file(Path::new("src/order.rs"), rust_code);
    graph.add_file(Path::new("gateway/payment.py"), python_code);

    assert!(graph.nodes.len() >= 5);

    // Prompt seeking payment processing
    let prompt = "Investigate the process_payment flow and why verify_funds is failing";
    let relevant = graph.retrieve_relevant_symbols(prompt, 3);

    assert!(!relevant.is_empty());
    assert_eq!(relevant[0].name, "process_payment");
    let names: Vec<&str> = relevant.iter().map(|n| n.name.as_str()).collect();
    assert!(names.contains(&"verify_funds") || names.contains(&"OrderProcessor"));
}

#[test]
fn test_cpg_damping_and_rrf_convergence() {
    let mut graph = CodePropertyGraph::new();
    let code = r#"
pub fn root() { branch_a(); branch_b(); }
pub fn branch_a() { leaf_1(); }
pub fn branch_b() { leaf_2(); }
pub fn leaf_1() {}
pub fn leaf_2() {}
pub fn isolated_island() {}
"#;
    graph.add_file(Path::new("src/flow.rs"), code);

    let seeds = graph.extract_prompt_seeds("call branch_a");
    assert!(!seeds.is_empty());

    let ppr = graph.personalized_pagerank(&seeds, 0.85, 20);
    let leaf_1_idx = graph.nodes.iter().position(|n| n.name == "leaf_1").unwrap();
    let island_idx = graph
        .nodes
        .iter()
        .position(|n| n.name == "isolated_island")
        .unwrap();

    assert!(ppr[leaf_1_idx] > ppr[island_idx]);
}

#[test]
fn test_cpg_context_rendering_bounded_token_cap() {
    let tmp = tempfile::tempdir().unwrap();
    let src_dir = tmp.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    std::fs::write(
        src_dir.join("calc.rs"),
        "pub fn calculate_sum(a: i32, b: i32) -> i32 { a + b }\npub fn calculate_product(a: i32, b: i32) -> i32 { a * b }\n",
    ).unwrap();

    let ctx = build_prompt_retrieved_context(tmp.path(), "Need calculate_sum implementation");
    assert!(ctx.contains("[Retrieved Code Context]"));
    assert!(ctx.contains("calculate_sum"));
    assert!(ctx.len() <= 10_000);

    let skel = build_workspace_skeleton_with_prompt(tmp.path(), Some("Need calculate_sum"));
    assert!(skel.contains("[Workspace Skeleton]"));
    assert!(skel.contains("[Retrieved Code Context]"));
}

#[test]
fn test_cpg_explicit_file_path_beats_evals_and_proxy_hubs() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    std::fs::create_dir_all(root.join("evals/tasks/canary/fixture")).unwrap();
    std::fs::create_dir_all(root.join("src/proxy")).unwrap();
    std::fs::create_dir_all(root.join("src/task")).unwrap();
    std::fs::create_dir_all(root.join("src/tools")).unwrap();

    std::fs::write(
        root.join("evals/tasks/canary/fixture/legacy.js"),
        "function process() {}\nfunction a() { process(); }\nfunction b() { process(); }\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/proxy/mod.rs"),
        "pub fn proxy_hub() {}\npub fn call_hub() { proxy_hub(); }\n#[cfg(test)]\nmod tests {\n    fn mock_upstream_handler() { proxy_hub(); }\n}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/task/wait.rs"),
        "pub struct StatusServer {}\npub async fn wait_handler() {}\npub async fn run_wait_cli() {}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/tools/web.rs"),
        "pub struct WebClient {}\npub fn extract_pdf_text() {}\n",
    )
    .unwrap();

    let ctx_wait = build_prompt_retrieved_context(root, "Fix timeout bug in src/task/wait.rs");
    assert!(ctx_wait.contains("src/task/wait.rs#"));
    assert!(!ctx_wait.contains("evals/"));
    assert!(!ctx_wait.contains("mock_upstream_handler"));

    let ctx_web = build_prompt_retrieved_context(root, "Investigate web.rs fetch logic");
    assert!(ctx_web.contains("src/tools/web.rs#"));
    assert!(!ctx_web.contains("evals/"));
}

#[test]
fn test_cpg_repo_explicit_path_and_eval_noise_suppression() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let graph = CodePropertyGraph::build_from_workspace(repo, 120);

    let wait_syms = graph.retrieve_relevant_symbols("Fix timeout handling in src/task/wait.rs", 5);
    assert!(!wait_syms.is_empty());
    assert!(
        wait_syms
            .iter()
            .all(|n| n.file == Path::new("src/task/wait.rs")),
        "expected src/task/wait.rs symbols at top, got: {:?}",
        wait_syms.iter().map(|n| &n.id).collect::<Vec<_>>()
    );

    let web_syms = graph.retrieve_relevant_symbols("Fix PDF extraction in src/tools/web.rs", 5);
    assert!(!web_syms.is_empty());
    assert!(
        web_syms
            .iter()
            .all(|n| n.file == Path::new("src/tools/web.rs")),
        "expected src/tools/web.rs symbols at top, got: {:?}",
        web_syms.iter().map(|n| &n.id).collect::<Vec<_>>()
    );

    let normal_syms = graph.retrieve_relevant_symbols("Investigate request processing", 8);
    assert!(
        !normal_syms
            .iter()
            .any(|n| n.file.to_string_lossy().contains("evals")),
        "evals fixtures must not crowd out src symbols: {:?}",
        normal_syms.iter().map(|n| &n.id).collect::<Vec<_>>()
    );
    assert!(
        !normal_syms
            .iter()
            .any(|n| n.name == "mock_upstream_handler"),
        "#[cfg(test)] mock_upstream_handler must not appear in production retrieval"
    );
}
