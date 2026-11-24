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
