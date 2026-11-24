//! Adversarial integration test suite for Castor's resilient code editing
//! and web inspection capabilities.
//!
//! Authored by `@verifier` to challenge:
//! 1. 2-space vs 4-space vs 8-space indentation shifts (relative indentation match).
//! 2. Mixed tab vs space indentation resilience.
//! 3. Trailing whitespace drift on blank and comment lines.
//! 4. Ambiguous multiple candidate occurrences with candidate line number reporting.
//! 5. Invalid syntax injection (must be strictly rejected by in-memory AST gate).
//! 6. `web_fetch` outline inspectability ensuring section queries and anchor hints exist.

use std::fs;
use std::path::PathBuf;

use castor::runner::ToolExecutor;
use castor::tools::fs::FsExecutor;
use castor::tools::web::WebClient;
use serde_json::json;

fn setup_test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("castor_adv_edit_{}_{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Challenge 1: 2-space vs 4-space vs 8-space indentation shifts.
/// The file on disk is deeply nested (8 spaces base indentation).
/// The model emits target and replacement with 2-space base indentation.
/// The resilient edit engine must detect the relative line structure, match the
/// unique block, and apply the replacement adopting the file's 8-space base indent.
#[tokio::test]
async fn test_adversarial_indentation_shift() {
    let dir = setup_test_dir("indent_shift");
    let file_path = dir.join("service.rs");
    let original = r#"pub struct Service {
        pub fn process(&self) {
                let a = 1;
                let b = 2;
                let c = a + b;
                println!("{}", c);
        }
}
"#;
    fs::write(&file_path, original).unwrap();
    let ex = FsExecutor::new(&dir).unwrap();

    // Model provides 2 spaces base indent instead of 16 spaces
    let target = "  let a = 1;\n  let b = 2;\n  let c = a + b;";
    let replacement = "  let a = 10;\n  let b = 20;\n  let c = a * b;";

    let res = ex
        .execute(
            "edit_file",
            &json!({
                "path": "service.rs",
                "target_content": target,
                "replacement_content": replacement,
                "replace_all": false,
            })
            .to_string(),
        )
        .await;

    assert!(
        res.is_ok(),
        "Expected indentation-shifted edit to succeed via relative matching, got error: {:?}",
        res
    );

    let updated = fs::read_to_string(&file_path).unwrap();
    assert!(
        updated.contains("                let a = 10;"),
        "Replacement must adopt the file's 16-space local base indentation. File content:\n{}",
        updated
    );
    assert!(updated.contains("                let b = 20;"));
    assert!(updated.contains("                let c = a * b;"));
}

/// Challenge 2: Mixed tab vs space indentation.
/// The file on disk uses hard tabs (`\t\t`) for indentation.
/// The model emits target and replacement using 8 spaces.
/// The resilient edit engine must match the relative line shape and replace cleanly.
#[tokio::test]
async fn test_adversarial_tab_space_mismatch() {
    let dir = setup_test_dir("tab_space");
    let file_path = dir.join("calculator.rs");
    let original =
        "fn calculate() {\n\t\tlet factor = 42;\n\t\tlet result = factor * 2;\n\t\tresult\n}\n";
    fs::write(&file_path, original).unwrap();
    let ex = FsExecutor::new(&dir).unwrap();

    // Model emits spaces instead of tabs
    let target = "        let factor = 42;\n        let result = factor * 2;";
    let replacement = "        let factor = 50;\n        let result = factor * 3;";

    let res = ex
        .execute(
            "edit_file",
            &json!({
                "path": "calculator.rs",
                "target_content": target,
                "replacement_content": replacement,
                "replace_all": false,
            })
            .to_string(),
        )
        .await;

    assert!(
        res.is_ok(),
        "Expected tab-indented file to match space-indented target, got error: {:?}",
        res
    );

    let updated = fs::read_to_string(&file_path).unwrap();
    assert!(
        updated.contains("let factor = 50;"),
        "Updated file must contain replacement content. File content:\n{}",
        updated
    );
}

/// Challenge 3: Trailing whitespace drift on blank and comment lines.
/// The file on disk has trailing spaces on comment and blank lines.
/// The model emits trimmed lines without trailing spaces.
/// The resilient edit engine must match ignoring trailing whitespace.
#[tokio::test]
async fn test_adversarial_trailing_whitespace_drift() {
    let dir = setup_test_dir("trailing_space");
    let file_path = dir.join("handler.rs");
    // Trailing spaces on comment and blank line
    let original =
        "fn handle() {\n    // verify status    \n    \n    let status = 200;\n    status\n}\n";
    fs::write(&file_path, original).unwrap();
    let ex = FsExecutor::new(&dir).unwrap();

    // Model emits clean trimmed lines
    let target = "// verify status\n\nlet status = 200;";
    let replacement = "// verify status\n\nlet status = 204;";

    let res = ex
        .execute(
            "edit_file",
            &json!({
                "path": "handler.rs",
                "target_content": target,
                "replacement_content": replacement,
                "replace_all": false,
            })
            .to_string(),
        )
        .await;

    assert!(
        res.is_ok(),
        "Expected trailing whitespace drift to be tolerated, got error: {:?}",
        res
    );

    let updated = fs::read_to_string(&file_path).unwrap();
    assert!(
        updated.contains("let status = 204;"),
        "Updated file must contain replacement. File content:\n{}",
        updated
    );
}

/// Challenge 4: Ambiguous multiple candidate occurrences.
/// File on disk contains duplicate blocks in two distinct functions.
/// With replace_all: false, the edit must fail with AmbiguousTargetError
/// AND must report the candidate line numbers to guide the model.
#[tokio::test]
async fn test_adversarial_ambiguous_occurrences_reports_lines() {
    let dir = setup_test_dir("ambiguous");
    let file_path = dir.join("duplicate.rs");
    let original = r#"fn alpha() {
    let flag = true;
    let count = 0;
}

fn beta() {
    let other = 10;
}

fn gamma() {
    let flag = true;
    let count = 0;
}
"#;
    fs::write(&file_path, original).unwrap();
    let ex = FsExecutor::new(&dir).unwrap();

    let target = "    let flag = true;\n    let count = 0;";
    let replacement = "    let flag = false;\n    let count = 1;";

    let res = ex
        .execute(
            "edit_file",
            &json!({
                "path": "duplicate.rs",
                "target_content": target,
                "replacement_content": replacement,
                "replace_all": false,
            })
            .to_string(),
        )
        .await;

    assert!(res.is_err(), "Expected ambiguous edit to fail");
    let err_msg = res.unwrap_err().to_string();
    assert!(
        err_msg.contains("AmbiguousTargetError"),
        "Error message must specify AmbiguousTargetError, got: {}",
        err_msg
    );
    // Must report candidate line numbers (line 2 and line 11)
    assert!(
        err_msg.contains('2') && (err_msg.contains("11") || err_msg.contains("10")),
        "Error message must report exact candidate line numbers to guide the model, got: {}",
        err_msg
    );
}

/// Challenge 5: Invalid syntax injection must be rejected by in-memory AST gate.
/// The candidate replacement introduces a syntax error (dangling plus with unclosed delimiter).
/// The AST gate must reject the write and leave the file byte-identical.
#[cfg(feature = "ast")]
#[tokio::test]
async fn test_adversarial_invalid_syntax_rejected_by_ast_gate() {
    let dir = setup_test_dir("syntax_gate");
    let file_path = dir.join("lib.rs");
    let original = "pub fn compute(x: i32) -> i32 {\n    x * 2\n}\n";
    fs::write(&file_path, original).unwrap();
    let ex = FsExecutor::new(&dir).unwrap();

    let target = "    x * 2";
    let replacement = "    x * 2 + (";

    let res = ex
        .execute(
            "edit_file",
            &json!({
                "path": "lib.rs",
                "target_content": target,
                "replacement_content": replacement,
                "replace_all": false,
            })
            .to_string(),
        )
        .await;

    assert!(res.is_err(), "AST gate must reject syntax error");
    let err_msg = res.unwrap_err().to_string();
    assert!(
        err_msg.contains("SyntaxValidationError"),
        "Error message must specify SyntaxValidationError, got: {}",
        err_msg
    );

    let content = fs::read_to_string(&file_path).unwrap();
    assert_eq!(
        content, original,
        "File must remain byte-identical on syntax gate rejection"
    );
}

/// Challenge 6: `web_fetch` outline inspectability test.
/// When fetching a large HTML page that exceeds the output threshold,
/// the truncated result must provide a structured section index / anchor outline
/// so the agent can discover sub-sections and issue targeted reads.
#[tokio::test]
async fn test_adversarial_web_fetch_large_page_section_outline() {
    use axum::Router;
    use axum::response::Html;
    use axum::routing::get;

    let app = Router::new().route(
        "/large_doc",
        get(|| async {
            let mut html = String::from("<html><body><h1>API Documentation</h1>");
            for i in 1..=50 {
                html.push_str(&format!(
                    "<h2 id=\"section-{i}\">Section {i}: Component Analysis</h2><p>{}</p>",
                    "Detailed specification content lorem ipsum dolor sit amet. ".repeat(30)
                ));
            }
            html.push_str("</body></html>");
            Html(html)
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = WebClient::new();
    let url = format!("http://{}/large_doc", addr);
    let result = client.fetch_docs(&url, None).await.unwrap();

    assert_eq!(result.status, 200);
    // When truncated, the document must include section outline / anchor hints
    assert!(
        result.markdown.contains("[Content truncated")
            || result.markdown.contains("[Section Index]")
            || result.markdown.contains("Section Outline"),
        "Large document must indicate truncation or outline, got: {}",
        &result.markdown[result.markdown.len().saturating_sub(400)..]
    );
    // Truncated large web page must provide a structured section outline / anchor index with hints
    assert!(
        result.markdown.contains("## Document Outline")
            || result.markdown.contains("[Section Index]")
            || result.markdown.contains("### Sections"),
        "Truncated web fetch output must generate a structural section outline with anchor links for discoverability, got: {}",
        &result.markdown[result.markdown.len().saturating_sub(600)..]
    );
}
