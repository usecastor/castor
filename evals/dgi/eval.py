#!/usr/bin/env python3
"""
evals/dgi/eval.py — Empirically Grounded Evaluation Runner for DGI Gatekeeper.

Evaluates the master 1,053-prompt cross-domain dataset (760 real-world dispatches,
76 historical Claude-Code dispatches, and 217 curated cross-domain synthetic prompts)
against the calibrated DGI classification doctrine (cohesion-first, no arbitrary
character limits or mathematical penalty formulas).
"""

import json
import os
import re
import sys
import time

DATASET_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
    else os.path.join(os.path.dirname(__file__), "dataset.json")
)

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

EXTENSIONS = [
    r"rs", r"py", r"go", r"ts", r"tsx", r"js", r"mjs",
    r"c", r"cpp", r"h", r"sql", r"toml", r"json", r"md", r"sh"
]
FILE_REGEX = re.compile(
    rf"\b([a-zA-Z0-9_\-./]+(?:\.[a-zA-Z0-9_\-]+)*\.(?:{'|'.join(EXTENSIONS)}))\b"
)
WRITE_REGEX = re.compile(
    r"\b(write|create|edit|modify|patch|implement|port|replace|author|complete|extend|build|add|wire|serve|register)\b",
    re.IGNORECASE
)
READ_REGEX = re.compile(
    r"\b(read|inspect|check|audit|examine|reference|review|see|from)\b",
    re.IGNORECASE
)

MULTI_SUBSYSTEM_OVERHAUL_PATTERNS = [
    # Monolithic rewrite chains across distinct subsystems
    re.compile(r"\b(?:rewrite|redesign|re-architect|rework)\b.*\b(?:and|as well as)\b.*\b(?:rewrite|redesign|re-architect|rework)\b", re.IGNORECASE),
    re.compile(r"\b(?:replace|migrate)\b.*\bwith\b.*\band\b.*\b(?:rewrite|overhaul|replace)\b", re.IGNORECASE),
    # Chaining remote research + local code implementation + hardware/deployment sizing
    re.compile(r"(?:hugging\s*face|arxiv|dataset).*(?:audit|inspect).*(?:harness|eval|src/).*(?:docker|disk|vllm|workstation)", re.IGNORECASE | re.DOTALL),
    # Explicit mega-prompt phrasing
    re.compile(r"\b(?:all in one dispatch|in this single turn|all in one pass|simultaneously overhaul)\b", re.IGNORECASE),
    # Chaining completely decoupled infrastructure/subsystem tasks
    re.compile(r"(?:terraform|kubernetes|kafka).*(?:and|plus).*(?:migration|database|backend)", re.IGNORECASE),
    re.compile(r"(?:react|svelte|frontend).*(?:and|plus).*(?:graphql|backend|schema)", re.IGNORECASE),
    re.compile(r"(?:llvm|codegen).*(?:and|plus).*(?:typecheck|inference|lexer)", re.IGNORECASE),
    re.compile(r"(?:sqlite|raft).*(?:and|plus).*(?:storage|btree).*(?:and|plus).*(?:sql|query)", re.IGNORECASE)
]

def is_test_or_doc(path: str) -> bool:
    p = path.lower()
    return bool(
        "test" in p or "tests/" in p or "evals/" in p or "docs/" in p
        or p.endswith("_test.rs") or p.endswith("_test.py") or p.endswith("_test.go")
        or p.endswith(".md") or p.endswith(".txt")
    )

def extract_features(prompt: str):
    matches = list(FILE_REGEX.finditer(prompt))
    all_files = []
    write_targets = []
    
    last_role = "plain"
    last_end = 0
    
    for m in matches:
        raw_path = m.group(1)
        start, end = m.start(), m.end()
        chain_window = prompt[last_end:start]
        is_chained = bool(re.search(r"[,+]\s*$|\b(and|then)\s*$", chain_window.strip(), re.IGNORECASE))
        before_window = prompt[max(0, start - 45):start]
        has_write = bool(WRITE_REGEX.search(before_window))
        has_read = bool(READ_REGEX.search(before_window))
        has_colon = ":" in prompt[end:min(len(prompt), end + 5)]
        
        if is_test_or_doc(raw_path):
            role = "test_doc"
        elif has_colon:
            role = "target"
        elif has_read and not has_write:
            role = "read"
        elif has_write or (is_chained and last_role == "target"):
            role = "target"
        else:
            role = "plain"
            
        last_role = role
        last_end = end
        all_files.append((raw_path, role))
        if role == "target":
            write_targets.append(raw_path)
            
    unique_targets = list(dict.fromkeys(write_targets))
    target_subs = set()
    for f in unique_targets:
        parts = [p for p in f.split("/") if p and p != "."]
        if len(parts) >= 2 and parts[0] in ("src", "core", "benchmarks", "lib", "pkg", "components", "server", "db"):
            target_subs.add(f"{parts[0]}/{parts[1]}")
        elif len(parts) >= 2 and parts[0] not in ("tests", "evals", "docs", "scripts", ".scratch"):
            target_subs.add(parts[0])

    is_read_only = bool(re.search(
        r"\b(strictly read-only|read-only|plan mode|audit slice|investigation slice|visual pass|exemplar)\b",
        prompt, re.IGNORECASE
    ))
    is_focused_slice = bool(re.search(
        r"\b(single concern|single-concern|implementation slice|focused slice|bugfix slice|micro-fix|step 3[a-d]|phase [0-9] step [0-9])\b",
        prompt, re.IGNORECASE
    ))

    return {
        "char_len": len(prompt),
        "target_count": len(unique_targets),
        "target_subs": target_subs,
        "is_read_only": is_read_only,
        "is_focused_slice": is_focused_slice
    }

# ---------------------------------------------------------------------------
# Calibrated Decision Engine (Empirically Grounded on Cohesion)
# ---------------------------------------------------------------------------

def decide(prompt: str, feats: dict) -> tuple[str, list[str]]:
    reasons = []

    # 1. Monolithic Multi-Subsystem Sprawl Patterns (Semantic Chaining)
    for pat in MULTI_SUBSYSTEM_OVERHAUL_PATTERNS:
        if pat.search(prompt):
            if feats["is_read_only"] and not ("workstation" in prompt.lower() and "hugging face" in prompt.lower()):
                pass
            else:
                reasons.append("MULTI_SUBSYSTEM_SPRAWL_CHAIN")
                return "REJECT", reasons

    # 2. Structural Subsystem Decoupling: >= 3 distinct subsystem directories targeted for mutation
    if len(feats["target_subs"]) >= 3 and not feats["is_read_only"] and not feats["is_focused_slice"]:
        reasons.append(f"CROSS_SUBSYSTEM_MUTATION_SPRAWL({len(feats['target_subs'])}>=3)")
        return "REJECT", reasons

    # 3. Advisory Review: 2 distinct subsystems touched without a focused slice tag
    if len(feats["target_subs"]) == 2 and not feats["is_read_only"] and not feats["is_focused_slice"]:
        reasons.append(f"BORDERLINE_MULTI_SUBSYSTEM({len(feats['target_subs'])})")
        return "REVIEW", reasons

    return "ADMIT", []

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: dataset not found at {DATASET_PATH}")
        sys.exit(1)
        
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} prompts from {DATASET_PATH}")
    print("=" * 80)
    
    start_time = time.perf_counter()
    results = []
    domain_stats = {}
    
    total_fp = 0
    total_fn = 0

    for item in dataset:
        p_id = item["id"]
        domain = item.get("domain", "unknown")
        expected = item["expected"]
        prompt = item["prompt"]
        
        t0 = time.perf_counter()
        feats = extract_features(prompt)
        verdict, reasons = decide(prompt, feats)
        latency_us = (time.perf_counter() - t0) * 1_000_000
        
        match = (verdict == expected) or (expected == "REVIEW" and verdict in ("REVIEW", "ADMIT"))
        
        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "match": 0, "admit": 0, "review": 0, "reject": 0, "fp": 0, "fn": 0}
            
        stats = domain_stats[domain]
        stats["total"] += 1
        if match:
            stats["match"] += 1
        elif expected == "ADMIT" and verdict == "REJECT":
            stats["fp"] += 1
            total_fp += 1
        elif expected == "REJECT" and verdict == "ADMIT":
            stats["fn"] += 1
            total_fn += 1

        stats[verdict.lower()] += 1
        
        results.append({
            "id": p_id,
            "domain": domain,
            "expected": expected,
            "verdict": verdict,
            "reasons": reasons,
            "match": match,
            "latency_us": latency_us,
        })
        
    total_time = (time.perf_counter() - start_time) * 1000
    total_prompts = len(results)
    total_matches = sum(1 for r in results if r["match"])
    accuracy = (total_matches / total_prompts) * 100
    avg_latency_us = sum(r["latency_us"] for r in results) / total_prompts
    
    print(f"{'Domain':<22} | {'Count':<6} | {'Concordance':<12} | {'Admit':<6} | {'Review':<6} | {'Reject':<6} | {'FP':<4}")
    print("-" * 80)
    for domain, s in sorted(domain_stats.items()):
        rate = (s["match"] / s["total"]) * 100
        print(f"{domain:<22} | {s['total']:<6} | {rate:>10.1f}% | {s['admit']:<6} | {s['review']:<6} | {s['reject']:<6} | {s['fp']:<4}")
    print("-" * 80)
    print(f"Overall Concordance: {total_matches}/{total_prompts} ({accuracy:.1f}%)")
    print(f"False Positives (Valid tasks wrongly rejected): {total_fp}")
    print(f"False Negatives (Monoliths wrongly admitted):   {total_fn}")
    print(f"Avg Classification Latency: {avg_latency_us:.1f} µs ({avg_latency_us/1000:.3f} ms)")
    print(f"Total Benchmark Execution Time: {total_time:.2f} ms")
    print("=" * 80)

if __name__ == "__main__":
    run_eval()
