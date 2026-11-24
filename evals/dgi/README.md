# Decomposition Granularity Index (DGI) Benchmark & Stress Test

This directory houses the benchmark dataset and evaluation suite for Castor's DGI Gatekeeper (`src/mcp/dgi.rs`).

## Dataset Architecture
- `dataset.json`: **Master benchmark dataset (1,053 prompts)** comprising:
  - **760 sanitized real-world dispatches** mined from open-source engineering workflows and anonymized task sessions (Patchwork M1/M2/M3, website overhaul, exemplars, and bugfixes).
  - **76 historical Claude-Code dispatches** from the primary Castor bootstrap session, including the 7 calibration anchors in `src/mcp/dgi_anchors.json`.
  - **217 curated cross-domain synthetic prompts** spanning 10 technical domains: Research, Coding (Rust, Python, Go, C++, TS), Writing & Documentation, Adversarial Edge Cases, Systems & Kernel, Web Frontend, Cloud & DevOps, Data & ML, Compilers & Languages, and Database & Storage.

## Calibration Doctrine: Retraction of Arbitrary Formulas
- **Retracted Arbitrary Heuristics:**
  - Hard character-length ceilings (`char_len > 2500`): Falsely rejected 348 cohesive, high-specification architectural dispatches containing full test suites and error contexts.
  - Arbitrary soft penalty scoring formulas (`score >= 2`): Lacked cross-domain generality and penalized descriptive prompts.
- **Empirically Grounded Classification (Cohesion-First):**
  - **Semantic Cohesion:** A task is `ADMIT` if all work serves one coherent goal within one subsystem, regardless of length.
  - **Subsystem Decoupling Detection:** Flags dispatches that mandate uncoordinated rewrites across 3+ unrelated subsystems or chain decoupled domains (e.g. remote benchmark scraping + local code refactoring + host hardware sizing).

## Running the Evaluation
```bash
# Evaluate master dataset (1,053 prompts)
python evals/dgi/eval.py

# Or evaluate a specific subset
python evals/dgi/eval.py evals/dgi/dataset_110.json
python evals/dgi/eval.py evals/dgi/dataset_claude_77.json
```
