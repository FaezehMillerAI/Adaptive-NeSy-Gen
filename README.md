# Adaptive NeSy-Gen

**Adaptive Claim-Level Neuro-Symbolic Verification for Chest X-Ray Report Generation**

AAAI 2026 — Official Implementation

---

## Overview

Adaptive NeSy-Gen generates chest X-ray reports using:
- A **training-free MedGemma backend** (zero-shot or retrieval-conditioned few-shot)
- **Frozen MedSigLIP** visual retrieval for evidence grounding
- **PrimeKG** biomedical knowledge-graph verification
- **LTN-style fuzzy verification** (biological / diagnostic / location clauses)
- A **Consistency Gate** (ACCEPT / REVISE / FLAG / ABSTAIN)
- **Faithful per-claim explanation traces** — procedural records of inference evidence

The main contribution is **adaptive, claim-level neuro-symbolic verification**: expensive graph reasoning is invoked only for uncertain linked claims, not for every sentence in every report.

> **Claim boundary:** The framework does not claim to eliminate hallucinations. Entity unsupported-content rates and clinical-label disagreement serve as proxies, validated with official CheXbert/RadGraph metrics.

---

## Quick Start

### Google Colab (recommended)

Open the main notebook:

```
notebooks/Adaptive_NeSyGen_AAAI_Complete.ipynb
```

The notebook walks through all 10 pipeline stages, evaluation, ablation studies, and explanation visualisation. Run it on a T4 or A100 GPU runtime.

**Prerequisites:**
1. Accept HuggingFace terms for [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) and [google/medsiglip-448](https://huggingface.co/google/medsiglip-448)
2. Add your `HF_TOKEN` to Colab Secrets

### Local installation

```bash
pip install -e ".[torch,eval,colab]"
pip install accelerate
```

---

## Method

### Pipeline Stages

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Visual Retrieval | MedSigLIP embeds query and training images; top-k by cosine similarity |
| 2 | Report Drafting | MedGemma generates Findings section (zero-shot or few-shot) |
| 3 | Claim Extraction | Draft split into clinical claims; entities linked to PrimeKG |
| 4 | Evidence Contract | Per-claim: s_visual, s_retrieval, n_support, s_KG, s_LTN, s_gate |
| 5 | Adaptive Routing | Fast path (τ_fast=0.85, m≥2) vs. graph escalation |
| 6 | PrimeKG Subgraph | Compact radiology subgraph from training-split seed entities |
| 7 | LTN Verification | τ_bio, τ_diag, τ_loc → τ(c_j) = mean clause satisfaction |
| 8 | Consistency Gate | ACCEPT / REVISE / FLAG / ABSTAIN with decision reason |
| 9 | Selective Revision | Entity-matched, polarity-preserving extractive replacement |
| 10 | Explanation Trace | Procedural per-claim evidence record saved to JSONL |

### Adaptive Routing

```
s_ground(c_j) = max(s_visual, s_retrieval)

Fast path:  s_ground(c_j) ≥ τ_fast  AND  n_support(c_j) ≥ m
            → ACCEPT without graph reasoning

Escalation: linked claims below threshold
            → PrimeKG subgraph + LTN + Consistency Gate
```

Default: τ_fast = 0.85, m = 2, τ_revise = 0.50

### Evidence-Bound Revision

A replacement c′_j is eligible only if:
```
Entities(c′_j) = Entities(c_j)   [same PrimeKG IDs]
Polarity(c′_j) = Polarity(c_j)   [negation preserved]
s_retrieval(c′_j) ≥ τ_revise     [threshold met]
```
If no eligible replacement exists, the original claim is preserved and marked FLAGGED.

---

## Datasets

### IU X-Ray (open access, recommended for smoke runs)
Downloaded automatically via Kaggle. No additional credentials needed.

### MIMIC-CXR
Requires PhysioNet credentialing. Set `RUN_DATASET='mimic_aug'` in the notebook.

> **MIMIC-CXR note:** MedGemma and MedSigLIP pretraining likely includes MIMIC-CXR. Experiments on MIMIC-CXR are described as *no task-specific fine-tuning*, **not** strict unseen-data zero-shot.

---

## Evaluation

### Lexical quality
BLEU-1/2/3/4, ROUGE-L, METEOR, CIDEr

### Clinical quality
- Official CheXbert F1 (prepare inputs with `scripts/evaluate_generation.py --prepare-official-inputs-dir`)
- Official RadGraph F1
- Entity precision / recall / F1 (PrimeKG-linked)
- Negation consistency

### Explainability
- Linked-claim coverage
- Entity-specific grounding coverage
- Adaptive escalation rate (all claims; linked claims only)
- Graph-path coverage among escalated claims
- Gate-decision distribution (ACCEPT / REVISE / FLAG / ABSTAIN)
- Revision rate
- Claim and end-to-end latency

### Integrity
- Exact-match leakage audit
- High-overlap (ROUGE-L ≥ 0.95) audit
- Same-study retrieval exclusion verification
- Unique-prediction ratio

---

## Ablation Studies

| # | Configuration | Key switch |
|---|---------------|------------|
| 1 | Retrieval only | `--ablation retrieval_only` |
| 2 | MedGemma zero-shot, no verification | `--draft-mode zero_shot --ablation no_verify` |
| 3 | MedGemma few-shot RAG, no verification | `--draft-mode few_shot --ablation no_verify` |
| 4 | RAG without PrimeKG/LTN | `--ablation no_graph` |
| 5 | Report-level PrimeKG/LTN | `--ablation report_level` |
| 6 | Adaptive audit, no revision | `--revision-policy audit_only` |
| 7 | **Full Adaptive NeSy-Gen** | *(default)* |
| 8 | Always-on verification | `--fast-accept-threshold 0.0` |
| 9 | No LTN (connectivity only) | `--ablation no_ltn` |
| 10 | No Consistency Gate | `--ablation no_gate` |
| 11 | Shuffled PrimeKG relations | `--ablation shuffled_kg` |
| 12 | τ_fast = 0.70 | `--fast-accept-threshold 0.70` |
| 13 | τ_fast = 0.95 | `--fast-accept-threshold 0.95` |

---

## Repository Structure

```
nesy_gen/
├── agents/
│   └── adaptive_verification.py   # Stages 3–10: claim extraction, routing, gate, revision, traces
├── baselines/
│   └── medsiglip_retrieval.py     # Stage 1: frozen visual retrieval with index cache
├── data/
│   └── schema.py                  # RadiologyExample dataclass, JSONL loader
├── evaluation/
│   ├── explainability.py          # Explainability metrics from trace JSONL
│   ├── generation_metrics.py      # BLEU, ROUGE-L, METEOR, CIDEr
│   └── clinical_metrics.py        # Entity F1, negation consistency
├── generation/
│   └── rag.py                     # RagCandidate dataclass
├── kg/
│   ├── entity_linking.py          # LexicalEntityLinker → PrimeKG IDs
│   └── primekg.py                 # PrimeKGGraph loader/wrapper
├── logic/
│   ├── ltn.py                     # NeuroSymbolicAuditor (τ_bio, τ_diag, τ_loc)
│   └── constraints.py             # ClauseScores, fuzzy clause computation
├── models/
│   ├── medgemma.py                # MedGemmaDrafter (Stage 2)
│   ├── gate.py                    # ConsistencyGate (Stage 8)
│   ├── nesy_gen.py                # NesyGenPipeline (composable)
│   └── pipeline_factory.py        # build_primekg_pipeline()
scripts/
├── generate_medgemma_adaptive_reports.py   # Full pipeline (Stages 1–10)
├── run_ablation_suite.py                   # All ablation configurations
├── evaluate_generation.py                  # Lexical + entity metrics
├── evaluate_adaptive_explanations.py       # Explainability metrics
├── audit_prediction_leakage.py             # Integrity audit
├── build_adaptive_explanation_report.py    # HTML explanation report
└── build_radiology_primekg.py              # PrimeKG radiology cache
notebooks/
└── Adaptive_NeSyGen_AAAI_Complete.ipynb   # Main experiment notebook
```

---

## Assumptions & Limitations

1. **Entity linker accuracy** must be evaluated separately (mention precision, linking accuracy, negation accuracy, coverage, manual error analysis). Linker errors propagate to all downstream stages.

2. **LTN scores** (τ_bio, τ_diag, τ_loc) measure satisfaction of implemented graph constraints — not the probability that a clinical statement is true.

3. **Explanation traces** are procedural records of evidence consumed during inference. They are not post-hoc LLM rationalizations and should not be claimed as complete causal explanations.

4. **PrimeKG coverage**: Some radiology entities/relations are absent. Unlinked claims are ABSTAIN; escalation rates should be reported both over all claims and over linked claims only.

5. **No task-specific fine-tuning** for MedGemma/MedSigLIP. This is distinct from strict zero-shot given likely MIMIC-CXR overlap in pretraining.

---

## Citation

```bibtex
@inproceedings{adaptive_nesy_gen_aaai2026,
  title     = {Adaptive Claim-Level Neuro-Symbolic Verification for Chest X-Ray Report Generation},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```
