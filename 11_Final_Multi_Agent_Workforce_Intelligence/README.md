# Final Project — Multi-Agent Workforce Intelligence System for the Energy Industry

**Course:** ITAI 2376 — Deep Learning Artificial Intelligence
**Author:** Jiri Musil
**Title:** Multi-Agent Workforce Intelligence System for the Energy Industry
**Submitted:** May 4, 2026
**Option:** B — Multi-Agent System

## Executive Summary

This is the capstone of my ITAI 2376 work. The system takes resumes, job descriptions, and twelve months of HRIS time-series data for a department, and produces a **boardroom-ready memo** that combines a hiring decision, a retention plan for the receiving team, and a backfill pipeline for the role being filled. Every number in the memo references a verified deep learning model output (not a synthesized approximation), every verdict cites the rule that produced it (R1 *strong stable* through R20 *decline critical*), and every memo includes a full execution trace so a senior reviewer can see which tools were called and what they returned.

The target user is a **People Strategy Lead, an HR Business Partner, or a CHRO** at a large energy company. The system was designed to address the specific gap that Accenture flagged at CERAWeek 2026: only 8% of energy organizations have meaningful skills visibility, while 85% of executives believe AI will transform their business and only 18% see tangible benefits. This system targets that visibility gap directly — it knows what skills a candidate has and what skills a team is about to lose, and it produces a hiring recommendation that integrates both.

## Boardroom Dashboard

![Workforce Intelligence Dashboard — Streamlit + Plotly](dashboard.png)

The Streamlit + Plotly dashboard is the operator-facing surface of the system. Headline KPIs (total headcount, active divisions, employee sentiment score, monthly attrition) sit at the top, followed by the live system architecture view that shows both agents (Talent Intelligence with the routed NER ensemble + SBERT matching, and Workforce Forecasting with the Bi-LSTM attrition model) communicating through the proprietary brain. The bottom panels show twelve months of headcount trajectory by department alongside the current workforce composition, so a CHRO can see *what is happening now* and *what is about to happen* on a single screen. The example shown is the Energy Industry profile (eight departments, 200 total headcount, 3.7/5 average satisfaction, 3.1% monthly attrition rate); the dashboard supports industry switching and a learning-engine toggle for adaptive runs.

## What the System Does

- **Resume + job description analysis.** Routed NER ensemble extracts skills, tools, certifications, degrees, employers, years of experience, industry, location, projects, and soft skills. Sentence-BERT semantic match plus Hungarian assignment produces a 19-signal skill gap analysis that names the specific missing skills and certifications instead of a generic match score.
- **HRIS time-series forecasting.** Bidirectional LSTM with attention forecasts attrition probability and headcount changes over 6–12 months. The output is a 19-signal workforce risk analysis that segments cohorts, links predictions to drivers, simulates retention strategies, and quantifies replacement-cost exposure.
- **Five end-to-end workflows.** Joint hire analysis, workforce risk scan, quarterly retention plan, candidate ranking, candidate-to-multiple-jobs triage.
- **Triple-procedure consensus** for high-stakes decisions. A sophisticated 20-cell decision matrix, a conservative 20-cell matrix, and a naive two-signal heuristic all run in parallel. Disagreements trigger automatic escalation: 3-of-3 agreement → strong consensus, 2-of-3 → senior reviewer, 0-of-3 → hiring committee.
- **Counterfactual flip explainer** on every DEFER and DECLINE verdict. Lists the minimal single-lever adjustments needed to flip the cell into a positive value, graded high/medium/low feasibility based on causal status (causal, mixed, correlational) and lead time.
- **Human-in-the-Lead memory layer** (`feedback.py`) provides a small, controlled, traceable, reversible adaptation channel. Six event types (verdict correction, rule override, confidence calibration, causal update, new intervention, general comment) with configurable thresholds. High-impact events require explicit approval rather than threshold voting.

## Architecture

```
INPUT  →  DL MODELS  →  AGENT BRAIN  →  21 TOOLS  →  OUTPUT (memo + trace)
                              ↑
                              ├── Talent Intelligence Agent
                              └── Workforce Forecasting Agent
```

### Deep learning models

| Model | Agent | Course Module | Role |
|-------|-------|---------------|------|
| DistilBERT (fine-tuned) | 1 | Module 05 — Transformers | Resume NER on the legacy 5-label classes |
| GLiNER (zero-shot, NAACL 2024) | 1 | Module 05 — Transformers | Generalist NER coverage across all 10 labels |
| ModernBERT v11 | 1 | Module 05 — Transformers | NER specialist that owns the TOOL class |
| Sentence-BERT (`all-MiniLM-L6-v2`) | 1 | Module 05 — Transformers | Cosine semantic match + Hungarian assignment for the skill gap diff |
| Bidirectional LSTM with attention | 2 | Module 03 — RNN and LSTM | Department-level attrition probability + headcount forecast |
| Routed per-class ensemble | 1 | Module 05 — Transformers | Per-class hard ownership across the four NER variants |
| Custom proprietary brain | both | Module 10 — Agentic AI | Plan-Act-Observe-Respond loop over 21 tools, no LLM dependency |

The four NER models cooperate via a **per-class router**. Each label is owned by the model that performs best on a held-out evaluation. Legacy 5 labels (SKILL, CERT, DEGREE, EMPLOYER, YEARS_EXP) → DistilBERT v6 ensemble. TOOL → ModernBERT v11. Gate 2 labels (INDUSTRY, LOCATION, PROJECT, SOFT_SKILL) → GLiNER 10-type. This is not a monolithic ensemble dressed up — it is explicit hard ownership per label class, backed by per-class F1 against an adjudicated dev set.

## The Pivot from the Midterm Blueprint

The midterm blueprint committed to **CrewAI + a hosted LLM as the agent's brain**. That path was built (it lives in `agents.py` and remains active when an LLM API key is present), but the **primary deliverable is a deterministic, LLM-free orchestrator written from scratch in `brain.py`**.

### Why I made this pivot

Workforce planning decisions are audited, regulated, and contestable. Every "do not advance" recommendation can be challenged on grounds of bias, and the EU AI Act classifies HR systems as high-risk for a reason. An LLM brain produces fluent narratives that cannot explain in plain language which rule produced a specific verdict. A 20-cell decision matrix where every cell contains a named rule (R1 through R20) and an explicit rationale is much more defensible.

CrewAI remains wired as the optional path for cases where free-text generation actually adds value — resume coaching, drafting job descriptions, multi-turn interview prep. For the actual hiring and retention decisions this system addresses, the LLM adds latency and cost without adding capability.

| Path | Latency per memo | Cost per memo |
|------|------------------|---------------|
| Proprietary deterministic brain | ~500 ms | $0.00 |
| CrewAI + hosted LLM | 15–30 s | $0.03–$0.10 |

## Evaluation Rigor

The dev set shipped with this project is a **canonical sidecar** of 25 documents and 840 spans that went through six pipeline phases (candidate review → repair overlay → quality preflight → tool/skill application → metadata enrichment → canonical promotion) before being stored in `data/processed/`. Every phase produces its own audit run with handoff record, validation report, and provenance log. Once artifacts are finalized they are locked read-only.

The pipeline aligns with:
- **NIST AI Risk Management Framework** — testing, evaluation, verification, validation throughout the AI lifecycle.
- **EU AI Act Article 14** — human oversight of high-risk AI systems.

### Integrity check (recomputable from the canonical sidecars)

| Check | Claimed | Actual | Match |
|-------|---------|--------|-------|
| `labels.json` sha256 | 4aa8f79c, 2,841,532 bytes | 4aa8f79c, 2,841,532 bytes | yes |
| `provenance.jsonl` sha256 | 6dd14b1f, 2,115,664 bytes (840 lines) | 6dd14b1f, 2,115,664 bytes | yes |
| `manifest.json` sha256 | 259fff16, 12,630 bytes | 259fff16, 12,630 bytes | yes |
| Record / line counts | 840 records, 840 provenance lines, 25 docs | 840 / 840 / 25 | yes |
| Per-label totals (SKILL 311, TOOL 277, ...) | sums to 840 | sums to 840 | yes |
| Override breakdown | 31 explicit overrides, 809 null | 31 / 809 | yes |
| Round-trip `masked_text[s:e] == text` | 840/840 pass | 840/840 pass | yes |
| Duplicate spans across documents | 0 | 0 | yes |
| Overlap or nesting in retained set | 0 pairs | 0 pairs | yes |
| Per-document masked-text hash match | 25 of 25 | 25 of 25 | yes |
| Forbidden source-key deep walk on every emitted JSON | 0 hits | 0 hits | yes |
| Schema version | v7 (10 labels) | v7 (10 labels) | yes |
| Blind set discipline | never opened, read, stat'd, or hashed | confirmed | yes |

When a department head, internal auditor, or labor relations attorney asks "How do I know the model is right?" the answer needs to be more than "Trust the model." Every evaluation number cited in the dashboard or in a memo traces back to a specific span in this canonical sidecar with its review history attached. The integrity bracket is the receipt.

### Honest model performance

- **DistilBERT NER (alone) on real held-out resumes:** F1 = 0.74
- **Routed ensemble (DistilBERT + GLiNER + ModernBERT) on the same held-out set:** F1 = 0.78
- **Bi-LSTM attrition (after the double-sigmoid fix):** prediction spread 0.12 to 0.57, Brier score 0.232 (near the theoretical floor for a balanced label)

The earlier NER fine-tune that reported F1 = 0.9999 on a synthetic val split was meaningless because the val set used the same templates as training. After rebuilding the held-out set from real annotated resumes, the honest number was 0.74 / 0.78. That is what the project ships.

## Lessons Learned

### What worked best
Making the deep learning models a **pure tool layer beneath a swappable orchestrator**. The same DistilBERT, GLiNER, ModernBERT, SBERT, and Bi-LSTM stack serves the proprietary brain, the optional CrewAI path, and the Streamlit boardroom dashboard. When I caught the Bi-LSTM double-sigmoid bug late in the build, the fix was a one-line change to the model file. Nothing in the agent layer needed to know.

### What did not work — twice
1. **Bi-LSTM saturation.** The model produced bimodal probabilities at 0.00 and 1.00 across all eight departments for almost two weeks. AUC alone did not flag it. Only when I plotted the probability distribution did the saturation become visible. Cause: a redundant `nn.Sigmoid()` in the classification head, paired with `BCEWithLogitsLoss` during training (which already applies sigmoid internally). The forward pass was computing `sigmoid(sigmoid(logit))`, which collapsed gradients on confident examples toward zero, so the optimizer kept pushing logits toward ±∞. After removing the redundant sigmoid, predictions spread to 0.12–0.57 and Brier landed at 0.232.
2. **NER F1 inflation on synthetic val.** Reporting 0.9999 felt great until I realized the val set used the same templates as training. Rebuilding from real annotated resumes dropped honest F1 to 0.74 (DistilBERT) and 0.78 (ensemble).

### The surprising thing I learned
**"Agentic" does not require an LLM.** The standard mental model from the course readings is that an agent is an LLM in a loop. Working through the five workflows in `brain.py` showed me that "agentic" is really three components wired together: tools, a reasoning loop, and memory. An LLM is just one way to implement the reasoning step, not the only option. Using a 20-cell decision matrix with explicit rules is more defensible in an audited domain, executes in 500 ms instead of 30 s, costs $0 per memo instead of $0.03–$0.10, and eliminates hallucinations because the text is templated from verified tool outputs. In multi-agent systems for high-stakes domains, **tool depth substitutes for LLM horsepower** in a way that is both underappreciated and easier to engineer.

## What I Would Build Next

Three extensions ranked by significance for a real ABB pilot:

1. **SHAP attribution** to replace gradient × input driver attribution. Shapley values are mathematically sound and more defensible if a department head questions a claim like "75% of risk comes from comp ratio."
2. **Survival head on the Bi-LSTM.** The current model predicts *whether* someone will leave. A survival head (Cox proportional hazards or discrete-time Weibull) would predict *when*. "Median time to exit: 4.2 months" is more actionable for CHRO planning than a binary risk score.
3. **Workday or SuccessFactors connector** to replace the synthesized monthly HRIS data with live reads of the previous 24 months. The architecture is already designed to support that swap.

## Files

- `Final_Project_Report_Musil.pdf` — full 10-page report including architecture, model details, evaluation rigor, integrity check, lessons learned, and 18 references
- `dashboard.png` — screenshot of the boardroom dashboard (Energy Industry profile)

The complete codebase (~21 tools, the `brain.py` orchestrator, `agents.py` CrewAI path, `feedback.py` memory layer, the canonical sidecar, evaluation pipeline, Streamlit dashboard, and full provenance logs) is hosted in the project's working repository. The PDF report describes the architecture, results, and integrity check at a level appropriate for portfolio review.

## Selected References

The full report cites 18 sources including:
- IEA *World Energy Employment 2025*
- Accenture *Talent Reinventor* research from CERAWeek 2026
- Vaswani et al., *Attention Is All You Need* (NeurIPS 2017)
- Sanh et al., *DistilBERT* (NeurIPS Workshop 2019)
- Reimers and Gurevych, *Sentence-BERT* (EMNLP 2019)
- Zaratiana et al., *GLiNER* (NAACL 2024)
- Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction* (arXiv 2021)
- Gal and Ghahramani, *Dropout as a Bayesian Approximation* (arXiv 2015)
- LSTNet labor-market forecasting (arXiv 2025)
- *Computers* (MDPI), *Employee Attrition Prediction Using Deep Neural Networks* (2021)
- CrewAI official documentation
- IEEE Spectrum, ManpowerGroup, CNBC reporting on the energy / data center workforce gap

Full bibliography in the PDF report.

## Acknowledgments

This work draws directly on Module 03 (RNN/LSTM), Module 05 (Transformers), and Module 10 (Agentic AI) of ITAI 2376, plus the patterns explored in Lab 02, Lab 03, Lab 05, and the Midterm Blueprint. It also reflects what I learned at CERAWeek 2026 in Houston (March 23–27) and several years of operational HR work at ABB Energy Industries.
