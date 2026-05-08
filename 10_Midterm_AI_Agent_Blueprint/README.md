# Midterm Project — AI Agent Blueprint

**Course:** ITAI 2376 — Deep Learning Artificial Intelligence
**Author:** Jiri Musil
**Title:** Multi-Agent Workforce Intelligence System for the Energy Industry
**Submitted:** April 6, 2026

## Problem Statement

I work as a People Strategy Lead for ABB's Energy Industries division in Houston, with workforce planning oversight across seven countries. After attending CERAWeek 2026 — four days, 10,000 energy leaders, 14 sessions — the strongest message was that the energy industry does not have enough workers to build and operate what it is planning, and nobody in the room had a credible plan to fix it.

The IEA's *World Energy Employment 2025* report shows 76 million energy jobs worldwide with 2.4 workers nearing retirement for every new hire under 25 in advanced economies. CERAWeek panels reported that 40% of plant operators could retire by the end of this decade. Accenture found at the same event that only 8% of energy organizations have meaningful skills visibility, while 85% of executives believe AI will transform their business and only 18% see tangible benefits. This blueprint targets that visibility gap directly.

## What This Document Is

This is the *design* for the final project — the architecture, the model choices, the framework selection, the data plan, the build schedule, and the anticipated challenges. It is **not** the implementation; that lives in [11_Final_Multi_Agent_Workforce_Intelligence/](../11_Final_Multi_Agent_Workforce_Intelligence/).

## Approach

### Option choice — Multi-Agent System (Option B)

Matching candidates to roles is a **language** problem (unstructured text in, structured signals out). Forecasting workforce attrition is a **time-series** problem (sequential numerical data in, probabilities and projections out). These are different enough that combining them into a single agent would be harder to build, debug, and retrain. Two specialized agents talking through an orchestrator maps cleanly onto how HR teams actually work — recruiters and workforce planners are different roles that exchange data.

### Architecture overview

```
INPUT → DL MODEL → AGENT BRAIN (LLM) → TOOL → OUTPUT
                            ↑
                            ├── Agent 1: Talent Intelligence
                            └── Agent 2: Workforce Forecasting
                            ↓
                    CrewAI Orchestrator
                            ↓
              Unified Streamlit + Plotly Dashboard
```

### Agent 1 — Talent Intelligence
- **Input:** resumes (PDF/DOCX), job descriptions
- **DL Model:** DistilBERT fine-tuned for NER (SKILL, CERTIFICATION, DEGREE, EMPLOYER, YEARS_EXPERIENCE) + Sentence-BERT embeddings + cosine similarity for matching
- **Brain:** LLM (Claude / GPT-4) that reasons about which candidates fit the role, identifies skill gaps, queries the O*NET taxonomy, and writes the ranked report
- **Tool:** O*NET skills taxonomy + vector database search
- **Output:** ranked candidate list with skill-gap breakdown

### Agent 2 — Workforce Forecasting
- **Input:** HRIS time-series data (headcount, turnover, tenure, comp ratios, performance) plus BLS energy-sector employment indices
- **DL Model:** Bidirectional LSTM that outputs individual attrition probabilities (binary classification, 6-month horizon) and department-level headcount forecasts (regression at 3, 6, 12 months)
- **Brain:** LLM that interprets predictions, queries BLS / HRIS for context, identifies whether attrition is driven by comp gaps, retirement cohorts, or market competition
- **Tool:** BLS API + HRIS database
- **Output:** attrition risk scores + hiring forecast report

### Agent framework — CrewAI
CrewAI was chosen over LangChain (too heavy for this case where DL models do the heavy lifting) and AutoGen (too conversation-oriented for a structured pipeline). CrewAI lets me wrap PyTorch models as registered tools and define agents with explicit roles, goals, and tool access — which mirrors how the system needs to be structured anyway.

### Tools & data table

| Category | Tool / Source |
|----------|---------------|
| Deep Learning | PyTorch, Hugging Face Transformers |
| NLP | spaCy, NLTK |
| Agent Framework | CrewAI |
| Data — Resumes | Kaggle Resume Dataset + synthetic energy-sector |
| Data — Workforce | IBM HR Analytics (Kaggle), BLS QCEW |
| Data — Jobs | O*NET, Rigzone, EnergyJobline |
| Dashboard | Streamlit + Plotly |
| Infrastructure | Python 3.11, Google Colab Pro |

### Build plan — 4 weeks

| Week | Dates | Deliverables |
|------|-------|--------------|
| 1 | Apr 7–13 | Repo setup, dataset prep, hand-label 80–100 energy resumes for NER, fine-tune DistilBERT |
| 2 | Apr 14–20 | SBERT pipeline + matching, synthesize HRIS time series, build & train Bi-LSTM |
| 3 | Apr 21–27 | Wire both agents into CrewAI Flow, build Streamlit dashboard, integration testing |
| 4 | Apr 28–May 3 | Evaluation (F1 for NER, AUC-ROC for attrition, MAE for forecast), bug fixes, write report, record demo |

### Anticipated challenges

1. **Not enough labeled energy resumes.** General NER models do not understand "P&ID review", "HAZOP facilitation", or "API 650 certified". Plan: spaCy pre-trained model as a starting point, manually correct 80–100 energy resumes, then augment with synonym swaps and back-translation.
2. **Lopsided attrition data.** ~85% stay / ~15% leave. Plan: SMOTE for balance, weighted loss, evaluate with AUC-ROC and precision-recall (not accuracy). Also: IBM HR is cross-sectional, so synthesize monthly snapshots to enable temporal training.
3. **Wiring custom PyTorch models into CrewAI.** CrewAI expects LLM agents. Plan: register each model as a custom tool function. Fallback: simplify to a sequential pipeline if Flow integration fails.
4. **GPU access and fairness.** Colab Pro has session limits. Plan: reduce epochs and hidden dims if needed. Fairness: review match-score distributions across demographic groups and report whatever I find, even if it is not clean.

## Files

- `Midterm_Blueprint_Musil.pdf` — full 11-page blueprint with all diagrams, tables, and 15 references

## What Changed Between This Blueprint and the Final Project

This is worth flagging explicitly. The framework choice was the biggest pivot.

- **Blueprint plan:** CrewAI + hosted LLM as the agent's "brain"
- **Final project shipped:** A custom deterministic, LLM-free orchestrator (`brain.py`) is the *primary* path. CrewAI + LLM is retained as an optional secondary path.
- **Why:** Workforce planning decisions are audited, regulated, and contestable. EU AI Act classifies HR systems as high-risk. An LLM brain produces fluent narratives but cannot explain in plain language which rule produced a specific verdict. The deterministic orchestrator runs in ~500 ms with $0 marginal cost and zero hallucinations because output text is templated from verified tool outputs.

The deep learning stack (DistilBERT, GLiNER, ModernBERT, SBERT, Bi-LSTM with attention) is shared across both paths. See the final project README for full details.

## Learning Outcomes

- Designing a system before building it is genuinely useful. Most of the questions I had to think through during the build were already on this blueprint, just answered better in retrospect.
- The "anticipated challenges" section is where the design got real. Lopsided data, dataset shape mismatches, framework integration risk, and fairness all turned into actual problems during implementation, and the mitigation plans I drafted here mostly held up.
- The blueprint format forced me to choose specific tools (DistilBERT vs other transformers, CrewAI vs LangChain, Bi-LSTM vs LSTNet) and defend each choice with course material plus current research. That habit transferred directly into how I now scope projects at work.
