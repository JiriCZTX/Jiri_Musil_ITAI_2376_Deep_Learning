# Jiri Musil — ITAI 2376 Deep Learning Portfolio

**Houston City College — Department of Science, Technology, Engineering & Math**
**ITAI 2376 Deep Learning Artificial Intelligence — Spring 2026**
**Instructor:** Professor Patricia McManus

---

## About Me

I am Jiri Musil, a People Strategy Lead at ABB's Energy Industries division in Houston, currently completing the Applied AI and Robotics program at Houston City College. My professional focus is workforce planning, HR business partnering, and AI deployment in highly regulated, audited domains. The projects in this portfolio reflect both the deep learning fundamentals taught in ITAI 2376 and an applied perspective on building AI systems that hold up under real-world scrutiny.

- **Email:** jiri.musil@students.hccs.edu
- **Course:** ITAI 2376 — Deep Learning Artificial Intelligence
- **Institution:** Houston City College, AI & Robotics Program

---

## Portfolio Overview

This repository documents my full learning journey through ITAI 2376 — from neural network fundamentals to a multi-agent AI system that I now use as the basis for a workforce intelligence pilot at ABB. The work moves from foundations (MLP, CNN, RNN), through modern architectures (Transformers, ViT, Diffusion Models), into reinforcement learning, and finishes with a production-grade multi-agent system that integrates DistilBERT, GLiNER, ModernBERT, Sentence-BERT, and a Bidirectional LSTM with attention behind a deterministic, auditable orchestrator.

Every project includes the actual source code (Jupyter notebooks or full PDFs of written work), a README with the problem, approach, results, and what I learned, and clear instructions for reproducing the results.

---

## Skills and Technologies

**Deep Learning Frameworks**
- PyTorch (primary): custom modules, training loops, autograd, hooks
- TensorFlow / Keras: Sequential and Functional API, ImageDataGenerator, callbacks

**Architectures Implemented**
- Multi-Layer Perceptrons (MLP) — built from scratch with NumPy and in PyTorch/Keras
- Convolutional Neural Networks (CNN) — custom architectures + transfer learning (ResNet50, MobileNetV2)
- Recurrent Networks — Vanilla RNN, GRU, LSTM, Bidirectional LSTM with attention
- Transformers — DistilBERT (fine-tuned), Sentence-BERT, GLiNER, ModernBERT
- Vision Transformer (ViT) — patch-based image classification with attention visualization
- Diffusion Models — DDPM-style generation on MNIST
- Reinforcement Learning — Q-Learning with epsilon-greedy exploration on CartPole

**Agent Frameworks and Orchestration**
- CrewAI (multi-agent orchestration with custom tool registration)
- Custom deterministic orchestrators (Plan-Act-Observe-Respond loop)
- Streamlit + Plotly dashboards

**Evaluation and Tooling**
- Hugging Face Transformers and Datasets
- scikit-learn metrics, ROC/AUC, confusion matrices, F1, Brier score
- Hyperparameter sweeps and learning-curve diagnostics
- NIST AI Risk Management Framework alignment, EU AI Act Article 14 compliance patterns

**Languages and Environments**
- Python 3.11, Jupyter, Google Colab (free tier and Pro)
- Git / GitHub, Markdown documentation

---

## Projects

The portfolio is organized chronologically, mirroring the course structure. Each folder contains source code, supporting documents, and a project-specific README.

### Foundations and Comparative Analysis

| # | Project | Type | Highlights |
|---|---------|------|------------|
| 01 | [A Comparative Analysis of ML and DL Tools and Frameworks](01_A01_ML_vs_DL_Frameworks_Preterm_Risk/) | Notebook + Report | PyTorch vs TensorFlow head-to-head on a synthetic preterm-risk classifier; both frameworks reach 96.7% test accuracy on the same data. |
| 02 | [Neural Network Zoo](02_A02_Neural_Network_Zoo/) | Slide deck | Visual taxonomy of CNN, RNN, LSTM, Autoencoder, and GAN — featured network is the "CNN Cheetah." |
| 03 | [Deep Learning for an 11-Year-Old](03_A03_Deep_Learning_For_11_Year_Old/) | Slide deck | Explanation of deep learning concepts at an 11-year-old reading level. |
| 04 | [Analyzing *Arrival* Through the Lens of NLP](04_A04_Arrival_Through_NLP_Lens/) | Written analysis | NLP themes (low-resource learning, segmentation, ambiguity, pragmatics) traced through scenes from the 2016 film. |

### Hands-On Labs

| # | Project | Architecture | Dataset | Key Result |
|---|---------|--------------|---------|------------|
| 05 | [Neural Networks from Scratch to Frameworks](05_L02_Neural_Networks_From_Scratch/) | MLP (NumPy → PyTorch → Keras) | Fashion-MNIST | 88.97% validation accuracy with PyTorch baseline; explored dropout, optimizers, activations. |
| 06 | [CNN — Puppy vs Bagel](06_L03_CNN_Puppy_vs_Bagel/) | Custom CNN + ResNet50 transfer learning | Puppy or Bagel (Kaggle) | Built a 4-block CNN from scratch and fine-tuned the last 30 layers of ResNet50. |
| 07 | [RNNs vs Transformers vs ViT](07_L05_RNN_Transformer_ViT_Comparison/) | Vanilla RNN, LSTM, GRU, DistilBERT, Simple CNN, ViT | AG News + CIFAR-10 | DistilBERT 90.15% vs LSTM 79.45% on AG News; ViT 95.50% vs CNN 31.00% on CIFAR-10 subset. |
| 08 | [Diffusion Models](08_L08_Diffusion_Models_MNIST/) | DDPM-style U-Net | MNIST | Generative model that learns to denoise pure noise back into recognizable digits, evaluated with CLIP. |
| 09 | [RL Foundations — CartPole](09_L09_RL_CartPole_QLearning/) | Q-Learning (epsilon-greedy) | Gymnasium CartPole-v1 | Random baseline averaged 18.5/500; Q-Learning agent learns over 500 episodes. |

### Term Projects

| # | Project | Scope | What It Demonstrates |
|---|---------|-------|----------------------|
| 10 | [Midterm — AI Agent Blueprint](10_Midterm_AI_Agent_Blueprint/) | Design document | Multi-agent system architecture, choice justification, deep-learning model selection, build plan, anticipated challenges. |
| 11 | [Final — Multi-Agent Workforce Intelligence System](11_Final_Multi_Agent_Workforce_Intelligence/) | End-to-end system | Production-style multi-agent system for energy-sector workforce planning. Combines DistilBERT, GLiNER, ModernBERT, SBERT, and a Bi-LSTM with attention behind a deterministic, audit-ready orchestrator. |

---

## Featured Project — Multi-Agent Workforce Intelligence System

The capstone of this portfolio is a multi-agent AI system that takes resumes and twelve months of HRIS time-series data and produces a boardroom-ready memo containing a hiring decision, a retention plan, and a backfill pipeline. Every number in the memo traces back to a verified deep learning model output, and every verdict is backed by a named rule (R1 through R20) plus a full execution trace.

The system was built to address a problem I live every day at ABB: the energy industry needs to hire and retain workers across seven countries, and existing HR tooling cannot tell us which skills our people have, let alone which skills are about to leave. The IEA's *World Energy Employment 2025* report shows 2.4 workers near retirement for every new hire under 25 in advanced economies, and Accenture's CERAWeek 2026 research found that only 8% of energy organizations have meaningful skills visibility. This system targets that visibility gap directly.

**Highlights**
- Two specialized agents (Talent Intelligence and Workforce Forecasting) coordinated by an orchestrator
- Deterministic, LLM-free reasoning core (~500 ms per memo) with optional CrewAI + LLM path
- Routed per-class NER ensemble: DistilBERT, GLiNER, ModernBERT each own the labels they are best at
- Bidirectional LSTM with attention for department-level attrition forecasting
- Triple-procedure consensus (sophisticated 20-cell, conservative 20-cell, naive heuristic) with automatic escalation on disagreement
- Counterfactual flip explainer on every DEFER/DECLINE verdict
- Human-in-the-Lead memory layer with traceable, reversible adaptation
- Evaluation on a canonical sidecar of 25 documents and 840 spans, locked read-only after acceptance
- Aligned with NIST AI RMF and EU AI Act Article 14

See [11_Final_Multi_Agent_Workforce_Intelligence/](11_Final_Multi_Agent_Workforce_Intelligence/) for the full report.

---

## Repository Structure

```
Jiri_Musil_ITAI_2376_Deep_Learning/
├── README.md                                      # This file
├── 01_A01_ML_vs_DL_Frameworks_Preterm_Risk/       # PyTorch vs TF/Keras comparison
├── 02_A02_Neural_Network_Zoo/                     # Network taxonomy slide deck
├── 03_A03_Deep_Learning_For_11_Year_Old/          # Explainer slide deck
├── 04_A04_Arrival_Through_NLP_Lens/               # NLP analysis of the film
├── 05_L02_Neural_Networks_From_Scratch/           # MLP fundamentals lab
├── 06_L03_CNN_Puppy_vs_Bagel/                     # CNN + transfer learning lab
├── 07_L05_RNN_Transformer_ViT_Comparison/         # RNN vs Transformer vs ViT lab
├── 08_L08_Diffusion_Models_MNIST/                 # Diffusion model lab
├── 09_L09_RL_CartPole_QLearning/                  # Reinforcement learning lab
├── 10_Midterm_AI_Agent_Blueprint/                 # Multi-agent system design
├── 11_Final_Multi_Agent_Workforce_Intelligence/   # End-of-course capstone
├── Presentation/                                  # Portfolio presentation (added separately)
├── requirements.txt                               # Python dependencies
└── .gitignore
```

---

## Running the Notebooks

Most notebooks were developed in Google Colab and run end-to-end on the free tier. To run locally:

```bash
git clone https://github.com/JiriCZTX/Jiri_Musil_ITAI_2376_Deep_Learning.git
cd Jiri_Musil_ITAI_2376_Deep_Learning
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Individual project READMEs list any extra dependencies (Kaggle API for Lab 03, Hugging Face access for Lab 05, Gymnasium for Lab 09).

---

## Course Reflection

ITAI 2376 took me from "I have heard of backpropagation" to building, evaluating, and shipping a multi-agent AI system that I would defend in front of an internal auditor. The most useful lessons were the unglamorous ones — how to build a held-out evaluation set that is not contaminated by training, why a model that reports F1 of 0.9999 on the val split is almost certainly broken, and how a one-line bug (a redundant `nn.Sigmoid()` paired with `BCEWithLogitsLoss`) can saturate predictions for two weeks if you only watch AUC and never plot the probability distribution. Those lessons came directly from the labs and from the time I spent pushing my final project past the point of "it runs" into "I can defend every number."

The other lesson worth naming is that "agentic" does not require an LLM. The standard mental model from the course readings is that an agent is an LLM in a loop, and I built that path. But for an audited HR domain, the LLM-free deterministic orchestrator turned out to be more defensible, faster, cheaper, and free of hallucinations, while the same DL stack sat underneath both paths. Tool depth substitutes for LLM horsepower more often than the discourse suggests.

---

## License and Attribution

Course materials, dataset references, and external libraries used in each project are credited in the per-project READMEs and in the references sections of the original PDFs. Code I wrote is shared for educational and portfolio purposes.

For questions about the work in this repository or the workforce intelligence system in particular, contact me at the email above.
