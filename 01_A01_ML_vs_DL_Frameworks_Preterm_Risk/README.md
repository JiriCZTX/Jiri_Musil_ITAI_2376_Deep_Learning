# A01 — A Comparative Analysis of Machine Learning and Deep Learning Tools and Frameworks

**Course:** ITAI 2376 — Deep Learning Artificial Intelligence
**Author:** Jiri Musil
**Use case:** Risk classification for preterm labor / preterm birth (synthetic data demonstration)

## Problem Statement

This pre-term assignment compares the two dominant deep learning frameworks — PyTorch and TensorFlow/Keras — on the same task using the same data, so any difference in result reflects framework behavior rather than dataset effects. The use case is a binary risk classifier for preterm birth, framed as a small-scale logistic regression. The data is synthetic and the model is explicitly not clinical; the point of the exercise is to make the framework comparison concrete and reproducible.

## Approach

- **Synthetic dataset.** 300 samples with three features (`age`, `prior_preterm`, `cervix_length_mm`). Labels are generated from a hidden logistic rule and the features are standardized. An 80/20 train/test split is fixed by NumPy seed.
- **PyTorch model.** A single `nn.Linear(3, 1)` layer trained for 200 epochs with `BCEWithLogitsLoss` and SGD at lr=0.1.
- **TensorFlow/Keras model.** A single `Dense(1, activation='sigmoid')` layer trained for 50 epochs with binary cross-entropy and SGD at lr=0.1.
- **Visualization.** Feature distributions by class, scatter plots, manual ROC/AUC computation (no scikit-learn), and confusion matrices.

## Files

- `ITAI2376_Demonstration_Preterm_Musil.ipynb` — runnable notebook with both frameworks side by side
- `A01a_Comparative_Analysis_Musil.pptx` — supporting slide deck
- `A01b_Comparative_Analysis_Report.docx` — written analysis

## Results

| Framework | Test Accuracy | Example Risk Score |
|-----------|---------------|--------------------|
| PyTorch (logistic, 200 epochs) | 0.967 | 0.071 |
| TensorFlow/Keras (logistic, 50 epochs) | 0.967 | 0.047 |

Both frameworks converge to essentially the same solution on this task, which is the intended takeaway — when the model is small and the problem is well-defined, the framework choice is mostly about ergonomics rather than capability.

## Learning Outcomes

- Confirmed that PyTorch and Keras produce equivalent results on identical data when the architecture is the same.
- Got a feel for the *philosophical* difference: PyTorch is explicit and Pythonic (you write the training loop), Keras is concise and declarative (`compile` + `fit`).
- Built an intuition for why almost every later lab in this course defaulted to PyTorch for custom architectures and used Keras for high-level transfer-learning pipelines.

## Requirements

```bash
pip install torch tensorflow numpy matplotlib
```

## How to Run

```bash
jupyter lab ITAI2376_Demonstration_Preterm_Musil.ipynb
```

Run all cells top to bottom. The notebook is self-contained — synthetic data is generated in cell 2 and does not require any external download.
