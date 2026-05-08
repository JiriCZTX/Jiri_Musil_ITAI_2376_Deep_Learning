# Lab 02 — Neural Networks: From Scratch to Frameworks

**Course:** ITAI 2376 — Module 02 Hands-On Lab
**Author:** Jiri Musil
**Dataset:** Fashion-MNIST (10 classes of 28×28 grayscale clothing images)

## Problem Statement

Before reaching for high-level deep learning frameworks, every practitioner should understand what is happening underneath. This lab walks through the fundamentals of neural networks twice — once with the conceptual scaffolding required to build one in any language, and then with concrete PyTorch and TensorFlow/Keras implementations of the same Fashion-MNIST classifier so the framework differences become visible side by side.

## Approach

- **Setup and data preparation.** Loaded Fashion-MNIST through `torchvision.datasets.FashionMNIST` (60,000 train + 10,000 test), split the training set 80/20 into train and validation, normalized images, and visualized samples.
- **PyTorch model.** Built a `nn.Module` MLP: `Flatten → Linear(784, 512) → ReLU → Linear(512, 512) → ReLU → Linear(512, 10)`. Trained with Adam (lr=1e-3) and `CrossEntropyLoss` for 10 epochs.
- **TensorFlow/Keras model.** Built the same architecture using the Sequential API, compiled with Adam + `SparseCategoricalCrossentropy(from_logits=True)`, and trained for 10 epochs.
- **Hyperparameter exploration.** Trained the PyTorch model with two extra learning-rate / epoch combinations (lr=0.01, epochs=5 and lr=0.0001, epochs=15) to develop an intuition for the search space.
- **Overfitting study.** Added two `nn.Dropout(0.5)` layers and re-trained to observe the regularization effect.
- **Challenge: try two improvements.** Swapped activations to LeakyReLU and switched the optimizer to SGD with momentum (lr=0.05, momentum=0.9), then compared peak validation accuracy.
- **Reflection questions.** Answered every embedded question with my actual numbers and what I observed.

## Files

- `Module_02_Lab_Jiri_Musil.ipynb` — full lab notebook with all student-coding sections completed and reflection answers filled in

## Results

| Configuration | Best Validation Accuracy |
|---------------|--------------------------|
| PyTorch baseline (lr=1e-3, 10 epochs) | **88.97%** (epoch 9) |
| PyTorch with dropout (rate=0.5) | 87.76% (epoch 10) |
| Keras baseline (lr=1e-3, 10 epochs) | 88.82% (epoch 7) |
| LeakyReLU + SGD momentum challenge | 88.29% (epoch 7) |
| lr=0.01 (too high) | 85.30% (best, oscillates) |
| lr=0.0001 (too low) | 89.17% (epoch 12 of 15) |

## Learning Outcomes

- The PyTorch training loop made the mechanics of `optimizer.zero_grad() → forward → loss.backward() → optimizer.step()` concrete. Once you see it spelled out, the Keras `compile + fit` workflow stops feeling like magic and starts feeling like a sensible default.
- Dropout reduced peak validation accuracy slightly but is a useful tool when the gap between training and validation is wide. On Fashion-MNIST with 48k samples, the model is not particularly prone to overfitting, so the regularization effect is muted.
- Learning-rate sensitivity was the most useful lesson. Too high, and the optimizer oscillates around the minimum; too low, and you run out of epochs before convergence. Picking lr is more impactful than picking the optimizer.

## Requirements

```bash
pip install torch torchvision tensorflow scikit-learn matplotlib seaborn
```

## How to Run

```bash
jupyter lab Module_02_Lab_Jiri_Musil.ipynb
```

Run cells top to bottom. The notebook downloads Fashion-MNIST automatically into `./data` on first run.
