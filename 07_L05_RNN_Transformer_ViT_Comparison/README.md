# Lab 05 — RNNs vs Transformers vs Vision Transformers

**Course:** ITAI 2376 — Module 05 Hands-On Lab
**Author:** Jiri Musil
**Datasets:** AG News (text classification, 4 classes) and CIFAR-10 (image classification, 10 classes)

## Problem Statement

This is the most demanding lab in the course. It trains and evaluates seven different models across text and image tasks so that the strengths and weaknesses of recurrent networks, transformers, and vision transformers can be compared head-to-head on the *same* data. The point is not to declare a winner but to see, in actual measured numbers, why architectural choices matter.

## Approach

The lab is structured in four parts:

### Part A — RNN text classification on AG News
Built three classifiers from scratch in PyTorch using the *same* embedding layer, *same* hidden dimension (128), *same* training loop, and the *same* 8,000-sample subset. Only the recurrent layer differed.

- **Vanilla RNN classifier** — `nn.RNN`, no gating, demonstrates the vanishing gradient problem.
- **LSTM classifier** — `nn.LSTM`, three gates (input, forget, output) plus cell state.
- **GRU classifier** — `nn.GRU`, two gates (reset, update), no separate cell state.

All three are bidirectional with a 0.3 dropout and a final fully connected head.

Also ran a hyperparameter sweep across hidden dim (64/128/256), dropout (0.1/0.3/0.5), and learning rate (1e-3/5e-4/1e-4) on the LSTM to develop intuition for how each axis affects accuracy.

### Part B — Transformer text classification on AG News
Fine-tuned **DistilBERT** (`distilbert-base-uncased`) on the same 8,000-sample AG News subset for 3 epochs at lr=2e-5 with `AdamW` and weight decay 0.01. Also ran a smaller hyperparameter sweep on batch size and learning rate.

### Part C — Vision Transformer on CIFAR-10
Fine-tuned **ViT** (`google/vit-base-patch16-224`) on a 2,000-image CIFAR-10 subset, then trained a `SimpleCNN` baseline (3 conv blocks + BN + GAP) on the same subset for a fair comparison.

Also implemented attention-map extraction with a forward hook on `model.vit.encoder.layer[-1].attention.attention`, averaged across heads, and visualized which patches the model focuses on for five different classes.

### Part D — Master comparison and reflection
Built a comprehensive comparison table covering test accuracy, training time, parameters, strengths, weaknesses, and best use cases for all six trained models. Wrote seven reflection questions on what I observed.

## Files

- `L05_Jiri_Musil_ITAI2376.ipynb` — full lab notebook with completed student-coding sections (A.6, A.7, A.8, B.4, B.5, C.4, C.5) and all reflection answers

## Results

### Text classification — AG News (8,000 train / 2,000 test)

| Model | Parameters | Avg Epoch Time | Test Accuracy |
|-------|------------|----------------|---------------|
| Vanilla RNN | 1,347,076 | 0.4 s | **70.30%** |
| LSTM | 1,545,220 | 1.2 s | 79.45% |
| GRU | 1,479,172 | 0.7 s | 81.15% |
| DistilBERT | ~66.9 M | 96.1 s | **90.15%** |

### Image classification — CIFAR-10 subset (2,000 train / 1,000 test)

| Model | Parameters | Train Time (5 epochs) | Test Accuracy | Inference (batch) |
|-------|------------|----------------------|---------------|--------------------|
| Simple CNN (from scratch) | 94,986 | ~30 s | 31.00% | 0.001 s |
| ViT (pre-trained, fine-tuned) | ~85.8 M | ~240 s | **95.50%** | 0.149 s |

### Per-class accuracy on AG News

The hardest category for every text model was **Business**, where vocabulary overlaps with **Sci/Tech**. Vanilla RNN bottomed out at 56.32% on Business while topping out at 76.50% on Sports. DistilBERT pulled all four categories above 87%.

## Learning Outcomes

- **The vanishing gradient problem is real.** A 10-point accuracy gap between Vanilla RNN (70.30%) and gated architectures (LSTM 79.45%, GRU 81.15%) on identical data, training loop, and hyperparameters. Same code, just different recurrent layer.
- **GRU beat LSTM in this run** (81.15% vs 79.45%) despite having fewer gates. With more epochs the gap might close — GRU's simpler structure often converges faster.
- **Pre-training is a step change, not an improvement.** DistilBERT after 1 epoch (89.50%) already beats GRU after 5 epochs (81.15%). ViT after 1 epoch hits ~95% on CIFAR-10 while my from-scratch CNN needs 5 epochs to get to 31%.
- **Attention maps are interpretable.** For airplane images the bright regions concentrate on fuselage and wings; for animals they focus on the head and body outline. Backgrounds get low attention. ViT looks at images more "holistically" than a CNN's local feature maps would.
- **DistilBERT optimal lr is ~50× smaller than LSTM's** (2e-5 vs 1e-3) because pre-trained weights need gentle updates — large learning rates cause catastrophic forgetting of the pre-trained representations.
- **Debugging story I learned the hard way:** wrote `output, (hidden, cell) = self.rnn(embedded)` for the Vanilla RNN by copy-pasting from LSTM. `nn.RNN` returns only `(output, hidden)` like GRU, no cell state. The fix was one line. Read error messages carefully before you start poking at tensor shapes.

## Requirements

```bash
pip install torch torchvision transformers datasets accelerate matplotlib numpy scipy pandas
```

A GPU (CUDA) is strongly recommended for DistilBERT and ViT fine-tuning. The lab is designed for Google Colab T4 GPU but runs on local GPUs as well.

## How to Run

```bash
jupyter lab L05_Jiri_Musil_ITAI2376.ipynb
```

Run cells top to bottom. The notebook downloads AG News and CIFAR-10 automatically through the Hugging Face `datasets` library and `torchvision`. Pre-trained models (DistilBERT, ViT) are pulled from Hugging Face on first run.

## Notes

- The reflection answers in the notebook are written from my own observations on this exact dataset and configuration. If you re-run with different hyperparameters or seeds, your numbers will differ — that is expected and is part of the point.
- Memory-efficient tip: reduce `TRAIN_SIZE` to 4,000 or `VIT_TRAIN_SIZE` to 1,000 if you hit OOM errors on free Colab.
