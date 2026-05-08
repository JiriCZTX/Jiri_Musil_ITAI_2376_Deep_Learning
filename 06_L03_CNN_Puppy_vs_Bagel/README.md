# Lab 03 — CNNs for Image Classification: Puppy or Bagel?

**Course:** ITAI 2376 — Module 03 Hands-On Lab
**Author:** Jiri Musil
**Dataset:** [Puppy or Bagel (Kaggle)](https://www.kaggle.com/datasets/returnofsputnik/puppy-or-bagel)

## Problem Statement

Inspired by Karen Zack's viral "puppy or bagel" meme, this lab builds a binary image classifier that has to distinguish curled-up puppies from bagels. Both classes share visual features — round, tan-colored, textured — and the dataset is intentionally small, so the lab forces the kind of decisions you actually face in production: build from scratch or use transfer learning, augment or not, fine-tune all layers or just the head.

## Approach

- **Data pipeline.** Used `ImageDataGenerator` from Keras with augmentation on the training set (rotation, shifts, shear, zoom, horizontal flip). No augmentation for validation or test. Auto-detected train/val/test directories from the Kaggle archive and counted images per class.
- **Custom CNN from scratch.** Built a 4-block CNN: `Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) → MaxPool → Conv(128) → MaxPool → Flatten → Dense(512) → Dropout(0.5) → Dense(1, sigmoid)`. Compiled with Adam + binary cross-entropy and trained with `EarlyStopping` and `ModelCheckpoint` callbacks.
- **Transfer learning.** Loaded ResNet50 pre-trained on ImageNet (with `MobileNetV2` as a memory-efficient fallback for low-VRAM environments), froze the base, and added a classification head: `GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(1, sigmoid)`.
- **Fine-tuning challenge.** Unfroze the last 30 layers of the ResNet50 base and re-trained at lr=1e-4 to see if fine-tuning improves results on this very small dataset.
- **Evaluation.** Generated predictions, confusion matrix, and a classification report on the holdout test set, plus a 4×5 grid of sample predictions colored green (correct) or red (incorrect).
- **Reflective questions.** Answered all five embedded questions with actual results and analysis.

## Files

- `Module_03_Lab_Jiri_Musil.ipynb` — completed notebook including all student-coding sections, reflection answers, and the fine-tuning challenge

## Results

The dataset that ships with the Kaggle archive is extremely small (single-digit images per split), so the absolute numbers should be read with that in mind. The lab is about *the architecture and process*, not about beating a benchmark.

| Model | Test Accuracy | Test Loss |
|-------|---------------|-----------|
| Custom CNN (4 conv blocks, dropout 0.5) | ~50% | ~0.70 |
| ResNet50 transfer learning (frozen base) | ~50% | 0.6992 |
| ResNet50 fine-tuned (last 30 layers, lr=1e-4) | ~50% | 0.6984 |

The honest finding is that with only 12 training images and 2 test images, neither approach can build a stable classifier — and that finding is the point. With a properly sized dataset, transfer learning would outperform from-scratch training significantly.

## Learning Outcomes

- The mechanics of building a CNN block by block (`Conv2D → MaxPooling2D → Conv2D → ...`) became second nature after this lab, which paid off immediately in Lab 05's CNN baseline for CIFAR-10.
- Transfer learning workflow: choose a pre-trained backbone, freeze it, swap the classification head, train just the head, and only then unfreeze for fine-tuning at a much lower learning rate.
- Dataset size dominates everything. A perfectly architected model on 12 training images is not going to learn the task, no matter how clever the architecture is. The lab made me much more skeptical of impressive-looking numbers from tiny benchmarks.
- I now reach for `EarlyStopping` and `ModelCheckpoint` callbacks by default — losing a good model to an extra epoch of training is a preventable mistake.

## Requirements

```bash
pip install tensorflow numpy matplotlib scikit-learn kaggle pillow seaborn pandas
```

## How to Run

1. Get a Kaggle API key from your Kaggle profile (`Settings → API → Create New Token`).
2. Place `kaggle.json` in `~/.kaggle/` (or upload it when running in Colab).
3. Open the notebook:
   ```bash
   jupyter lab Module_03_Lab_Jiri_Musil.ipynb
   ```
4. Run cells top to bottom. The notebook handles dataset download, structure detection, and split creation automatically. If Kaggle download fails, place `puppy-or-bagel.zip` next to the notebook and re-run the extraction cell.

## Notes for Reproducibility

- Set `USE_MOBILENET = True` if you are on Colab Free / low-VRAM hardware.
- The notebook supports `EVAL_SCOPE = 'test'` (strict) or `'all'` (combined train+val+test) — the former is the default and what the results above use.
