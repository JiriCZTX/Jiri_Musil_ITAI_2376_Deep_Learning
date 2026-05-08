# Lab 08 — Creating Images with AI: Hands-On Diffusion Models

**Course:** ITAI 2376 — Module 08 Hands-On Lab
**Author:** Jiri Musil
**Dataset:** MNIST handwritten digits (28×28 grayscale, 60,000 training samples)

## Problem Statement

Diffusion models are the technology behind modern image generators (DALL·E, Stable Diffusion, Midjourney). The intuition is simple but counter-intuitive: teach a network to *reverse* the process of an image gradually getting blurry with noise. Once it can clean up noise step by step, the same network can take pure random noise and "denoise" it into a brand-new image that resembles the training set. This lab implements that idea from scratch on MNIST.

## Approach

- **Dataset selection.** Used MNIST as the standard option (works on free Colab, ~2 GB GPU, ~30 minutes training). The notebook also has commented-out blocks for Fashion-MNIST, CIFAR-10, and CelebA for users with more compute.
- **Reproducibility setup.** Set NumPy, Python `random`, PyTorch (CPU and CUDA) seeds to 42 and enabled `torch.backends.cudnn.deterministic = True` so re-runs produce the same outputs.
- **Train/validation split.** 80/20 random split of the 60,000 training images.
- **Model components.** Built the standard U-Net-style components for a denoising diffusion probabilistic model: convolutional blocks, time embeddings, and a noise prediction network, with `einops` for clean tensor reshaping.
- **Forward (diffusion) process.** Implemented the noise-adding schedule: starting with a clean image, gradually add Gaussian noise over many timesteps until you reach pure noise.
- **Reverse (denoising) process.** Trained the U-Net to predict the noise that was added at each timestep. Inverse the process at inference time: start from pure noise and step backward, predicting and removing noise at each step.
- **Training.** Adam optimizer, MSE loss between predicted and actual noise, ~30 epochs at batch size 64.
- **Generation.** Sampled new digits from pure Gaussian noise and visualized them.
- **Visualization of the generation process.** Captured intermediate denoising steps to show the model going from noise → fuzzy → recognizable → clean digit.
- **CLIP evaluation.** Used CLIP to score whether the generated images are recognizable enough to be classified by another AI model.

## Files

- `L08_Jiri_Musil_ITAI2376.ipynb` — full lab notebook including model components, training loop, sampling, intermediate-step visualization, CLIP evaluation, and assessment question answers

## Results

After ~30 epochs, the trained model can generate recognizable handwritten digits from pure Gaussian noise. The intermediate-step visualization makes the denoising process visible: at the beginning of sampling the output looks like static, then it becomes blurry shapes, then digit-like blobs, then clean digits. CLIP evaluation provides an external sanity check — the generated outputs are recognizable enough that a separate vision-language model can classify them.

## Learning Outcomes

- Diffusion models flip the usual generation problem on its head. Instead of generating from scratch, you train a network to *undo* a known corruption process. That reframing makes the model surprisingly stable to train.
- The forward (noising) schedule is just a fixed sequence — it does not need to be learned. The work happens in the reverse (denoising) process where the network predicts the noise that was added.
- Reproducibility for generative models matters more than for classifiers. Without seed control you cannot debug or compare runs because every single sample is random.
- The CLIP evaluation step is a useful pattern: if your generated images are too blurry or class-incoherent for another AI to recognize them, they are also probably bad for downstream applications.

## Requirements

```bash
pip install torch torchvision einops matplotlib numpy pillow ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

GPU is strongly recommended — diffusion training on CPU is impractically slow.

## How to Run

```bash
jupyter lab L08_Jiri_Musil_ITAI2376.ipynb
```

Run cells top to bottom. Make sure exactly one of the dataset blocks (MNIST, Fashion-MNIST, CIFAR-10, CelebA) is uncommented in Step 2 — the validation cell will raise an explicit error if no dataset is selected. MNIST is the safe default for free Colab.

## Notes

- For Colab Free users: change Runtime → Change runtime type → GPU before starting.
- If you want to scale up to CIFAR-10 or CelebA, expect 1–4 hours of training and 4–8 GB of GPU VRAM.
- The intermediate-step visualization is in Step 7 — it is the most pedagogically useful cell in the notebook because it makes the denoising process visible.
