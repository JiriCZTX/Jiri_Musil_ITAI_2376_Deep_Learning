# A02 — Neural Network Zoo

**Course:** ITAI 2376 — Deep Learning Artificial Intelligence
**Author:** Jiri Musil

## Problem Statement

The first conceptual hurdle in any deep learning course is that there are too many network types, all with similar-sounding names, and a beginner cannot tell which one to reach for. This assignment reframes the problem: each major network type becomes a "zoo animal" with a personality that matches its strengths, so that "convolutional neural network" stops being a label to memorize and starts being a mental image you can reason from.

## Approach

- Treat the family of neural networks as a zoo: each network is an animal, the animal's personality matches what the network is good at, and the goal is to *recognize* which animal belongs to which task rather than memorize equations.
- Walk through the basic neuron model (inputs, weights, bias, activation) and how neurons compose into input, hidden, and output layers.
- Introduce five network types as zoo animals: CNN Cheetah (image hunter), RNN Raccoon (sequence follower), LSTM Lemur (longer memory), Autoencoder Owl (compact summaries), GAN Chameleon (realistic generation).
- Pick one featured animal — the **CNN Cheetah** — and explain why CNNs are well-suited for image tasks like sorting photos, face detection, and object recognition.

## Files

- `A02_Neural_Network_Zoo_Musil.pdf` — full slide deck (8 slides)

## Learning Outcomes

- I can now look at a task description ("predict the next word in a sentence", "compress an image", "generate fake-looking images") and make a defensible first guess about which architecture to try.
- I understand that there is no single "best" network — the right answer depends on the data and the goal.
- The "zoo animal" mental model carried into later labs: when I built a CNN in Lab 03, the cheetah analogy actually helped me reason about why convolutional filters work the way they do.

## References

The deck cites standard introductory sources including Dive into Deep Learning, GeeksforGeeks, AWS, Christopher Olah's LSTM blog post, the TensorFlow autoencoder tutorial, and educational material on GANs. Full reference list is on the final slide.
