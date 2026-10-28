# Data Poisoning Attacks & Defenses on a DCGAN

An AI security experiment that measures how a generative adversarial network (GAN) behaves when its training data is poisoned, and tests a defense that filters poisoned samples before training.

Built in PyTorch as part of my M.S. in AI Development at Kennesaw State University.

## Overview

- **Model:** DCGAN (Deep Convolutional GAN) that generates 64×64 face images
- **Clean data:** [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) celebrity faces
- **Poison data:** Pixar-style animated faces, out-of-distribution images injected into the training set
- **Attack:** train separate models with 0%, 5%, 25%, and 50% of the training set poisoned
- **Defense:** [Spectral Signatures](https://arxiv.org/abs/1811.00636) (Tran et al., 2018). Project the centered data onto its top singular vectors, flag the highest-scoring outliers, and remove them before training.

## What's measured

For each experiment, the trained discriminator is evaluated on:

| Metric | Meaning |
|---|---|
| Accuracy | How well it separates real faces from generated ones |
| Precision / Recall | Real-image classification quality |
| **Poison acceptance rate** | Share of poison images the discriminator accepts as "real". This shows how much the poison has shifted what the model considers normal. |

## Results

Running the notebook saves these charts:

- `metrics_comparison.png`: accuracy, precision, and recall for each poisoning level
- `poison_acceptance.png`: poison acceptance rate as poisoning increases
- `loss_curves.png`: generator and discriminator loss for each experiment
- `generated_images_comparison.png`: final generated samples from each model

<!-- After running, add images, for example:
![Metrics](metrics_comparison.png)
![Poison acceptance](poison_acceptance.png)
-->

## How to run

This project was built in **Google Colab** (GPU runtime recommended).

1. Open `ai_security_project.py` in Colab (or paste the cells into a new notebook)
2. Upload `img_align_celeba.zip` when prompted. Download it from the [CelebA site](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
3. Put a zip of Pixar-style face images at `/content/drive/MyDrive/pixar_dataset.zip`
4. Run all cells. The charts are saved as PNG files.

The datasets are not included in this repo (CelebA's license doesn't allow redistribution).

## Limitations

- Training is kept short (3 epochs) to fit within Colab's limits
- The "fake" images used to score the discriminator come from a freshly initialized generator, not the trained one
- Spectral Signatures runs on raw pixel features. Using learned features, such as the discriminator's penultimate layer, would likely separate poison images more cleanly.

## Tech

Python · PyTorch · torchvision · NumPy · scikit-learn · Matplotlib
