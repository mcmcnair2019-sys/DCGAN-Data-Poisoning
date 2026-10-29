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

Each model was trained for 3 epochs on 6,665 clean CelebA images plus the poison images for that experiment.

| Experiment | Poison images | Accuracy | Precision | Recall | Poison acceptance |
|---|---|---|---|---|---|
| Poison 0% (baseline) | 0 | 0.902 | 1.000 | 0.804 | **46.6%** |
| Poison 5% | 350 | 0.378 | 0.431 | 0.756 | **90.8%** |
| Poison 25% | 2,221 | 0.782 | 1.000 | 0.564 | **40.6%** |
| Poison ~50% (48.6%) | 6,297 | 0.933 | 1.000 | 0.866 | **99.4%** |
| Defended ~50% (Spectral Signatures) | 6,297 before filtering | 0.760 | 1.000 | 0.520 | **93.2%** |

### Key findings

- **Poisoning teaches the discriminator to accept the poison as real.** The clean baseline accepted 46.6% of Pixar faces as real, which shows they already look somewhat face-like. With 5% poison, acceptance jumped to 90.8%. With about 50% poison, the discriminator accepted 99.4% of them, so the poison became part of what the model considers "real."
- **A small amount of poison caused the most instability.** At 5% poison, accuracy dropped from 0.902 to 0.378 and precision from 1.000 to 0.431, so a few hundred out-of-distribution images was enough to disrupt the discriminator.
- **The 25% run was an outlier.** Its acceptance rate (40.6%) came in below the baseline. GAN training is noisy, and with only 3 epochs and a single run per setting, results can vary from run to run. Repeating each experiment with several random seeds would make the trend more reliable.
- **The spectral signature defense helped only slightly at high poison levels.** It flagged and removed 1,944 of 12,962 samples (15%), which lowered poison acceptance from 99.4% to 93.2%. Since nearly half the dataset was poisoned, removing 15% could not remove most of the poison. Spectral Signatures is designed for small poison fractions, where poisoned samples stand out as outliers. At about 50% poison, the poison is no longer an outlier. Defended training also briefly destabilized early on, with the discriminator overpowering the generator during the first two epochs before recovering.

### Charts

Running the notebook also saves these charts:

- `metrics_comparison.png`: accuracy, precision, and recall for each poisoning level
- `poison_acceptance.png`: poison acceptance rate as poisoning increases
- `loss_curves.png`: generator and discriminator loss for each experiment
- `generated_images_comparison.png`: final generated samples from each model

<!-- Upload the PNGs to the repo, then uncomment:
![Metrics](metrics_comparison.png)
![Poison acceptance](poison_acceptance.png)
![Loss curves](loss_curves.png)
![Generated images](generated_images_comparison.png)
-->

### Next steps

- Run the defense at lower poison levels (5% and 25%), where Spectral Signatures should work better
- Set the filter's removal fraction closer to the expected poison rate
- Run the filter on learned features (the discriminator's penultimate layer) instead of raw pixels
- Repeat each experiment across multiple seeds and report averages

## How to run

This project was built in **Google Colab** (GPU runtime recommended).

1. Open `ai_security_project.py` in Colab (or paste the cells into a new notebook)
2. Upload `img_align_celeba.zip` when prompted. Download it from the [CelebA site](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
3. Put a zip of Pixar-style face images at `/content/drive/MyDrive/pixar_dataset.zip`
4. Run all cells. The charts are saved as PNG files.

The datasets are not included in this repo (CelebA's license doesn't allow redistribution).

## Limitations

- Training is kept short (3 epochs, one run per setting) to fit within Colab's limits
- The "50%" experiment ended up at 48.6% poison, because the poison set had only 6,297 images
- The "fake" images used to score the discriminator come from a freshly initialized generator, not the trained one
- Spectral Signatures runs on raw pixel features. Using learned features, such as the discriminator's penultimate layer, would likely separate poison images more cleanly.

## Tech

Python · PyTorch · torchvision · NumPy · scikit-learn · Matplotlib
