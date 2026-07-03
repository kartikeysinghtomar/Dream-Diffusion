# DreamDiffusion — BCS Summer Project 2025

Implementation of DreamDiffusion for EEG to image generation, developed as part of the Brain and Cognitive Society (BCS) Summer Project 2025 at IIT Kanpur.

## Overview

DreamDiffusion is a framework that generates images directly from EEG (electroencephalogram) signals recorded while subjects viewed images from the ImageNet dataset. EEG data was collected using a PsychoPy paradigm in which images were presented to participants and their neural responses were recorded.

This project implements and fine-tunes the DreamDiffusion pipeline, adapting a pre-trained text-to-image diffusion model to accept EEG representations as conditioning input instead of text embeddings.

## Background

EEG-based image generation faces three core challenges: high signal noise, limited information content in EEG, and individual variability across subjects. DreamDiffusion addresses these through two key contributions:

1. **Temporal Masked Signal Modelling (TMSM):** An EEG encoder is pre-trained using a masked autoencoding objective over time-series EEG segments, allowing it to learn robust representations without relying on large labelled EEG-image datasets.

2. **CLIP Alignment:** The CLIP image encoder is used as an additional supervisory signal during fine-tuning, aligning EEG embeddings to the joint text-image embedding space of Stable Diffusion. This allows the model to bridge the gap between EEG signals and visual semantics even with limited paired data.

The pre-trained Stable Diffusion model is then conditioned on these EEG embeddings to synthesise images that correspond to what the subject was viewing.

## Repository Structure

| File | Description |
|---|---|
| `dream diffusion final pipeline.ipynb` | End-to-end notebook covering EEG preprocessing, encoder training, CLIP alignment, and image generation |

## Requirements

The notebook relies on the following primary dependencies:

- Python 3.8+
- PyTorch
- HuggingFace Diffusers and Transformers
- OpenAI CLIP
- MNE (EEG preprocessing)
- NumPy, Matplotlib

## References

- Bai, W., Wang, X., et al. *DreamDiffusion: Generating High-Quality Images from Brain EEG Signals.* arXiv:2306.16934, 2023. [Link](https://arxiv.org/pdf/2306.16934)
