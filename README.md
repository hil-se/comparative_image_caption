# Modeling Image-Caption Rating from Comparative Judgments

## Overview

This study investigates whether comparative learning can serve as an effective alternative to regression-based learning for image-caption rating.  

Instead of training directly on numeric human ratings, the comparative model learns from pairwise judgments indicating which of two image-caption pairs better match each other.

---

## Method Summary

Each image-caption pair is represented using multimodal embeddings:

- Image encoder: ResNet-50 (2048 dimensions)
- Caption encoder: all-MiniLM-L6-v2 (384 dimensions)
- Concatenated representation: 2432 dimensions

Two learning paradigms are implemented:

### 1. Regression Model
- Trained on normalized human ratings
- Optimized with ranking-penalized MAE
- Evaluated using MSE, MAE, Pearson ρ, and Spearman rs

### 2. Comparative Learning Model
- Trained on pairwise preference judgments
- Uses hinge loss on score differences
- Learns relative utility ordering instead of absolute scores

---

## Architecture

Both models share the same dual-encoder multimodal architecture.

### Regression Framework

![Regression Framework](docs/regression_framework.png)

### Comparative Learning Framework

![Comparative Framework](docs/comparative_framework.png)

---

## Dataset

Experiments are conducted on the **Validated Image Caption Rating (VICR) dataset**:

- 15,646 image-caption pairs  
- 68,217 human ratings  
- Ratings from 1 to 5  

Ratings are normalized to [0, 1] for training.  
An 80/20 train-test split is applied.

Sample data is included in the `data/` directory.

---

## Repository Structure
```text

comparative_image_caption/
│
├── code/
│ ├── generate_embedding.py
│ ├── regression_model.py
│ ├── pairwise_model.py
│ ├── caption_ranking.py
│ ├── comparative_acc.py
│ ├── comparative_interrater_agreement.py
│ ├── task1_agreement_metrics.py
│ ├── compute_averages.py
│ └── checkpoint/
│
├── data/
│
├── docs/
│ ├── regression_framework.png
│ └── comparative_framework.png
|
|── results/
│
└── README.md
```
---

### Execution Order

1. `generate_embedding.py`  
   Generates multimodal embeddings using ResNet-50 and MiniLM.

2. `regression_model.py`  
   Trains and evaluates the regression baseline.

3. `pairwise_model.py`  
   Trains the comparative learning model using hinge loss.
   
5. `caption_ranking.py`  
   Evaluates same-image caption preference modeling.

6. `comparative_acc.py`  
   Computes correlation metrics and observed agreement.

7. Human agreement analysis:
   - `task1_agreement_metrics.py`
   - `comparative_interrater_agreement.py`
   - `compute_averages.py`

## Results

### Regression Baseline

- MSE = 0.0447  
- MAE = 0.1593  
- Pearson ρ = 0.7609  
- Spearman rs = 0.7089  

### Comparative Learning (N = 20)

- Pearson ρ = 0.6758  
- Spearman rs = 0.6914  

### Same-Image Caption Comparison

- Observed agreement (po) = 0.8625  
- Pearson ρ = 0.7074  
- Spearman rs = 0.7367  

Comparative learning approaches regression performance while relying only on relative preference supervision.

---

### Human Evaluation

Eight annotators completed direct rating and comparative tasks.

Inter-rater agreement (between raters):

| Task | po | κ |
|------|----|----|
| Direct Rating | 0.85 | 0.69 |
| Pairwise (Different Images) | 0.95 | 0.85 |
| Same-Image Comparison | 0.90 | 0.78 |

Comparative judgments consistently demonstrate higher inter-rater reliability than direct numeric ratings.


