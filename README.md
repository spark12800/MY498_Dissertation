# Algorithmic Biases in Computer Vision: How CelebA Encodes and Reproduces Gender Stereotypes in Appearance

This repository contains the code and analysis for my MSc dissertation at the London School of Economics and Political Science, which examines how large-scale facial image datasets encode and reproduce cultural biases related to gender, age, and beauty. The project uses the CelebA dataset to analyse latent attribute clusters, estimate predictors of attractiveness judgement, and evaluate group fairness in deep learning models.  

---

## Repository Structure  

- **`Study1_2.ipynb`**  
  Implements **hierarchical clustering and t-SNE** of 39 binary facial attributes (Study 1a), a cluster-based analysis of attractiveness and gender differences (Study 1b), and predictive modelling with **logistic regression and XGBoost + SHAP** (Study 2).  

- **`Study3.ipynb`**  
  Fine-tunes a **ResNet-18 classifier** (with dropout & freezing layers) on CelebA to predict beauty judgement. Includes data preprocessing (cropping, resizing to 224×224, normalization), protected group construction (Young/Old × Male/Female), and subgroup evaluation using **Grad-CAM** visualisations and fairness metrics (Demographic Parity, Equal Opportunity, worst-group accuracy).  

- **`results/`**  
  Contains figures (heatmaps, dendrograms, t-SNE plots, SHAP summary plots, Grad-CAM overlays).  

- **`requirements.txt`**  
  List of dependencies for replicating the environment.  

---

## Setup & Requirements  

- Python 3.10+  
- Key libraries:  
  - `torch`, `torchvision`  
  - `xgboost`, `optuna`, `shap`  
  - `pandas`, `numpy`, `scipy`  
  - `matplotlib`, `seaborn`, `adjustText`  
  - `statsmodels`, `patsy`  

Install requirements:  
```bash
pip install -r requirements.txt
