# Knee Osteoarthritis Classification Using Bone Distance

Computer vision pipeline for automated knee osteoarthritis (OA) severity prediction using femur–tibia bone distance measurements extracted from 3D knee MRI scans.

This project evaluates whether geometric bone-distance features can serve as an indirect proxy for cartilage thickness in OA classification.


---

## Project Overview

Knee osteoarthritis (OA) severity is commonly measured using the Kellgren–Lawrence (KL) grading system. Cartilage thickness is a key biomarker but is difficult to segment reliably from MRI.

This project explores an alternative:

1. Use pre-segmented femur and tibia masks (generated via U-Net).
2. Measure inter-bone distance across selected MRI slices.
3. Construct fixed-length feature vectors.
4. Train machine learning models to classify OA severity.

---

## Dataset

- **Source:** Osteoarthritis Initiative (OAI) database  
- **Cases:** ~195 knees  
- **Imaging:** 3D MRI (DESS sequence)  
- **Slices per case:** 160  
- **Resolution:** 384 × 384 pixels  

Each MRI volume contains 160 2D slices.  
Binary bone masks (femur + tibia) are provided for each slice.

Not all slices contain valid anatomical regions. An automated slice selection step identifies usable slices for distance measurement.

---

## Methodology

### 1. Slice Selection

Some slices do not contain complete femur and tibia regions.

Automated filtering:
- Detect presence of both bones
- Exclude incomplete anatomical slices
- Identify valid slice range per patient

This reduces noisy measurements.

---

### 2. Bone Distance Measurement

For each valid slice:

- Extract femur and tibia boundaries from binary masks
- Compute vertical inter-bone distance
- Evaluate multiple strategies:
  - Center-line distance
  - Multi-point averaged distance
  - Region-based averaged distances

Final approach:
- Divide joint space into equal regions
- Compute average distance per region
- Concatenate into fixed-length feature vector

---

### 3. Feature Vector Construction

Each case is represented as:


Where:
- `di` = averaged inter-bone distance from a region/slice
- All cases share identical vector length

---

### 4. Machine Learning Models

Models evaluated:

- Support Vector Machine (SVM)
- Random Forest
- Multilayer Perceptron (MLP)
- Gradient Boosting (LightGBM)
- Residual Neural Network with Attention Block (RNNAB)

**Evaluation method:**  
10-fold cross-validation

**Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

**Best observed performance:**
- F1 ≈ 0.76  
- AUC ≈ 0.76  

---

## Repository Structure
Beautiful. You’re building this the right way — clean structure first, then clarity.

Below is the finished continuation starting from ## Repository Structure, fully formatted for Markdown and consistent with what you’ve already written.

You can paste this directly under your current content.

## Repository Structure



knee-oa-bone-distance-cv/
├── data/
│ ├── masks/ # Binary femur + tibia masks
│ ├── labels.csv # KL grade / OA labels
│ └── README.md # Data usage notes
│
├── preprocessing/
│ ├── slice_selection.py # Valid slice range detection
│ └── distance_measurement.py # Inter-bone distance algorithms
│
├── feature_extraction/
│ └── build_feature_vectors.py # Construct fixed-length vectors
│
├── models/
│ ├── train_svm.py
│ ├── train_rf.py
│ ├── train_mlp.py
│ ├── train_lightgbm.py
│ └── train_rnnab.py
│
├── evaluation/
│ └── cross_validation.py # 10-fold evaluation pipeline
│
├── results/
│ ├── metrics_summary.csv
│ └── model_comparison.md
│
├── requirements.txt
└── README.md
