---

# 📌 Assignment 2

## Hybrid Collaborative Filtering with Triple Gradient Boosting Ensemble

**Bayesian Statistics + SVD (Latent Factors) + Interaction Features**
**XGBoost + LightGBM + CatBoost Ensemble**

---

# 📖 Project Overview

This project implements a **hybrid collaborative filtering system** on the MovieLens-100K dataset to predict user ratings (1–5).

The model combines:

* Bayesian user & item statistics
* Latent factor modeling via Truncated SVD
* Explicit interaction features
* Triple gradient boosting ensemble
* Strict 5-fold cross-validation
* Zero data leakage design

The hyperparameters correspond to the **best Optuna trial (Trial 8)**.

---

# 🎯 Objective

Predict:

```
(user, item) → rating ∈ {1,2,3,4,5}
```

Using multi-class classification.

Evaluation metric:

```
Accuracy
```

---

# 🏗 Pipeline Architecture

```
Raw Data
   ↓
Bayesian User & Item Statistics (Train Only)
   ↓
SVD (8 Components) – Train Only
   ↓
Latent Interaction (Dot Product)
   ↓
Feature Matrix Construction
   ↓
Triple Gradient Boosting Training
   ↓
Probability Averaging
   ↓
5-Fold Accuracy Evaluation
```

---

# 🔒 No Data Leakage Guarantee

The pipeline ensures:

* Global mean computed only on training data
* User/item statistics computed only on training data
* SVD fitted only on training pivot matrix
* Test users/items not seen in training → zero latent vectors
* No metadata used
* No target leakage

Each fold is fully isolated.

---

# 🧠 Feature Engineering

## 1️⃣ Bayesian Smoothing

For user:

[
\hat{\mu}_u =
\frac{n_u \cdot \bar{r}_u + \alpha \cdot \mu}
{n_u + \alpha}
]

For item:

[
\hat{\mu}_i =
\frac{n_i \cdot \bar{r}_i + \alpha \cdot \mu}
{n_i + \alpha}
]

Where:

* ( \mu ) = global mean
* ( \alpha = 20 )

Additional features:

* Mean
* Count
* Standard deviation

---

## 2️⃣ Truncated SVD (k = 8)

Matrix factorization:

[
R \approx U_k \Sigma_k V_k^T
]

Extracted features:

* 8 user latent factors
* 8 item latent factors

---

## 3️⃣ Interaction Feature

[
\text{svd_dot} = U_u \cdot V_i
]

Captures collaborative compatibility between user and item.

---

# 📊 Final Feature Set

| Category    | Count  |
| ----------- | ------ |
| User Stats  | 3      |
| Item Stats  | 3      |
| User SVD    | 8      |
| Item SVD    | 8      |
| Interaction | 1      |
| **Total**   | **23** |

---

# 🤖 Models Used

### 1️⃣ XGBoost (GPU Enabled)

* Depth = 8
* 724 trees
* Multi-class softprob objective

### 2️⃣ LightGBM

* 680 trees
* Leaf-wise tree growth
* Histogram-based splitting

### 3️⃣ CatBoost

* 691 iterations
* Depth = 5
* Ordered boosting

---

# 🧮 Ensemble Strategy

Probability averaging:

[
P_{final} =
\frac{
P_{xgb} + P_{lgb} + P_{cat}
}{3}
]

Final prediction:

[
\hat{y} = \arg\max P_{final}
]

This reduces variance and improves generalization.

---

# 🔁 Cross-Validation Protocol

Predefined MovieLens splits:

```
u1.base / u1.test
u2.base / u2.test
u3.base / u3.test
u4.base / u4.test
u5.base / u5.test
```

Final accuracy:

[
\text{Average Accuracy} =
\frac{1}{5} \sum Acc_i
]

---

# 📈 Final Results

| Fold        | Accuracy   |
| ----------- | ---------- |
| Fold 1      | 0.4642     |
| Fold 2      | 0.4669     |
| Fold 3      | 0.4602     |
| Fold 4      | 0.4590     |
| Fold 5      | 0.4531     |
| **Average** | **0.4607** |

---

# 📊 Performance Interpretation

* Consistent performance across folds
* Low variance between folds
* Strong collaborative-only baseline
* No metadata used
* Fully leakage-safe

The model achieves **46.07% average classification accuracy** on exact rating prediction (5-class task), which is strong for pure collaborative filtering without deep learning or metadata.

---

# ⚙️ Requirements

```bash
pip install numpy pandas xgboost lightgbm catboost scikit-learn
```

---


Ensure dataset directory:

```
ml-100k/
    u1.base
    u1.test
    ...
    u5.base
    u5.test
```

---

# 🚀 Key Strengths

* Strict no-leakage implementation
* Bayesian shrinkage for sparsity handling
* Latent factor modeling
* Interaction feature engineering
* Triple ensemble diversity
* GPU acceleration
* Optuna-tuned hyperparameters
* Robust 5-fold evaluation

---

# 👤 Author

Vinayak Koli
Assignment 2 – Hybrid Collaborative Filtering System

---

                    
  
