# Sobel-Gradient-MLP
Code and resources for "A Sobel-Gradient MLP Baseline for Handwritten Character Recognition"

# Sobel-Gradient MLP (MNIST & EMNIST Letters)

**Code and resources for:**  
*“A Sobel-Gradient MLP Baseline for Handwritten Character Recognition”* — Azam Nouri

---

## Abstract
We revisit whether **first-order edges alone** can support high-quality handwritten character recognition (HCR). A compact, fully-connected MLP consumes only the two Sobel derivatives \((G_x, G_y)\) of \(28\times28\) images. Each channel is min–max normalized per image, concatenated, and flattened to a 1,568-D vector. With **no convolutions and no augmentation**, the model attains **98.0%** (MNIST) and **92.0%** (EMNIST Letters) in a single run under a stratified 80/20 split of TFDS train+test, indicating that much of the class-discriminative structure of digits/letters resides in **where** edges occur and **in which direction** they change.

---

## TL;DR
- **Input:** only Sobel \(G_x, G_y\) (signed derivatives)  
- **Model:** 3-layer MLP (Dense+BN+ReLU+Dropout)  
- **No CNNs, no aug** — simple, interpretable baseline  
- **Results (single run):** MNIST 98.0%, EMNIST Letters 92.0%

---

## Results (single run; stratified 80/20 split)
| Model | MNIST | EMNIST Letters |
|---|---:|---:|
| Sobel-Gradient MLP (this work) | **98.0%** | **92.0%** |

> **Metric choice.** MNIST and EMNIST Letters have ~uniform per-class counts, and the split is stratified; **top-1 accuracy** is therefore appropriate as the primary metric.

---

## Reproducibility (high-level)
- **Data:** MNIST; EMNIST Letters (via **TensorFlow Datasets — TFDS**)
- **Split protocol:** Concatenate TFDS `train+test` → stratified 80/20 (`random_state=42`); Keras holds out 10% of the training portion for validation.
- **Preprocessing:** scale pixels to \([0,1]\) `float32`; compute Sobel \(G_x, G_y\) (`ksize=3`); per-image, per-channel min–max to \([0,1]\).
- **Architecture:** Dense(1024) → Dense(512) → Dense(256) → Dense(#classes), with BN, ReLU, Dropout(0.5/0.4/0.3).
- **Optimisation:** Adam (Keras defaults), batch=128, up to 50 epochs, EarlyStopping(patience=4, restore best), ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6).
- **Metric:** top-1 accuracy on the held-out 20% test split (single run).

---

## Notebooks
- [`notebooks/Sobel_Gradient_pipeline_MNIST.ipynb`](notebooks/Sobel_Gradient_pipeline_MNIST.ipynb)  
- [`notebooks/Sobel_Gradient_pipeline_EMNIST_Letters.ipynb`](notebooks/Sobel_Gradient_pipeline_EMNIST_Letters.ipynb)  
- [`notebooks/CNN_Baseline_EMNIST_Letters.ipynb`](notebooks/CNN_Baseline_EMNIST_Letters.ipynb)

---

## Environment & Installation
```bash
# Python 3.9+ recommended
pip install --upgrade pip

# Core deps
pip install tensorflow tensorflow-datasets opencv-python tqdm scikit-learn numpy

