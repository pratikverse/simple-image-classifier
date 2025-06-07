
# 🧠 Simple Image Classifier using Logistic Regression

A lightweight and interpretable image classification project built from scratch using **Logistic Regression**. This repository demonstrates core machine learning concepts without relying on high-level libraries like TensorFlow or PyTorch. The implementation emphasizes clarity, educational value, and performance benchmarking on a small-scale binary classification dataset.

---

## 📌 Overview

This project classifies images (e.g., **cat vs. non-cat**) using a **custom-built logistic regression model**. It includes all steps of a machine learning pipeline:

- Dataset loading & preprocessing (from `.h5` files)
- Logistic regression model built from scratch
- Forward and backward propagation
- Cost function and optimization using gradient descent
- Accuracy evaluation and prediction on new examples
- Visualization of results and cost function convergence

---

## 🧰 Tech Stack

| Component        | Technology         |
|------------------|--------------------|
| Language         | Python 3.x         |
| IDE/Notebook     | Jupyter Notebook   |
| Libraries        | NumPy, Matplotlib, h5py, SciPy |

> 📁 **Note:** All dependencies are listed in [`requirements.txt`](./requirements.txt)

---

## 📂 Dataset

The dataset is binary-labeled and consists of RGB images stored in HDF5 format.

- **Files:**  
  - `train_catvnoncat.h5`  
  - `test_catvnoncat.h5`
- **Format:**  
  - Shape: (m, 64, 64, 3) — where `m` is the number of images  
  - Labels: 1 = Cat, 0 = Non-cat

Make sure the dataset files are placed in the same directory as the notebook or provide the correct path when loading.

---

## 🚀 Getting Started

Follow the steps below to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/pratikverse/simple-image-classifier.git
cd simple-logistic-image-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook logsitic-model.ipynb
```

Open the notebook and run the cells sequentially to:

- Load and visualize the dataset
- Initialize parameters
- Train the logistic regression model
- Evaluate performance
- Test custom image input (optional)

---

## 📈 Performance

| Metric              | Value (Approx.) |
|---------------------|-----------------|
| Training Accuracy   | ~99%            |
| Test Accuracy       | ~70%            |

> ⚠️ This performance highlights the model’s limitations in generalization due to overfitting. Further improvements can be implemented using more advanced techniques.

---

## 📊 Visualization

Key plots and results available in the notebook:

- Cost function vs. Iterations
- Correct vs. Misclassified Predictions
- Training/Test set performance

---

## 🛠️ Future Improvements

- ✅ Add regularization (L2)
- ✅ Modularize training into utility functions
- 🔄 Extend to multi-class classification
- 🔄 Convert to neural network-based approach
- 🔄 Benchmark with `scikit-learn` and other classical ML models
- 🔄 Enable image upload via CLI/GUI for real-time prediction

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch (`feature/your-feature-name`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 👤 Author

**Pratik Shrivastava**

- [GitHub](https://github.com/pratikverse)
- [LinkedIn](https://www.linkedin.com/in/pratikshrivastava19/)

---

## 🙏 Acknowledgements

This project is inspired by Andrew Ng’s Deep Learning Specialization and similar educational projects in machine learning fundamentals.
