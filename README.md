# 🌾 Wheat Disease Detection

A deep learning project that classifies **4 wheat disease categories** from leaf images using transfer learning — benchmarking VGG19, ResNet50, ResNet101, InceptionV3, and an improved ResNet variant with TensorFlow/Keras.

---

## 🦠 Disease Categories

| Class | Description |
|---|---|
| **Fusarium Head Blight** | Fungal infection causing bleached spikelets |
| **Leaf Rust** | Orange pustules on leaf surface |
| **Tan Spot** | Tan/brown necrotic lesions |
| **Healthy Wheat** | No visible disease symptoms |

---

## 🧠 Models Benchmarked

| # | Model | Backbone | Framework |
|---|---|---|---|
| 1 | VGG19 | 19-layer CNN (Oxford) | TensorFlow/Keras |
| 2 | ResNet50 | 50-layer Residual Network | TensorFlow/Keras |
| 3 | InceptionV3 | Inception module architecture | TensorFlow/Keras |
| 4 | ResNet101 | 101-layer Residual Network | TensorFlow/Keras |
| 5 | ResNet Improved | Fine-tuned ResNet variant | TensorFlow/Keras |

All models pretrained on ImageNet and fine-tuned on the wheat disease dataset.

---

## 📁 Repository Structure

```
Wheat-Disease-Detection/
├── notebooks/
│   ├── 01_vgg19_wheat_disease.ipynb
│   ├── 02_resnet50_wheat_disease.ipynb
│   ├── 03_inceptionv3_wheat_disease.ipynb
│   ├── 04_resnet101_wheat_disease.ipynb
│   └── 05_resnet_improved_wheat_disease.ipynb
└── README.md
```

---

## ⚙️ Setup

```bash
pip install tensorflow keras scikit-learn imutils matplotlib numpy opencv-python
```

---

## 🔬 Methodology

1. **Data Loading** — Read images from class-labeled subdirectories
2. **Preprocessing** — Resize to 224×224, normalize with ImageNet mean subtraction
3. **Augmentation** — Rotation (±30°), zoom, shifts, horizontal flip
4. **Transfer Learning** — Freeze ImageNet backbone, train custom head
5. **Custom Head** — AveragePooling2D → Dense(512, ReLU) → Dropout(0.4) → Softmax(4)
6. **Training** — Adam (lr=1e-3), categorical cross-entropy, 50 epochs, batch 64
7. **Evaluation** — Accuracy, classification report, confusion matrix

---

## 📊 Results

| Model | Val Accuracy |
|---|---|
| VGG19 | — |
| ResNet50 | — |
| InceptionV3 | — |
| ResNet101 | — |
| ResNet Improved | — |

> Results to be updated after full training runs on the complete dataset.

---

## 🔗 References

- [Simonyan & Zisserman — VGGNet (2014)](https://arxiv.org/abs/1409.1556)
- [He et al. — Deep Residual Learning (2015)](https://arxiv.org/abs/1512.03385)
- [Szegedy et al. — InceptionV3 (2016)](https://arxiv.org/abs/1512.00567)

---

## 👤 Author

**Yohan Gala** — B.Tech Computer Engineering, KJSIT Mumbai  
[GitHub](https://github.com/Yohangala)
