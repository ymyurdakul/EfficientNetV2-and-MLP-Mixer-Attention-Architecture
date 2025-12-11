# 🚀 EfficientNetV2-S + Linear-Attention MLP-Mixer Hybrid

A lightweight, high-performance hybrid architecture that combines:

* **EfficientNetV2-S backbone** 🧱
* **Linear-Attention MLP-Mixer head** 🧠⚡

Designed for medical image classification (e.g., **brain tumor MRI**), but fully generalizable to any image classification task.

---

## 🌟 Features

### 🧩 Architecture Highlights

* EfficientNetV2-S for hierarchical convolutional feature extraction
* Linear Attention for efficient global representation learning
* MLP-Mixer token + channel mixing enhanced with attention
* Fully configurable mixer depth and dimensions
* Optional ImageNet-pretrained backbone

### 📦 Provided Components

* EfficientNetV2-S implementation
* Linear Attention module
* Mixer block with token + channel mixing
* Hybrid model factory using `@register_model`
* Ready-to-use training structure

---

## 📁 Recommended Project Structure

```
project/
│
├── models/
│   ├── efficientnetv2_s_backbone.py
│   ├── mixer_attention_head.py
│   ├── hybrid_effnetv2s_mixer.py
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── README.md
└── requirements.txt
```

---

## 📚 Dependencies (Extensive)

Below is a **complete and expanded list** of all dependencies typically needed for development, training, evaluation, visualization, and deployment.

### 🔧 Core Deep Learning Stack

```
tensorflow>=2.9.0
keras>=2.9.0
tensorflow-addons>=0.20.0
keras-cv-attention-models>=1.3.0
```

### 🔬 Scientific & Numerical

```
numpy>=1.22
scipy>=1.7
opencv-python>=4.5
scikit-image>=0.19
scikit-learn>=1.0
numba>=0.55
pandas>=1.3
```

### 📊 Visualization & Logging

```
matplotlib>=3.5
seaborn>=0.11
tensorboard>=2.9
tensorboardX>=2.5
plotly>=5.0
```

### ⚙️ Performance / Hardware

```
tensorflow-probability>=0.16
ml-dtypes>=0.2
protobuf>=3.19
```

### 🖥️ System Utilities

```
tqdm>=4.64
PyYAML>=6.0
psutil>=5.8
```

### 💻 Optional GPU Acceleration

```
nvidia-cudnn>=8.4
nvidia-cublas>=11.6
nvidia-cuda-toolkit>=11.6
```

👉 Check GPU availability:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---




---

## ▶️ Installation

### 🅰️ Standard Python Environment

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 🅱️ Conda Environment

```bash
conda create -n effnet_mixer python=3.9
conda activate effnet_mixer
pip install -r requirements.txt
```

---

## 🚀 Using the Model

```python
from models.hybrid_effnetv2s_mixer import EfficientNetV2S_MLPMixerAttentionSmall

model = EfficientNetV2S_MLPMixerAttentionSmall(
    input_shape=(384, 384, 3),
    num_classes=3,
    mixer_tokens_mlp_dim=128,
    mixer_channels_mlp_dim=512,
    mixer_blocks=4,
)

model.summary()
```

---


---

## ❤️ Credits

Developed as a hybrid architecture combining ConvNets + Mixer models for efficient image understanding.

For improvements, suggestions, or extended architectures, feel free to update or extend this file.

---

**Enjoy building with EfficientNetV2-S + Linear Attention MLP-Mixer!** 🎉
