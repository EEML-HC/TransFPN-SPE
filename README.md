# Cross-Domain Few-Shot Detection of Power Equipment Using TransFPN: A Feature Pyramid Network Approach

This repository provides the official implementation of **TransFPN**, a transfer-learning-based framework for **Cross-Domain Few-Shot Object Detection (CD-FSOD)** in power equipment inspection scenarios.

The framework integrates a **Vision Transformer (ViT) encoder–decoder** with a **Feature Pyramid Network (FPN)** to improve cross-domain feature transferability and multi-scale detection capability under limited labeled data conditions.

This code corresponds to the research paper:

**Cross-Domain Few-Shot Detection of Power Equipment Using TransFPN: A Feature Pyramid Network Approach**

---

# 1. Repository Structure
```
TransFPN-SPE  
│  
├── annotations/ # Dataset annotations  
├── few_shot/ # Few-shot sampling configuration  
├── mmdet/ # Detection framework (based on MMDetection)  
│  
├── SPE Dataset.zip # Substation Power Equipment dataset  
│  
├── box_10shot_*.txt # Few-shot training splits  
│  
├── train.txt # Training image list  
├── test.txt # Testing image list  
│  
├── requirements.txt # Python dependencies  
├── environment.yml # Conda environment configuration  
│  
└── README.md  
```
---

# 2. Environment and Requirements

## Hardware

The experiments were conducted on a workstation with the following configuration:

- **GPU:** 2 × NVIDIA TITAN X (24GB VRAM)
- **CPU:** Intel Core i9-12900K
- **Memory:** 128 GB RAM

---

## Software

The experiments were conducted with the following software environment:

- Python = 3.7
- PyTorch = 1.7
- CUDA = 10.2
- MMDetection

---

## Install Dependencies

### Option 1 (recommended)
```
conda env create -f environment.yml
```

### Option 2
```
pip install -r requirements.txt
```

---

# 3. Dataset

## SPE Dataset (Substation Power Equipment)

This project provides the **SPE dataset**, designed for evaluating cross-domain few-shot object detection in power inspection scenarios.

The dataset includes several common substation equipment categories:

- Arrester
- Circuit Breaker
- Current Transformer
- Isolating Switch
- Voltage Transformer

Few-shot learning settings are provided with:

- 1-shot
- 2-shot
- 3-shot
- 5-shot
- 10-shot

Few-shot split files are located in:
```
box_10shot_*.txt
```

---

# 4. Key Algorithm

## TransFPN Framework

TransFPN addresses **cross-domain few-shot detection**, where:

- Source domain: natural image datasets
- Target domain: power equipment inspection images

The framework consists of three main components.

---

### 1. Pre-trained Transformer Encoder

A **Vision Transformer (ViT)** encoder pre-trained using **Masked Autoencoder (MAE)** is used to extract global feature representations.

Advantages:

- Strong semantic representation
- Improved cross-domain feature transferability

---

### 2. Transformer-Compatible Feature Pyramid Network

Unlike CNN backbones, Transformer layers produce feature maps with the same spatial resolution.

To address this issue, we design a **Transformer-compatible FPN** that:

- Constructs multi-scale feature maps
- Preserves spatial details
- Improves small object detection

Multi-scale features are generated using:

- Upsampling
- Downsampling
- Feature fusion

---

### 3. Multi-scale RoI Feature Modulation

To better utilize pyramid features, we introduce a **RoI feature modulation mechanism**:

$$
F = F_{ss} + \alpha F_{ms}
$$

where

- `F_ss` : RoI aligned proposal feature
- `F_ms` : multi-scale pyramid feature
- `α` : learnable weight

This improves robustness under cross-domain few-shot settings.

---

# 5. Training

Prepare dataset paths and configuration files, then run:

```
tools/dist_train.sh few_shot/voc_fin_split1_10shots.py
```
---

# 6. Testing

Evaluate the trained model:
tools/dist_test.sh "path/to/config/file.py" "path/to/trained/weights.pth" 1 --eval bbox
```
tools/dist_test.sh few_shot/voc_fin_split1_10shots.py work_dirs/voc_fin_split1_10shots/latest.pth --eval bbox
```
---

# 7. Code and Data Availability

To improve research transparency and reproducibility, the source code and dataset are publicly available:
https://github.com/EEML-HC/TransFPN-SPE

# 8. Citation

If you use this code or dataset in your research, please cite:
