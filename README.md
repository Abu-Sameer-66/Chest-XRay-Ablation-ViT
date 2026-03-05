<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=000000,434343,000000&height=250&section=header&text=Chest%20X-Ray%20Ablation%20Study&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Vision%20Transformers%20vs%20CNN%20Hybrids&descAlignY=60&descAlign=50" width="100%"/>
</div>

<br/>





<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-ViT_B/16-000000?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Domain-Medical_AI-00C853?style=for-the-badge&logo=health&logoColor=white" />
</div>

---

## 🔬 The Accuracy Paradox: Research Objective
Many public medical datasets yield 95%+ accuracies due to clean, binary clinical tagging. This empirical deep learning research project tests if standard State-of-the-Art (SOTA) architectures maintain their high accuracy when exposed to real-world, text-mined datasets containing overlapping diseases (e.g., a patient having both Cardiomegaly and Pneumonia).



---

## 🧬 The Ablation Study Sequence
We systematically engineered and trained five architectures from scratch to observe the evolution of spatial awareness and feature extraction:

* **ANN Baseline:** Proved the loss of spatial hierarchies via image flattening.
* **CNN Standard:** Introduced spatial convolutions with Global Average Pooling (GAP) for parameter efficiency.
* **HAM (Hierarchical Attention Mechanism):** Implemented custom spatial attention maps. Proved that attention mechanisms without pre-training struggle to generalize on small medical datasets.
* **Vision Transformer (ViT-B/16):** Applied transfer learning. Achieved rapid training convergence but exposed the dataset's labeling contradictions under standard Cross-Entropy Loss.
* **Custom Hybrid (ResNet18 + Transformer Encoder):** Fused local spatial edge detection (CNN) with global context processing (Transformer sequence modeling).

---

## 🚧 The "Data Wall" & Multi-Label Engineering
[cite_start]During evaluation, the ViT achieved a near 0.000 training loss but capped at ~65.8% validation accuracy. A custom diagnostic script revealed that a massive percentage of minority class labels (like Cardiomegaly) contained overlapping diseases. 

To resolve the mathematical deadlock of Softmax/Cross-Entropy forcing mutually exclusive predictions, the pipeline was upgraded to a **Multi-Label Architecture using BCEWithLogitsLoss**, allowing independent probability predictions across all classes. 

---

## 📊 Empirical Results Summary
Despite the SOTA Multi-Label approach, the dataset's inherent noise created a mathematical ceiling. Below is the comparative telemetry from the evaluation pipeline:

| Architecture | Val Accuracy | Precision | Recall | Macro F1 |
| :--- | :--- | :--- | :--- | :--- |
| **ANN Baseline** | [cite_start]21.4%  | [cite_start]0.22  | [cite_start]0.21  | [cite_start]0.12  |
| **CNN Standard** | [cite_start]49.6%  | [cite_start]0.45  | [cite_start]0.50  | [cite_start]0.28  |
| **HAM** | [cite_start]32.8%  | [cite_start]0.32  | [cite_start]0.33  | [cite_start]0.22  |
| **ViT-B/16 (SOTA)** | [cite_start]65.8%  | [cite_start]0.65  | [cite_start]0.66  | [cite_start]0.44  |
| **Custom Hybrid** | [cite_start]60.4%  | [cite_start]0.58  | [cite_start]0.60  | [cite_start]0.42  |
| **ViT (Multi-Label)** | [cite_start]60.4%  | [cite_start]0.63  | [cite_start]0.60  | [cite_start]0.48  |

*This empirically proves the "Accuracy Paradox": on heavily noisy, text-mined clinical datasets, SOTA models hit a ceiling due to data contradictions, unlike clean binary datasets where 95%+ is common.*

---

## 🛠️ Tech Stack & MLOps Architecture
* **Framework:** PyTorch
* **Compute:** Distributed Data Parallel evaluation
* **Optimization:** Automatic Mixed Precision (AMP), Cosine Annealing LR Scheduler, Weighted Cross-Entropy Loss (for extreme class imbalance).
* **Automation:** Custom Object-Oriented `ResearchEvaluator` class for automated metric logging and dynamic Confusion Matrix generation.
