# HA MOACO: A Structure-Aware Graph RAG Framework for Small Language Models

This repository provides the official implementation of **HA-MOACO**, a training-free, structure-aware Graph RAG framework designed for efficient multi-hop reasoning in Small Language Models (SLMs). 
## 📖 Overview
HA MOACO leverages a novel **Ant Colony Optimization (ACO)** mechanism to perform joint inference-time optimization of reasoning paths over heterogeneous knowledge graphs. By balancing semantic relevance, factual grounding, and token budget constraints, the framework enables SLMs to achieve high-performance reasoning without the need for additional fine-tuning.

## 📁 Repository Structure
- `src/`: Core implementation of the HA MOACO framework.
- `prompts/`: Configuration file (`prompts.yaml`) containing all prompt templates.
- `data/`: Scripts for graph construction and preprocessing (`data_prep.py`).
- `experiments/`: Configuration files and instructions for reproducing experimental results (Table 4 & 5).

## 🚀 Getting Started
### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- `transformers`, `networkx`, `numpy`

### Installation
```bash
git clone https://github.com/analist141/HA-MOACO.git
cd HA-MOACO
pip install -r requirements.txt
🛠 Reproducibility
To ensure full reproducibility of the results reported in the paper:

Ensure the random seed is set to 42 (configured in config.yaml).
Use the provided dataset splits (BioASQ 8b and HotpotQA distractor setting).
Execute the pipeline using:

content_copy
bash
