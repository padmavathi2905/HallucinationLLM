# Graph-Based Relational Modeling for Hallucination Detection in Large Language Models

## Overview

Large Language Models (LLMs) often generate fluent but factually incorrect responses, known as hallucinations. This project proposes a **graph-based relational reasoning framework** to detect hallucinations by explicitly modeling relationships between claims, evidence, and entities.

We implement and compare two approaches:

- **Flat similarity-based baseline** using Sentence-BERT embeddings and cosine similarity  
- **Graph-based relational model** using a heterogeneous multi-relational graph with a Relational Graph Convolutional Network (RGCN)

The system is trained on the **FEVER dataset** and evaluated on both FEVER and **TruthfulQA** to test generalization.

---

## Repository Structure

```
.
├── AI_Project_Code.ipynb   # Main implementation (ENTRY POINT)
├── README.md               # Documentation
├── requirements.txt        # Dependencies
└── outputs/                # (Optional) results and plots
```

---

## Entry Point

The entire pipeline is implemented in:

AI_Project_Code.ipynb

This notebook includes:
- Data preprocessing
- Feature extraction
- Graph construction
- Model training
- Evaluation and visualization

---

## System Requirements

- Python 3.9+
- pip
- Internet connection (for datasets)
- Minimum 8GB RAM
- Recommended: GPU (for faster training)

---

## Dependencies

Install all dependencies:

```
pip install -r requirements.txt
```

requirements.txt:

```
torch
torchvision
torchaudio
torch-geometric
sentence-transformers
scikit-learn
spacy
datasets
matplotlib
pandas
```

Download spaCy model:

```
python -m spacy download en_core_web_sm
```

---

## Step-by-Step Execution

### 1. Clone repository

```
git clone <your-repo-link>
cd <repo-name>
```

### 2. Create virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Launch notebook

```
jupyter notebook
```

### 5. Run project

- Open `AI_Project_Code.ipynb`
- Run all cells sequentially from top to bottom
- Do not skip preprocessing or training steps

---

## Alternative: Google Colab

1. Upload notebook to Colab  
2. Enable GPU (Runtime → Change runtime type)  
3. Run all cells  

---

## Methodology

### Problem Formulation

Hallucination detection is framed as a **binary classification task**:

- 0 → Non-hallucinated (Supported)  
- 1 → Hallucinated (Refuted)  

---

### Dataset

**Training Dataset: FEVER**
- SUPPORTED → label 0  
- REFUTED → label 1  
- NOT ENOUGH INFO removed  

**Split:**
- 70% training  
- 15% validation  
- 15% testing  

**Evaluation Dataset:**
- TruthfulQA (out-of-domain generalization)

---

### Flat Similarity-Based Model

- Sentence-BERT (`all-MiniLM-L6-v2`)
- 384-dimensional embeddings
- Cosine similarity between claim and evidence
- Threshold-based classification
- Does not model relationships

---

### Graph-Based Relational Model

#### Graph Construction

- **Nodes:** claim, evidence, entities  
- **Edges:**
  - Claim ↔ Evidence  
  - Claim ↔ Entity  
  - Evidence ↔ Entity  

Captures relational dependencies beyond text similarity.

---

#### Model Architecture

- 2-layer Relational Graph Convolutional Network (RGCN)
- Hidden dimension: 256  
- Dropout: 0.3  
- ReLU activation  
- Sigmoid output for classification  

---

#### Training Configuration

- Optimizer: Adam  
- Learning rate: 0.001  
- Batch size: 16  
- Loss: Weighted Binary Cross Entropy  
- Model selection based on validation F1 score  

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score (primary metric)  

F1-score is emphasized due to class imbalance.

---

## Results Summary

- Flat model achieves high precision but low recall  
- Graph-based model improves recall and F1-score  
- Demonstrates better detection of relational inconsistencies  
- Shows improved performance in reasoning-heavy scenarios  

---

## Expected Output

After execution, the notebook generates:

- Performance metrics (Accuracy, Precision, Recall, F1)  
- Comparison between flat and graph-based models  
- Visualization plots  
- Evaluation results on:
  - FEVER dataset  
  - TruthfulQA dataset  

---

## Reproducibility

To reproduce results:

1. Follow installation steps exactly  
2. Run notebook sequentially  
3. Ensure internet access for dataset download  
4. Do not modify pipeline steps  

All hyperparameters are explicitly defined.  
Minor variation may occur due to randomness.

---

## Limitations

- Simplified evidence construction for TruthfulQA  
- No external retrieval system  
- Graph size limited for efficiency  
- Performance depends on entity extraction quality  

---

## Future Work

- Integrate retrieval-based evidence grounding  
- Improve relation typing using NLI  
- Extend to multi-sentence reasoning  
- Evaluate on larger hallucination benchmarks  
- Explore scalable graph architectures  

---

## Conclusion

This project demonstrates that hallucination detection is fundamentally a relational reasoning problem. By modeling structured relationships between claims, evidence, and entities, the graph-based approach captures inconsistencies that flat similarity methods fail to detect, leading to improved performance in complex reasoning scenarios.

---
