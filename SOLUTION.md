
# Solution Report

## 1. Reproducibility Instructions
Experiments were done by using Google Colab with T4.


### Commands to run in Google Colab
```bash
!git clone https://github.com/Humpty1944/SMILES-2026-Hallucination-Detection.git
%cd SMILES-2026-Hallucination-Detection
!pip install --no-cache-dir -r requirements.txt
!python solution.py
```

---

## 2. Final Solution Description

### Component Overview

- **`aggregation.py`**: Extracts representations from middle layers (12–16): last-token embedding and mean of last 3 tokens. Two geometric features are appended: representation drift (mean pairwise layer distance) and signal intensity (L2-norm of centroid).

- **`splitting.py`**: `StratifiedKFold(n_splits=5)` for stable, class-balanced evaluation on small data.

- **`probe.py`**: `sklearn` Logistic Regression with strong L2 regularization - chosen over neural nets to avoid overfitting .

- **`solution.py`**: `USE_GEOMETRIC = True`.

---

### aggregation.py details

**`aggregate`**  
Mid-layers (12–16) best capture semantic integration for hallucination detection [[source](https://huggingface.co/blog/krogoldAI/llm-hallucination-detection)].  
- Per layer: extract last-token embedding and mean of last 3 tokens.  
- Average vectors across the selected layers.  


**`extract_geometric_features`**  
Stack last-token vectors from target layers.  
1. **Mean pairwise Euclidean distance between layer vectors**. A high mean distance between layer vectors indicates instability, suggesting that the model may be losing grounding in the input data and becoming more prone to hallucinations. 
2. **L2-norm of layer-vector**. An anomalously high or low L2-norm of the layer-vector reflects extreme states of model confidence: either overconfidence in a fabricated response or signal attenuation due to a lack of relevant knowledge.

---

### probe.py details

**Logistic Regression**  
Small dataset  with high-dimensional features makes complex models overfit.  
- L2 regularization controls overfitting directly.  
- Makes better generalization on limited data.  

---

### splitting.py details

**Stratified 5-Fold CV**  
Small data requires reliable, low-variance metric estimation.  
- `StratifiedKFold` preserves class ratio in each fold, ensuring unbiased performance estimates.   

---

## 3. Failed Attempts

| Idea / Approach | Implementation Details (Simplified) | Why Discarded or Limited |
|----------------|-------------------------------------|--------------------------|
| **Multi-Layer Concatenation** | Combined hidden states from all 24 layers or selected groups  | Severe overfitting |
| **More Geometric Features** | Extracted norms, distances, EigenScores, drift metrics (6–19 features) | Overfitting; high feature dimension|
| **Pooling across Entire Input** | Pooling/feature calculation across entire sequence | Hallucination signals only appear in response embeddings; No significant gain |
| **Complex NN** | Added Dropout, BatchNorm, AdamW, weight decay | Overfitting |
| **Complex Pooling Strategies** | Attention-weighted pooling, max-pooling, combinations | Add complexity without performance gain |
| **PCA** | Compressed 900 features to 64 components | No improvements; The dataset is too small to gain something usefull from compression |
| **1-Fold CV** | Single train/validation/test split| While sometimes showing better metrics, results were not stable across runs |

### Summary
1.  Hallucination signals are localized
2.  Simplicity wins on small data.
3.  Geometric features must be minimal.
