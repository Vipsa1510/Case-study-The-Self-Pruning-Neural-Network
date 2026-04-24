
#  Case Study – Self-Pruning Neural Network

##  

**Problem Title:** The Self-Pruning Neural Network

---

##  Preamble

In real-world scenarios, deploying large neural networks is challenging due to memory and computational constraints. Model pruning is a common technique to reduce model size by removing less important weights.

This project implements a **self-pruning neural network**, where the model learns to prune its own weights during training using a gating mechanism and sparsity regularization.

---

##  The Core Problem

The objective is to design a neural network that can:

* Learn to identify unimportant weights
* Dynamically reduce their influence
* Achieve a balance between **accuracy and sparsity**

This is done by associating each weight with a learnable **gate parameter**.

---

##  Methodology

###  Prunable Linear Layer

A custom `PrunableLinear` layer is implemented where:

* Each weight has a corresponding **gate score**
* Gates are computed using a **sigmoid function**
* Effective weights = `weight × gate`

---

###  Sparsity Regularization

The loss function is defined as:

> **Total Loss = Classification Loss + λ × Sparsity Loss**

* Sparsity loss is computed using **L1 norm of gate values**
* Encourages gates to move toward **zero**, enabling pruning

---

###  Training Strategy

* Model trained on **CIFAR-10 dataset**
* Multiple values of λ tested to observe trade-off
* Metrics tracked:

  * Test Accuracy
  * Sparsity Level (%)

---

##  Results

| Lambda | Test Accuracy | Sparsity (%) |
| ------ | ------------- | ------------ |
| 0.005  | 53.09%        | 1.71%        |
| 0.01   | 51.95%        | 1.72%        |
| 0.02   | 50.94%        | 1.71%        |

###  Observation

* Increasing λ slightly reduces accuracy
* Sparsity remains low (~1.7%), indicating weak pruning
* Model prefers **soft reduction of weights** rather than full pruning

---

##  Gate Distribution Analysis

The distribution of gate values shows most values concentrated in the range **0 to 0.1**, with an average of ~0.056.

This indicates:

* Many weights are being reduced in importance
* Very few weights are fully pruned (close to zero)

---

##  Conclusion

The self-pruning network successfully learns the classification task while demonstrating the effect of sparsity regularization.

Although sparsity is relatively low, the results clearly show the **trade-off between accuracy and pruning strength**. The model reduces weight importance but does not strongly eliminate connections, indicating scope for stronger pruning techniques.

---

##  Future Work

* Apply stronger sparsity regularization (higher λ)
* Use sharper gating mechanisms for more effective pruning
* Explore CNN-based architectures for improved accuracy
* Apply hard pruning after training for better compression

---

##  Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

---

##  How to Run

```bash
pip install torch torchvision matplotlib
```

Run the notebook:

```bash
Self_Pruning_Neural_Network.ipynb
```
---

#  Key Takeaways

* Implemented **self-pruning mechanism from scratch**
* Built a **dynamic pruning system during training**



#  Final Note

This project focuses on understanding and implementing **adaptive model pruning**. While sparsity achieved is modest, the approach validates the concept and highlights directions for improvement in practical scenarios.


