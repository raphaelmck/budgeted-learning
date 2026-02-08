# How Should Compute Be Spent? Model Size, Training Time, and Ensembles under Fixed Budgets

## Abstract

Modern neural network scaling studies typically assume training to convergence, obscuring how model size should be chosen under limited compute budgets. In this work, we empirically study compute-constrained training by fixing a training budget defined as the product of model parameters and optimization steps. Using controlled experiments on CIFAR-10, we investigate how performance depends on allocating compute across model size, training duration, and multiplicity. We observe a clear size–compute crossover: smaller models outperform larger ones under tight budgets, while larger models dominate only once sufficiently trained. Crucially, we find that ensembles of small models can outperform all single-model alternatives under the same total compute, achieving higher accuracy, lower variance, and substantially improved worst-case performance. These results suggest that compute allocation is a critical and underexplored design dimension, with ensembles offering a robust strategy in compute-limited regimes.

## 1. Introduction

Training modern neural networks is increasingly constrained by finite computational resources. In many practical settings—ranging from academic research and small-scale experimentation to latency-sensitive or energy-constrained applications—models cannot be trained to full convergence. Instead, practitioners must decide how to allocate a limited training compute budget across competing dimensions such as model size, training duration, and the number of models trained.

Most existing neural scaling studies focus on asymptotic regimes in which models are trained until convergence, revealing predictable relationships between model size, dataset size, and performance. While these results have guided the design of large-scale systems, they provide limited guidance in compute-constrained regimes, where models are trained for only a small number of optimization steps. In such settings, it is unclear whether compute is best spent on larger models trained briefly, smaller models trained longer, or multiple models trained in parallel.

In this work, we study this question empirically by fixing a training compute budget defined as the product of the number of trainable parameters and the number of optimization steps. This budget captures the dominant cost of training in many settings and enables controlled comparisons between different compute allocation strategies. Using simple convolutional architectures on CIFAR-10, we systematically vary model width, training duration, and ensemble size while holding total training compute constant.

Our experiments reveal two consistent phenomena. First, we observe a clear size–compute crossover: under tight compute budgets, smaller models outperform larger ones due to receiving more optimization steps, while larger models only dominate once sufficient compute is available to train them effectively. Second, we find that ensembles of small models can substantially outperform single-model alternatives under the same total compute budget. These ensembles achieve higher mean accuracy, reduced variance across random seeds, and improved worst-case performance, despite using no additional training compute.

Together, these results highlight compute allocation as a critical design dimension in neural network training. Rather than viewing ensembles solely as a regularization or variance-reduction technique, our findings suggest they can serve as an effective strategy for deploying limited compute. More broadly, our work emphasizes the importance of studying training dynamics in finite-compute regimes that more closely reflect practical constraints.

Contributions.
The main contributions of this work are:

We empirically characterize a size–compute crossover in compute-constrained training, showing when smaller models outperform larger ones and when this relationship reverses.

We demonstrate that ensembles of small models outperform single models of any size under fixed training compute, achieving higher accuracy and lower variance.

We provide a simple experimental framework for studying compute allocation strategies in neural network training.

## 2. Related Work

## 3. Experimental Setup

We evaluate compute allocation strategies using controlled experiments on image classification, varying model size, training duration, and ensemble size while holding total training compute fixed.

### 3.1 Compute Budget Definition

We define the training compute budget as

\[ C = P \times T \]

where $P$ is the number of trainable parameters in the model and $T$ is the number of stochastic gradient descent optimization steps. This quantity serves as a simple proxy for training-time compute, capturing the dominant cost of forward and backward passes during optimization.

All experiments are conducted under a fixed budget $C$, and different model configurations are compared by adjusting the number of training steps 

\[T = \left\lfloor \frac{C}{P} \right\rfloor \] 

To ensure meaningful training dynamics, we enforce a minimum number of optimization steps for all runs.

For ensemble experiments with $k$ members, the total budget $C$ is evenly divided across ensemble members, such that each model is trained with budget $C/k$ and the aggregate compute across the ensemble matches that of a single-model baseline.

### 3.2 Dataset

All experiments are conducted on the CIFAR-10 dataset, which consists of 50,000 training images and 10,000 test images across 10 classes. Images are normalized using standard per-channel mean and variance statistics. No data augmentation is applied in order to isolate the effects of compute allocation and optimization dynamics.

### 3.3 Model Architecture

We use a simple convolutional neural network architecture consisting of three convolutional layers with ReLU activations and batch normalization, followed by global average pooling and a linear classification head. Model capacity is controlled via a width multiplier $w$, which scales the number of channels in each convolutional layer proportionally.

This architecture is intentionally lightweight and standardized to reduce confounding architectural effects and focus on the interaction between model size and training compute. Across experiments, width multipliers $w \in \{ 0.5, 1.0, 2.0 \}$ are used, resulting in models with parameters counts ranging from approximately $2.4 \times 10^4$ to $3.7 \times 10^5$.

### 3.4 Training Protocol

Models are trained using stochastic gradient descent with momentum. We use a fixed learning rate and batch size across all experiments, with no learning rate scheduling. Each run is initialized with a different random seed, affecting both parameter initialization and data order.

Training proceeds for the number of optimization steps determined by the compute budget and model size. For ensemble experiments, each ensemble member is trained independently using a distinct random seed and its allocated share of the compute budget.

### 3.5 Evaluation and Ensembling

Model performance is evaluated on the CIFAR-10 test set using classification accuracy and cross-entropy loss. All reported results are averaged over three independent runs with different random seeds, and we report both mean performance and standard deviation to assess stability.

For ensemble models, predictions are combined by averaging the class probability outputs of individual ensemble members at inference time. Ensemble performance is evaluated using the same test metrics as single models, without additional fine-tuning or post-processing.

### 3.6 Implementation Details

All experiments are implemented in PyTorch and run on consumer-grade hardware. Training is performed using the Metal Performance Shaders (MPS) backend when available. To ensure reproducibility, all hyperparameters and random seeds are logged, and results are aggregated using a consistent evaluation pipeline.

## 4. Results

We present results demonstrating how performance depends on model size, training compute, and ensembling under fixed compute budgets. Unless otherwise stated, results are reported as mean test accuracy over three random seeds, with standard deviation used to assess stability.

### 4.1 Size–Compute Crossover
### 4.2 Ensembles under Fixed Compute

## 5. Discussion

## 6. Limitations

## 7. Conclusion

## References
