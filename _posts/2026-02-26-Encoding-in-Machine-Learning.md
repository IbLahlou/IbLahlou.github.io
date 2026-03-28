---
title: "Encoding in Machine Learning: Designing Categorical Geometry"
date: 2026-02-26
categories: [Machine Learning, Feature Engineering]
tags: [encoding, ml-engineering, feature-design, modeling]
math: true
---

# Introduction

Before a model learns, before a loss is minimized, before a gradient flows, a quieter decision is made.

How do we represent the world?

A categorical variable looks innocent. A column of regions. Product IDs. User segments. Device types. We call them “features” as if they were naturally numeric, as if the model were merely waiting for them to be formatted correctly.

But a category is not a number.  
It is a partition.  
It is a declaration that some observations are equivalent under a certain abstraction.

The moment we transform it, we decide something far more consequential than data formatting. We decide whether two categories are neighbors or strangers. Whether they lie on a line or float in orthogonal isolation. Whether identity matters more than frequency. Whether behavior matters more than structure. Whether similarity is imposed or allowed to emerge.

And once we decide, the model never questions that choice.

What is distance between “France” and “Germany”?  
Should “Premium” be twice “Standard”?  
Is rarity itself meaningful, or only correlation with outcome?  
When we collapse identity into expectation, are we modeling behavior    or leaking it?  
When we embed categories in dense vectors, are we discovering structure    or inventing it?

Only at the end do we name the act: to **encode**  from Latin _in-_ (“into”) and _codex_ (“a book of rules”)   is to inscribe something into a system. In machine learning, it is the transformation of information into numerical form so a model can process it.

Encoding is the first act of modeling.  
It is where epistemology becomes geometry.

Everything that follows    bias, variance, generalization, fairness, stability    is downstream of that act.

# 1. Categorical Encoding: From Set Theory to Metric Geometry

A categorical variable is, fundamentally, a finite set.  
It partitions observations into equivalence classes without prescribing any topology or metric structure.

Machine learning models, however, do not operate on sets. They operate on vectors in $\mathbb{R}^k$. Optimization requires inner products, gradients, norms, and distances. The act of encoding is therefore not symbolic translation   it is geometric imposition.

Formally, encoding defines a mapping

$$
f : \mathcal{C} \rightarrow \mathbb{R}^k
$$

This function endows the discrete set $\mathcal{C}$ with geometry.  
It determines adjacency, distance, orientation, and dimensionality.

Once $f$ is applied, the model no longer interacts with $\mathcal{C}$.  
It interacts with the induced embedding $f(\mathcal{C})$.

The properties of this embedding constrain the hypothesis space.


> Encoding determines whether categories become collinear scalars, orthogonal axes, empirical expectations, or learned manifolds.
{: .prompt-info }
---

## 1.1 Structural Taxonomy of Encoders

All categorical encoders can be classified by the geometry they impose.

| Encoding Class | Geometric Form | Structural Assumption |
|---------------|----------------|-----------------------|
| Label | 1D linear axis | Total ordering exists |
| One-Hot | Orthogonal basis | Categories independent |
| Ordinal | Ordered scalar | Rank + uniform spacing |
| Target | Scalar statistic | Predictive behavior defines identity |
| Frequency | Density scalar | Prevalence carries signal |
| Embedding | Learned manifold | Similarity emerges from optimization |

This taxonomy is geometric, not procedural.

---

# 2. Label Encoding: Linear Embedding of a Nominal Set

Label encoding maps each category to an integer:

$$
\{c_1, \dots, c_k\} \mapsto \{0, 1, \dots, k-1\}
$$

The transformation embeds a non-ordered set into a one-dimensional Euclidean space.  
The induced metric is absolute difference:

$$
d(c_i, c_j) = |f(c_i) - f(c_j)|
$$

This metric implies uniform spacing and total order.

### 2.1 Interaction with Linear Models

Consider a linear hypothesis:

$$
\hat{y} = \beta_0 + \beta_1 f(c)
$$

The model assumes monotonic variation of $\hat{y}$ with respect to encoded category index.  
The slope $\beta_1$ enforces directional structure.

If $\mathcal{C}$ is nominal, this is structural hallucination.

### 2.2 Interaction with Tree-Based Models

Decision trees split on thresholds:

$$
f(c) \leq t
$$

The arbitrary ordering determines grouping structure during early splits.  
Although trees are less sensitive than linear models, the embedding still shapes partitioning topology.


> Label encoding introduces artificial ordinal structure unless rank is intrinsic.
{: .prompt-warning }

> **Visual Suggestion:**  
> Show arbitrary category ordering on a line and illustrate how different random permutations change tree splits.
{: .prompt-info }
---

# 3. One-Hot Encoding: Orthogonal Basis Representation

One-hot encoding maps categories to canonical basis vectors:

$$
c_i \mapsto e_i \in \mathbb{R}^k
$$

where $e_i$ contains a 1 in position $i$ and 0 elsewhere.

### 3.1 Induced Geometry

The Euclidean distance between any two categories is constant:

$$
\|e_i - e_j\|_2 = \sqrt{2}, \quad i \neq j
$$

Thus, no category is closer to another.  
The embedding assumes complete independence.

### 3.2 Statistical Consequences

The parameter space expands from dimension 1 to $k$.  
For a linear model:

$$
\hat{y} = \beta_0 + \sum_{i=1}^{k} \beta_i e_i
$$

Each category receives its own parameter.  
Variance increases when support for some categories is small.

For identifiability with intercept, one dimension must be dropped due to perfect collinearity:

$$
\sum_{i=1}^{k} e_i = 1
$$

### 3.3 High-Cardinality Regime

When $k$ is large, dimensionality scales linearly.  
Sparse representations increase memory and degrade conditioning.


> High-cardinality one-hot encoding inflates parameter dimensionality and increases estimator variance for rare categories.
{: .prompt-warning }

{: .prompt-info }
> **Visual Suggestion:**  
> Sparse matrix visualization contrasting low vs high cardinality.

---

# 4. Target Encoding: Conditional Expectation Embedding

Target encoding replaces identity with empirical expectation:

$$
c \mapsto \mathbb{E}[Y \mid C=c]
$$

The category becomes a sufficient statistic for outcome tendency.

### 4.1 Naïve Estimation

Let $\mu_c$ denote the sample mean for category $c$ and $n_c$ its count.  
Direct substitution yields high variance when $n_c$ is small.

### 4.2 Shrinkage Formulation

Regularized estimate:

$$
\hat{\mu}_c =
\frac{n_c \mu_c + \alpha \mu}{n_c + \alpha}
$$

where $\mu$ is the global mean and $\alpha$ controls shrinkage.

As $n_c \to 0$, $\hat{\mu}_c \to \mu$.  
As $n_c \to \infty$, $\hat{\mu}_c \to \mu_c$.

This is equivalent to empirical Bayes shrinkage under a conjugate prior interpretation.

### 4.3 Leakage Mechanism

If $\hat{\mu}_c$ is computed using full training data, then each observation contributes to its own encoding:

$$
y_i \rightarrow \hat{\mu}_{c_i}(y_i)
$$

This creates a circular dependency.

{: .prompt-warning }
> Target encoding must be computed out-of-fold to prevent target leakage.

{: .prompt-info }
> **Visual Suggestion:**  
> Flow diagram showing self-information leakage path.

---

# 5. Frequency Encoding: Density-Based Representation

Frequency encoding maps categories to empirical prevalence:

$$
c \mapsto \frac{n_c}{N}
$$

The induced geometry reflects statistical mass, not semantic similarity.

Categories with equal frequency collapse to identical representations.  
The encoding assumes that rarity or dominance is predictive.

Unlike target encoding, it introduces no leakage because it is independent of $Y$.

---

# 6. Learned Embeddings: Parametric Geometry

Neural architectures define encoding as a trainable lookup:

$$
f(c) = v_c \in \mathbb{R}^d
$$

where $v_c$ is optimized jointly with the predictive objective.

Similarity becomes emergent rather than imposed.  
Dimensionality $d$ is independent of cardinality $k$.

Embeddings generalize one-hot encoding by allowing the basis to rotate and compress.

{: .prompt-info }
> **Visual Suggestion:**  
> 2D embedding scatter plot showing semantic clustering of categories.

---

# 7. Bias–Variance Perspective

Each encoder navigates the bias–variance tradeoff differently.

- Label encoding introduces bias via artificial ordering.
- One-hot reduces bias but increases variance.
- Target encoding reduces dimensionality but risks high variance and leakage.
- Frequency encoding reduces variance but sacrifices identity.
- Embeddings balance expressivity and compactness through learned structure.

Encoding selection is therefore a statistical decision, not a formatting choice.

---

# 8. Model–Encoding Coupling

The appropriate encoder depends on model class:

Linear models interpret magnitude directly.  
Tree models partition via thresholds.  
Neural networks learn transformations over dense representations.

Encoding and model architecture form a coupled system.  
The embedding defines the geometry; the model defines transformations over that geometry.

Once geometry is fixed, learning is constrained within it.

---
# Conclusion

Most engineers think the model is where intelligence lives.

It isn’t.

It lives in the representation.

Once a categorical variable has been embedded into $\mathbb{R}^k$, the model can only reason within that geometry. It cannot undo an imposed order. It cannot rediscover identity that was collapsed. It cannot separate categories that were merged by frequency. It cannot remove leakage that was baked into expectation.

The hypothesis space is shaped long before training begins.

And here is the uncomfortable part:

If you cannot precisely articulate the geometry your encoding imposes   the metric it defines, the assumptions it encodes, the bias it injects, the variance it amplifies   then you are not controlling your model.

You are guessing at its world.

Encoding is where domain semantics become statistical structure.  
If you do not understand that structure deeply, you are building systems whose reasoning you cannot fully explain   systems that make decisions about credit, hiring, medical triage, fraud detection, recommendation   based on geometries you never examined.

And if that does not make you uneasy, it should.

Because the model is not misunderstanding the data.

It is faithfully executing the geometry you gave it.