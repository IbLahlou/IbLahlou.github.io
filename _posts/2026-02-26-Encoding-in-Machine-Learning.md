---
title: "Encoding in Machine Learning: Designing Categorical Geometry"
description: A geometric and statistical exploration of encoder families — from categorical mappings to latent spaces, positional signals, and semantic retrieval encoders
date: 2026-02-26
categories:
  - Machine Learning
  - Feature Engineering
tags:
  - encoding
  - ml-engineering
  - feature-design
  - modeling
  - transformers
  - representation-learning
math: true
---

## Introduction

Before a model learns, before a loss is minimized, before a gradient flows, a quieter decision is made.

How do we represent the world?

A categorical variable looks innocent. A column of regions. Product IDs. User segments. Device types. We call them "features" as if they were naturally numeric, as if the model were merely waiting for them to be formatted correctly.

But a category is not a number.
It is a partition.
It is a declaration that some observations are equivalent under a certain abstraction.

The moment we transform it, we decide something far more consequential than data formatting. We decide whether two categories are neighbors or strangers. Whether they lie on a line or float in orthogonal isolation. Whether identity matters more than frequency. Whether behavior matters more than structure. Whether similarity is imposed or allowed to emerge.

And once we decide, the model never questions that choice.

What is distance between "France" and "Germany"?
Should "Premium" be twice "Standard"?
Is rarity itself meaningful, or only correlation with outcome?
When we collapse identity into expectation, are we modeling behavior — or leaking it?
When we embed categories in dense vectors, are we discovering structure — or inventing it?

Only at the end do we name the act: to **encode** — from Latin _in-_ ("into") and _codex_ ("a book of rules") — is to inscribe something into a system. In machine learning, it is the transformation of information into numerical form so a model can process it.

Encoding is the first act of modeling.
It is where epistemology becomes geometry.

Everything that follows — bias, variance, generalization, fairness, stability — is downstream of that act.

---

## 1. A Taxonomy of Encoder Families

Encoding is not one technique. It is a family of techniques, each answering a different question about what structure to impose and from where.

| Family | What it encodes | Geometry imposed | Typical domain |
|--------|----------------|-----------------|----------------|
| **Categorical** | discrete identity | scalar or orthogonal basis | tabular ML |
| **Latent Space** | compressed structure | learned manifold | generative models |
| **Positional** | sequence order | frequency-based or learned offsets | Transformers, LLMs |
| **Semantic** | meaning of objects or pairs | dense vector or scalar score | NLP, retrieval |

Each family operates on a different input type, serves a different modeling objective, and imposes a fundamentally different geometry.

The choice of encoder family is not a preprocessing detail — it defines what the model is allowed to know about the world.

> Encoding determines whether categories become collinear scalars, orthogonal axes, empirical expectations, or learned manifolds.
{: .prompt-info }

---

## 2. Categorical Encoders

Categorical encoders address the most classical problem: a finite set of labels must become numbers.

Formally, they define a mapping from a discrete set $\mathcal{C}$ into some Euclidean space:

$$
f : \mathcal{C} \rightarrow \mathbb{R}^k
$$

The choice of $f$ determines adjacency, distance, and orientation. Once applied, the model interacts only with the induced geometry — never again with the original set.

| Encoder | Geometric form | Key assumption |
|---------|---------------|----------------|
| Label | 1D ordered axis | Total ordering exists |
| One-Hot | Orthogonal basis | Categories fully independent |
| Ordinal | Ordered scalar | Uniform rank spacing |
| Target | Conditional mean | Outcome tendency defines identity |
| Frequency | Density scalar | Prevalence is predictive |

### 2.1 Label Encoding

Label encoding maps each category to an integer:

$$
\{c_1, \dots, c_k\} \mapsto \{0, 1, \dots, k-1\}
$$

The induced metric is absolute difference $d(c_i, c_j) = |f(c_i) - f(c_j)|$, implying uniform spacing and total order.

Two arbitrary label assignments for the same five regions produce different tree splits at threshold $f(c) \leq 1$:

| Region  | Encoding A | Encoding B |
|---------|-----------|-----------|
| North   | 0         | 3         |
| South   | 1         | 0         |
| East    | 2         | 1         |
| West    | 3         | 4         |
| Central | 4         | 2         |

- **Split A** (`f(c) ≤ 1`): {North, South} vs {East, West, Central}
- **Split B** (`f(c) ≤ 1`): {South, East} vs {North, West, Central}

Same model, same threshold — entirely different groupings, determined solely by encoding order.

> Label encoding introduces artificial ordinal structure unless rank is intrinsic to the domain.
{: .prompt-warning }

### 2.2 One-Hot Encoding

One-hot encoding maps categories to canonical basis vectors $c_i \mapsto e_i \in \mathbb{R}^k$, where $e_i$ has a 1 in position $i$ and 0 elsewhere.

The Euclidean distance between any two categories is constant: $\|e_i - e_j\|_2 = \sqrt{2}$ for $i \neq j$. No category is closer to another — the embedding assumes complete independence.

**Low cardinality** (3 categories — manageable):

| Observation | is_France | is_Germany | is_UK |
|-------------|-----------|------------|-------|
| obs_1       | 1         | 0          | 0     |
| obs_2       | 0         | 1          | 0     |
| obs_3       | 0         | 0          | 1     |

**High cardinality** (6 categories — sparse, mostly zeros):

| Observation | is_FR | is_DE | is_UK | is_ES | is_IT | is_PL |
|-------------|-------|-------|-------|-------|-------|-------|
| obs_1       | 1     | 0     | 0     | 0     | 0     | 0     |
| obs_2       | 0     | 0     | 0     | 0     | 1     | 0     |
| obs_3       | 0     | 0     | 0     | 1     | 0     | 0     |

With $k = 500$ product IDs, each row is 499 zeros and 1 one. Memory scales with $n \times k$. For identifiability with an intercept, one dimension must be dropped to avoid perfect collinearity.

> High-cardinality one-hot encoding inflates parameter dimensionality and increases estimator variance for rare categories.
{: .prompt-warning }

### 2.3 Target Encoding

Target encoding replaces identity with empirical expectation:

$$
c \mapsto \mathbb{E}[Y \mid C = c]
$$

The category becomes a sufficient statistic for outcome tendency. To control variance under small sample sizes, shrinkage is applied:

$$
\hat{\mu}_c = \frac{n_c \mu_c + \alpha \mu}{n_c + \alpha}
$$

where $\mu$ is the global mean and $\alpha$ controls regularization. This is equivalent to empirical Bayes shrinkage under a conjugate prior.

The critical risk is leakage: if $\hat{\mu}_c$ is computed using full training data, each observation's $y$ contributes to its own encoding.

**Naïve (leaky):** each observation's $y$ is in its own $\hat{\mu}_c$.

| obs | category | y    | $\hat{\mu}_c$ (full data) | self-contribution |
|-----|----------|------|---------------------------|-------------------|
| 1   | A        | 1.00 | 0.75                      | yes               |
| 2   | A        | 0.50 | 0.75                      | yes               |
| 3   | B        | 0.20 | 0.20                      | yes               |

**Out-of-fold (correct):** obs 1's encoding is computed on folds that exclude obs 1.

| obs | category | y    | $\hat{\mu}_c$ (excl. self) | self-contribution |
|-----|----------|------|----------------------------|-------------------|
| 1   | A        | 1.00 | 0.625                      | no                |
| 2   | A        | 0.50 | 0.583                      | no                |
| 3   | B        | 0.20 | —                          | no                |

> Target encoding must be computed out-of-fold to prevent target leakage.
{: .prompt-warning }

### 2.4 Frequency Encoding

Frequency encoding maps categories to empirical prevalence:

$$
c \mapsto \frac{n_c}{N}
$$

The geometry reflects statistical mass, not semantic similarity. Categories with equal frequency collapse to identical representations. Unlike target encoding, it introduces no leakage — it is independent of $Y$.

---

## 3. Latent Space Encoders

Latent space encoders do not map a predefined set to a predefined space. They learn the encoding itself from data, optimizing for a task — reconstruction, generation, or classification.

The common structure is a bottleneck: high-dimensional input is compressed into a lower-dimensional latent representation $z$, which the model must use to accomplish its objective.

| Encoder type | Latent $z$ | Training signal | Key property |
|--------------|-----------|-----------------|--------------|
| Autoencoder | deterministic point | reconstruction loss | unsupervised compression |
| VAE | distribution $(\mu, \sigma)$ | reconstruction + KL | structured, traversable latent space |
| VQ-VAE | discrete codebook index | reconstruction + commitment loss | discrete latent space |

### 3.1 Autoencoders

An autoencoder learns to encode by learning to reconstruct.

$$
z = f_\theta(x), \qquad \hat{x} = g_\phi(z), \qquad \mathcal{L} = \|x - g_\phi(f_\theta(x))\|^2
$$

The bottleneck $z \in \mathbb{R}^d$ with $d \ll \dim(x)$ forces the encoder to compress. Whatever survives the bottleneck is the representation.

| Layer | Dimension | Role |
|-------|-----------|------|
| Input $x$ | 784 (e.g. 28×28) | Raw observation |
| Encoder output $z$ | 32 | Compressed representation |
| Decoder output $\hat{x}$ | 784 | Reconstruction |

The geometry of $z$ is not prescribed — it emerges from what is necessary to reconstruct $x$.

> Autoencoder encodings minimize reconstruction error, not predictive accuracy. A good reconstruction embedding is not necessarily a good predictive embedding.
{: .prompt-warning }

### 3.2 Variational Autoencoders

A variational autoencoder replaces the deterministic bottleneck with a distribution. The encoder outputs parameters of a posterior:

$$
q_\theta(z \mid x) = \mathcal{N}(\mu_\theta(x),\, \sigma^2_\theta(x))
$$

A sample is drawn via the reparameterization trick, keeping the gradient path differentiable:

$$
z = \mu_\theta(x) + \sigma_\theta(x) \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

The training objective adds KL regularization:

$$
\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{reconstruction}} + \underbrace{D_{\text{KL}}\big(q_\theta(z \mid x) \,\|\, \mathcal{N}(0,I)\big)}_{\text{regularization}}
$$

The KL term pulls the posterior toward a standard Gaussian, enforcing a structured, continuous latent space.

| Property | Autoencoder | VAE |
|----------|-------------|-----|
| Latent $z$ | deterministic point | distribution $(\mu, \sigma)$ |
| Latent structure | unregularized | regularized toward $\mathcal{N}(0,I)$ |
| Sampling new points | not meaningful | meaningful interpolation |
| Objective | reconstruction only | reconstruction + KL divergence |

Because the latent space is regularized, points sampled between two known encodings decode into plausible observations. The geometry is smooth and traversable.

> The VAE encodes not a point but a region of uncertainty. This enables controlled generation and interpolation — autoencoders cannot do this reliably.
{: .prompt-info }

---

## 4. Positional Encoders

Attention mechanisms are permutation-invariant. A Transformer has no built-in notion of sequence order — it treats a sentence and a shuffled version of that sentence identically.

Positional encoders solve this by injecting a position-dependent signal into each token representation:

$$
x'_t = x_t + PE(t)
$$

| Variant | How position enters | Learnable | Extrapolates to longer sequences |
|---------|--------------------|-----------|---------------------------------|
| Sinusoidal (Vaswani 2017) | additive, fixed | no | yes (by design) |
| Learned absolute | additive lookup table | yes | no |
| RoPE | multiplicative rotation in attention | partially | yes |
| ALiBi | additive bias on attention scores | no | yes |

### 4.1 Sinusoidal Positional Encoding

The canonical formulation assigns each position $t$ a vector of alternating sine and cosine values at geometrically spaced frequencies:

$$
PE_{(t,\, 2i)} = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \qquad
PE_{(t,\, 2i+1)} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)
$$

High-frequency dimensions encode fine-grained position; low-frequency dimensions encode coarse position.

| Position | dim 0 ($\sin$) | dim 1 ($\cos$) | dim 2 ($\sin$) | dim 3 ($\cos$) |
|:--------:|:--------------:|:--------------:|:--------------:|:--------------:|
| 0        | 0.000          | 1.000          | 0.000          | 1.000          |
| 1        | 0.841          | 0.540          | 0.010          | 1.000          |
| 2        | 0.909          | −0.416         | 0.020          | 1.000          |
| 3        | 0.141          | −0.990         | 0.030          | 1.000          |
| 4        | −0.757         | −0.654         | 0.040          | 0.999          |

The inner product $PE(t)^\top PE(t')$ depends only on the offset $t - t'$, not on absolute position. This gives the model a natural inductive bias toward relative position.

### 4.2 Relative and Rotary Variants

Modern large language models move position into attention rather than into token representations.

**RoPE** (Rotary Position Embedding) encodes position as a rotation applied to query and key vectors before computing attention. The rotation angle depends on the position difference, making attention scores inherently position-relative.

**ALiBi** (Attention with Linear Biases) adds a fixed negative slope to attention scores as a function of key-query distance — no vector modification required.

Both variants improve length generalization: a model trained on sequences of length 512 can be applied to sequences of length 2048 without degradation.

> Positional encodings are not learned from labels — they are geometric injections. The model learns to use them; it does not learn what they are.
{: .prompt-info }

---

## 5. Semantic Encoders

Semantic encoders map natural language — or structured objects — into representations that reflect meaning, not just identity.

The central question they answer: how similar is A to B?

| Encoder type | Input | Output | Similarity computation |
|--------------|-------|--------|----------------------|
| Bi-encoder | single text | dense vector | $f(q)^\top f(d)$ (dot product) |
| Cross-encoder | text pair $(q, d)$ | scalar score | full attention over the pair |
| Poly-encoder | query + multiple candidates | weighted sum | intermediate between the two |

### 5.1 Bi-Encoders

A bi-encoder encodes query and document independently:

$$
s(q, d) = f_\theta(q)^\top f_\theta(d)
$$

Because $f_\theta(d)$ can be precomputed for all documents, retrieval scales to billions of candidates via approximate nearest neighbor search.

The constraint is strong: each representation must be independently sufficient. Cross-document reasoning is impossible — the model cannot attend to tokens in $d$ while encoding $q$.

### 5.2 Cross-Encoders

A cross-encoder concatenates query and document and processes the pair jointly:

$$
s = f_\theta\bigl([q \,;\, \text{[SEP]} \,;\, d]\bigr)
$$

Full attention operates over both inputs simultaneously, allowing every token in $q$ to interact with every token in $d$. The output is a scalar relevance score.

| Property | Bi-Encoder | Cross-Encoder |
|----------|-----------|--------------|
| Encoding | $f(q)$, $f(d)$ separately | $f([q; d])$ jointly |
| Precomputation | yes — index documents offline | no — must recompute per pair |
| Latency at query time | fast (ANN lookup) | slow (full forward pass per pair) |
| Expressivity | limited | high |
| Typical role | first-stage retrieval | second-stage reranking |

In production retrieval systems, both are used in sequence: a bi-encoder retrieves a candidate set, a cross-encoder reranks it.

> Cross-encoders do not produce embeddings — they produce scores. The encoding is not a point in space; it is a function of a pair.
{: .prompt-info }

---

## 6. Bias–Variance Across Encoder Families

Each encoder family navigates the bias–variance tradeoff from a different axis.

**Categorical encoders:**
- Label encoding introduces bias via artificial ordering.
- One-hot reduces bias but increases variance with cardinality.
- Target encoding reduces dimensionality but risks high variance and leakage.
- Frequency encoding reduces variance but discards identity.

**Latent space encoders:**
- Autoencoders minimize reconstruction variance, not predictive variance — the objectives diverge.
- VAEs regularize the latent space, trading slight reconstruction bias for a smoother, lower-variance geometry.

**Positional encoders:**
- Sinusoidal PE is fixed — zero variance by construction, but its bias is baked in by design.
- Learned absolute PE has low bias but does not generalize beyond training length.
- RoPE and ALiBi achieve low bias and low variance through relative structure.

**Semantic encoders:**
- Bi-encoders have bounded expressivity — their independence constraint is a structural bias.
- Cross-encoders have no independence constraint, but do not produce reusable embeddings.

Encoding selection is a statistical decision, not a formatting choice.

---

## 7. Model–Encoding Coupling

The appropriate encoder family depends on the model and task:

| Model class | Encoding family | Reason |
|-------------|----------------|--------|
| Linear model | Categorical (one-hot, target) | Magnitude is interpreted directly |
| Tree model | Categorical (label, target) | Thresholds partition the axis |
| MLP | Categorical + learned embeddings | Dense input preferred |
| Transformer | Positional + semantic | Attention requires position signal |
| Generative model | Latent space | Bottleneck defines generation |
| Retrieval system | Semantic (bi + cross) | Scalability vs accuracy tradeoff |

Encoding and model architecture form a coupled system. The embedding defines the geometry; the model defines transformations over that geometry.

Once geometry is fixed, learning is constrained within it.

---

## Conclusion

Most engineers think the model is where intelligence lives.

It isn't.

It lives in the representation.

Once a categorical variable has been embedded into $\mathbb{R}^k$, the model can only reason within that geometry. It cannot undo an imposed order. It cannot rediscover identity that was collapsed. It cannot separate categories that were merged by frequency. It cannot remove leakage that was baked into expectation.

The hypothesis space is shaped long before training begins.

And here is the uncomfortable part:

If you cannot precisely articulate the geometry your encoding imposes — the metric it defines, the assumptions it encodes, the bias it injects, the variance it amplifies — then you are not controlling your model.

You are guessing at its world.

Encoding is where domain semantics become statistical structure.
If you do not understand that structure deeply, you are building systems whose reasoning you cannot fully explain — systems that make decisions about credit, hiring, medical triage, fraud detection, recommendation — based on geometries you never examined.

And if that does not make you uneasy, it should.

Because the model is not misunderstanding the data.

It is faithfully executing the geometry you gave it.
