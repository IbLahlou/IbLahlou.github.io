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
pin: true
math: true
image:
  path: /assets/img/panels/panel14@4x.png
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

| Family           | What it encodes             | Geometry imposed                   | Typical domain     |
| ---------------- | --------------------------- | ---------------------------------- | ------------------ |
| **Categorical**  | discrete identity           | scalar or orthogonal basis         | tabular ML         |
| **Latent Space** | compressed structure        | learned manifold                   | generative models  |
| **Positional**   | sequence order              | frequency-based or learned offsets | Transformers, LLMs |
| **Semantic**     | meaning of objects or pairs | dense vector or scalar score       | NLP, retrieval     |

Each family operates on a different input type, serves a different modeling objective, and imposes a fundamentally different geometry.

The choice of encoder family is not a preprocessing detail — it defines what the model is allowed to know about the world.

> Encoding determines whether categories become collinear scalars, orthogonal axes, empirical expectations, or learned manifolds.
> {: .prompt-info }

---

## 2. Categorical Encoders

Categorical encoders address the most classical problem: a finite set of labels must become numbers.

Formally, they define a mapping from a discrete set $\mathcal{C}$ into some Euclidean space:

$$
f : \mathcal{C} \rightarrow \mathbb{R}^k
$$

The choice of $f$ determines adjacency, distance, and orientation. Once applied, the model interacts only with the induced geometry — never again with the original set.

| Encoder   | Geometric form   | Key assumption                    |
| --------- | ---------------- | --------------------------------- |
| Label     | 1D ordered axis  | Total ordering exists             |
| One-Hot   | Orthogonal basis | Categories fully independent      |
| Ordinal   | Ordered scalar   | Uniform rank spacing              |
| Target    | Conditional mean | Outcome tendency defines identity |
| Frequency | Density scalar   | Prevalence is predictive          |

### 2.1 Label Encoding

Label encoding maps each category to an integer:

$$
\{c_1, \dots, c_k\} \mapsto \{0, 1, \dots, k-1\}
$$

The induced metric is absolute difference $d(c_i, c_j) = |f(c_i) - f(c_j)|$, implying uniform spacing and total order.

Two arbitrary label assignments for the same five regions produce different tree splits at threshold $f(c) \leq 1$:

| Region  | Encoding A | Encoding B |
| ------- | ---------- | ---------- |
| North   | 0          | 3          |
| South   | 1          | 0          |
| East    | 2          | 1          |
| West    | 3          | 4          |
| Central | 4          | 2          |

- **Split A** (`f(c) ≤ 1`): {North, South} vs {East, West, Central}
- **Split B** (`f(c) ≤ 1`): {South, East} vs {North, West, Central}

Same model, same threshold — entirely different groupings, determined solely by encoding order.

> Label encoding introduces artificial ordinal structure unless rank is intrinsic to the domain.
> {: .prompt-warning }

> **Bias–variance:** Label encoding is a zero-variance transformation — it is deterministic — but it introduces maximal structural bias for nominal data by imposing a total order that does not exist in the domain.
> {: .prompt-tip }

### 2.2 One-Hot Encoding

One-hot encoding maps categories to canonical basis vectors $c_i \mapsto e_i \in \mathbb{R}^k$, where $e_i$ has a 1 in position $i$ and 0 elsewhere.

The Euclidean distance between any two categories is constant: $\|e_i - e_j\|_2 = \sqrt{2}$ for $i \neq j$. No category is closer to another — the embedding assumes complete independence.

**Low cardinality** (3 categories — manageable):

| Observation | is_France | is_Germany | is_UK |
| ----------- | --------- | ---------- | ----- |
| obs_1       | 1         | 0          | 0     |
| obs_2       | 0         | 1          | 0     |
| obs_3       | 0         | 0          | 1     |

**High cardinality** (6 categories — sparse, mostly zeros):

| Observation | is_FR | is_DE | is_UK | is_ES | is_IT | is_PL |
| ----------- | ----- | ----- | ----- | ----- | ----- | ----- |
| obs_1       | 1     | 0     | 0     | 0     | 0     | 0     |
| obs_2       | 0     | 0     | 0     | 0     | 1     | 0     |
| obs_3       | 0     | 0     | 0     | 1     | 0     | 0     |

With $k = 500$ product IDs, each row is 499 zeros and 1 one. Memory scales with $n \times k$. For identifiability with an intercept, one dimension must be dropped to avoid perfect collinearity.

> High-cardinality one-hot encoding inflates parameter dimensionality and increases estimator variance for rare categories.
> {: .prompt-warning }

> **Bias–variance:** One-hot carries near-zero structural bias — no ordering is imposed, all categories are equidistant. The cost is variance: parameter count scales with $k$, and rare categories with $n_c \ll n$ have their parameters estimated from very few observations.
> {: .prompt-tip }

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
| --- | -------- | ---- | ------------------------- | ----------------- |
| 1   | A        | 1.00 | 0.75                      | yes               |
| 2   | A        | 0.50 | 0.75                      | yes               |
| 3   | B        | 0.20 | 0.20                      | yes               |

**Out-of-fold (correct):** obs 1's encoding is computed on folds that exclude obs 1.

| obs | category | y    | $\hat{\mu}_c$ (excl. self) | self-contribution |
| --- | -------- | ---- | -------------------------- | ----------------- |
| 1   | A        | 1.00 | 0.625                      | no                |
| 2   | A        | 0.50 | 0.583                      | no                |
| 3   | B        | 0.20 | —                          | no                |

> Target encoding must be computed out-of-fold to prevent target leakage.
> {: .prompt-warning }

> **Bias–variance:** The naive estimator has high variance when $n_c$ is small. Shrinkage reduces variance at the cost of pulling category estimates toward the global mean, introducing bias.
> {: .prompt-tip }

### 2.4 Frequency Encoding

Frequency encoding maps categories to empirical prevalence:

$$
c \mapsto \frac{n_c}{N}
$$

The geometry reflects statistical mass, not semantic similarity. Categories with equal frequency collapse to identical representations. Unlike target encoding, it introduces no leakage — it is independent of $Y$.

> **Bias–variance:** Frequency encoding is a low-variance, high-bias estimator when categorical identity carries signal — it substitutes prevalence for meaning, collapsing distinct categories to identical representations whenever their counts coincide.
> {: .prompt-tip }

---

## 3. Latent Space Encoders

A [**latent space**](https://en.wikipedia.org/wiki/Latent_space), also known as a **latent feature space** or **embedding space**, is an [embedding](https://en.wikipedia.org/wiki/Embedding_\(machine_learning\) "Embedding (machine learning)") of a set of items within a [manifold](https://en.wikipedia.org/wiki/Manifold "Manifold") in which items resembling each other are positioned closer to one another. Position within the latent space can be viewed as being defined by a set of [latent variables](https://en.wikipedia.org/wiki/Latent_variable "Latent variable") that emerge from the resemblances from the objects.

Latent space encoders do not map a predefined set to a predefined space. They learn the encoding itself from data, optimizing for a task  reconstruction, generation, or classification.

The common structure is a bottleneck: high-dimensional input is compressed into a lower-dimensional latent representation $z$, which the model must use to accomplish its objective.

All autoencoder variants share the encoder–decoder skeleton but differ in what they regularize, what they corrupt, or what architecture they impose — the objective function is where the design choices live.

### 3.1 Autoencoders

An autoencoder frames representation learning as a reconstruction problem. It is composed of two parametric functions trained jointly:

- **Encoder** $f_\theta : \mathcal{X} \rightarrow \mathcal{Z}$ — compresses the input down to a low-dimensional code $z$.
- **Decoder** $g_\phi : \mathcal{Z} \rightarrow \mathcal{X}$ — reconstructs the original input from $z$ alone.

The encoder maps $x$ to a latent code $z$, the decoder reconstructs $\hat{x}$ from $z$, and the whole system is trained end-to-end to minimize reconstruction error:

$$
z = f_\theta(x), \qquad \hat{x} = g_\phi(z), \qquad \mathcal{L} = \|x - g_\phi(f_\theta(x))\|^2
$$

The information bottleneck — $\dim(z) \ll \dim(x)$ — is the key constraint. Since the decoder must recover $x$ from $z$ alone, the encoder must retain the most statistically informative structure and discard redundancy. This is conceptually related to truncated PCA, but the encoder is nonlinear, so the compression can exploit higher-order structure that linear projections miss.

**Simple example — tabular data.** A user profile has 8 features: age, tenure, monthly spend, login frequency, number of products, support tickets, days since last purchase, account tier (numeric). These 8 numbers form $x$. The encoder compresses them: 8 → Dense(4, ReLU) → Dense(2) → $z = [0.7,\; -1.2]$. Two numbers now summarize the entire profile. The decoder reconstructs: 2 → Dense(4, ReLU) → Dense(8) → $\hat{x}$. After training, $z_1$ might separate high-value from low-value users; $z_2$ might separate active from churned ones — but these labels are never given. The geometry is discovered from what the decoder needs to reconstruct.

| Layer                 | Dimension | Role                                 |
| --------------------- | --------- | ------------------------------------ |
| Input $x$             | 8         | Raw user features                    |
| Encoder hidden layer  | 4         | Intermediate compression             |
| Bottleneck $z$        | 2         | Compressed representation            |
| Decoder hidden layer  | 4         | Intermediate expansion               |
| Output $\hat{x}$      | 8         | Reconstruction                       |

**Visual example — images.** For a 28×28 grayscale digit image (784 pixels), the same skeleton scales up: 784 → Dense(256, ReLU) → Dense(128, ReLU) → Dense(32) → $z$, then reversed. The 32-number code might capture stroke thickness, digit slant, or loop closure — whatever axes of variation best explain the training set. The decoder reconstructs 784 pixel values from those 32 numbers. The further the architecture from convolutional structure, the more the encoder wastes capacity modeling spatial correlations that convolutions handle for free.

What the encoder **does not** do in either case: assign predetermined meaning to dimensions, or guarantee nearby points in $\mathcal{Z}$ decode to similar inputs. Two separately trained autoencoders on the same data will discover entirely different coordinate systems.

No prior is placed on $z$. The latent space is organized however minimizes the loss — which makes autoencoders effective for anomaly detection (out-of-distribution inputs reconstruct poorly) but poorly suited for generation (arbitrary samples from $\mathcal{Z}$ decode to noise, since there is no guarantee the space between encoded points is meaningful).

> Autoencoder encodings minimize reconstruction error, not predictive accuracy. A good reconstruction embedding is not necessarily a good predictive embedding.
> {: .prompt-warning }

> **Bias–variance:** When reconstructive and discriminative dimensions diverge, autoencoders exhibit an inductive bias mismatch — the encoder may retain high-variance but low-signal dimensions (they help reconstruction) while discarding low-variance but high-signal dimensions (they are easily reconstructed from context). The latent space itself has no regularization, so its geometry can be arbitrarily fragmented.
> {: .prompt-tip }

### 3.2 Variational Autoencoders

A plain autoencoder has no prior over $\mathcal{Z}$. Two observations that are semantically similar may land in distant, unrelated regions of latent space, because nothing in the objective penalizes that. Sampling an arbitrary point from $\mathcal{Z}$ and decoding it produces incoherent outputs, because the decoder has only been trained on points that are direct outputs of the encoder — the complement of that set is effectively out-of-distribution.

A **Variational Autoencoder** (VAE) addresses this by reformulating the problem as variational inference. Rather than learning a deterministic encoding, the encoder learns a posterior distribution over latent codes. The architecture remains encoder–decoder, but the encoder now outputs the parameters of a Gaussian:

- **Encoder** $f_\theta : \mathcal{X} \rightarrow (\mu, \sigma)$ — two parallel output heads instead of one. For the user profile above: $\mu = [0.70,\; -1.20]$ (the most likely location in $\mathcal{Z}$ for this user) and $\sigma = [0.15,\; 0.30]$ (how uncertain the encoder is along each latent dimension).
- **Sampling** — $z$ is drawn from $q_\theta(z \mid x) = \mathcal{N}(\mu_\theta(x),\, \sigma_\theta^2(x))$, a different point each forward pass.
- **Decoder** $g_\phi : \mathcal{Z} \rightarrow \mathcal{X}$ — same role as in a plain autoencoder; reconstructs the input from the sampled $z$.

Encoding to a distribution rather than a point means every input occupies a region in $\mathcal{Z}$. When the regularization term enforces overlap between those regions, the complement of the training encodings is no longer out-of-distribution — the latent space becomes dense enough that arbitrary samples decode coherently.

**The reparameterization trick** makes this trainable. Sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is not differentiable — gradients cannot flow back through a stochastic node to reach $\mu$ and $\sigma$. The trick externalizes the randomness: sample auxiliary noise $\varepsilon \sim \mathcal{N}(0, I)$ independently of the parameters, then construct $z$ deterministically:

$$
z = \mu_\theta(x) + \sigma_\theta(x) \cdot \varepsilon
$$

**Concretely**

(tabular): $\mu = [0.70,\; -1.20]$, $\sigma = [0.15,\; 0.30]$, draw $\varepsilon = [0.50,\; -0.80]$,
so $z = [0.70 + 0.15 \times 0.50,\; -1.20 + 0.30 \times (-0.80)] = [0.775,\; -1.44]$. 

This $z$ is passed to the decoder. On the next pass, a new $\varepsilon$ is drawn, producing a slightly different $z$  the encoder learns to keep $\sigma$ small for dimensions that matter for reconstruction, and the KL term prevents $\sigma$ from collapsing to zero entirely.

**Visual example**: on MNIST, each digit class occupies a fuzzy region in $\mathcal{Z}$ rather than a single point. Sampling a $z$ between the "3" region and the "8" region and decoding it produces a smoothly morphed intermediate — closed loops gradually opening. A plain autoencoder cannot do this because the space between the two encoded points has never been seen during training and decodes to noise.

From the optimizer's perspective, $\varepsilon$ is a fixed constant. $\mu$ and $\sigma$ are differentiable functions of the input, so gradients propagate through the entire encoder normally.

**The training objective** is the Evidence Lower BOund (ELBO), written as two terms:

$$
\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{reconstruction}} \;+\; \underbrace{D_{\text{KL}}\!\left(q_\theta(z \mid x) \;\|\; \mathcal{N}(0,I)\right)}_{\text{latent regularization}}
$$

The reconstruction term keeps the encoding informative — same pressure as a plain autoencoder. The KL term penalizes how much the encoder's posterior $q_\theta(z \mid x)$ deviates from the prior $\mathcal{N}(0, I)$. It is zero when the posterior matches the prior exactly, and grows as they diverge. Its effect is to anchor every encoding region near the origin with bounded spread, so the latent space cannot collapse into isolated islands or grow unbounded.

The two terms are in fundamental tension: reconstruction pushes the encoder toward sharp, concentrated posteriors (more information preserved); the KL term pushes toward diffuse posteriors that all resemble the prior (more regularity). The model settles at a balance determined by the relative weighting — in practice, a scalar $\beta > 1$ on the KL term ($\beta$-VAE) can increase disentanglement at the cost of reconstruction fidelity.

| Property             | Autoencoder         | VAE                                   |
| -------------------- | ------------------- | ------------------------------------- |
| Latent $z$           | deterministic point | sampled from $(\mu, \sigma)$          |
| Latent structure     | unregularized       | regularized toward $\mathcal{N}(0,I)$ |
| Gradient through $z$ | direct              | via reparameterization                |
| Sampling new points  | not meaningful      | meaningful interpolation              |
| Objective            | reconstruction only | reconstruction + KL divergence        |

Because the latent space is regularized, points sampled between two known encodings decode into plausible observations. The geometry is smooth and traversable.

> The VAE encodes not a point but a region of uncertainty. This enables controlled generation and interpolation — plain autoencoders cannot do this reliably.
> {: .prompt-info }

> **Bias–variance:** The KL term introduces bias by pulling the posterior toward the prior $\mathcal{N}(0,I)$, but reduces geometric variance in $\mathcal{Z}$ — the latent space becomes smoother and more predictable across the data manifold. The $\beta$ weight controls this tradeoff directly: higher $\beta$ increases bias and lowers variance; lower $\beta$ approaches the unregularized autoencoder.
> {: .prompt-tip }

### 3.3 A Note on Latent Space Geometry

All encoders above assume $\mathcal{Z} = \mathbb{R}^d$ — a flat Euclidean space. This is appropriate for most data, but it is a structural assumption, not a necessity. When the data has intrinsic hierarchical or graph-structured topology, Euclidean geometry distorts the representation: exponentially growing structures (trees, taxonomies, scale-free networks) cannot be embedded faithfully in polynomial-volume space without dimensional blowup.

Curved geometries — hyperbolic space in particular — resolve this by matching the geometry of the latent space to the geometry of the data manifold. This connection between representation power and geometric structure is explored in depth through the lens of graph theory in a dedicated post.

> For a full treatment of Euclidean vs. non-Euclidean latent spaces, graph-theoretic representations, and how geometry determines what a model can express, see [**The Power of Representation: Graph Theory and Geometric Learning**](/posts/graph-theory-and-geometric-representation-learning/).
{: .prompt-info }

---

## 4. Positional Encoders

Attention mechanisms are permutation-invariant. A Transformer has no built-in notion of sequence order — it treats a sentence and a shuffled version of that sentence identically.

Positional encoders solve this by injecting a position-dependent signal into each token representation:

$$
x'_t = x_t + PE(t)
$$

| Variant                   | How position enters                  | Learnable | Extrapolates to longer sequences |
| ------------------------- | ------------------------------------ | --------- | -------------------------------- |
| Sinusoidal (Vaswani 2017) | additive, fixed                      | no        | yes (by design)                  |
| Learned absolute          | additive lookup table                | yes       | no                               |
| RoPE                      | multiplicative rotation in attention | partially | yes                              |
| ALiBi                     | additive bias on attention scores    | no        | yes                              |

### 4.1 Sinusoidal Positional Encoding

Each position $t$ is mapped to a $d$-dimensional vector by applying sine and cosine at geometrically decreasing frequencies across dimension pairs:

$$
PE_{(t,\, 2i)} = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \qquad
PE_{(t,\, 2i+1)} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)
$$

The design is a multi-scale decomposition of position. Low-index dimensions ($i$ small) have high frequency — they oscillate rapidly and resolve fine-grained positional differences between adjacent tokens. High-index dimensions ($i$ large) have low frequency — they change slowly and encode coarse positional structure across long spans. Together the $d$ dimensions uniquely identify any position up to the sequence length the frequencies can represent.

Using both sine and cosine at each frequency is deliberate: any phase shift $PE(t + \Delta)$ can be expressed as a linear transformation of $PE(t)$, making relative position a linear operation in the encoding space.

| Position | dim 0 ($\sin$, high freq) | dim 1 ($\cos$, high freq) | dim 2 ($\sin$, low freq) | dim 3 ($\cos$, low freq) |
| :------: | :-----------------------: | :-----------------------: | :----------------------: | :----------------------: |
|    0     |           0.000           |           1.000           |          0.000           |          1.000           |
|    1     |           0.841           |           0.540           |          0.010           |          1.000           |
|    2     |           0.909           |          −0.416           |          0.020           |          1.000           |
|    3     |           0.141           |          −0.990           |          0.030           |          1.000           |
|    4     |          −0.757           |          −0.654           |          0.040           |          0.999           |

The inner product $PE(t)^\top PE(t')$ depends only on the offset $t - t'$, not on absolute position. This gives the model a structural inductive bias toward relative position: the similarity between two positional encodings is a function of their distance, not their location in the sequence.

> **Bias–variance:** Sinusoidal PE has zero variance by construction — it is a deterministic mapping with no trainable components. Its bias is determined entirely by the design choice: the functional form assumes position structure is well-captured by a fixed multi-frequency basis.
> {: .prompt-tip }

### 4.2 Relative and Rotary Variants

Modern large language models move position into attention rather than into token representations.

**RoPE** (Rotary Position Embedding) encodes position as a rotation applied to query and key vectors before computing attention. The rotation angle depends on the position difference, making attention scores inherently position-relative.

**ALiBi** (Attention with Linear Biases) adds a fixed negative slope to attention scores as a function of key-query distance — no vector modification required.

Both variants improve length generalization: a model trained on sequences of length 512 can be applied to sequences of length 2048 without degradation.

> **Bias–variance:** Learned absolute PE introduces variance proportional to the number of trainable position vectors and fails to generalize beyond the maximum sequence length seen during training. RoPE and ALiBi encode position as a relative offset, reducing both the bias from absolute position assumptions and the variance from length generalization failures.
> {: .prompt-tip }

> Positional encodings are not learned from labels — they are geometric injections. The model learns to use them; it does not learn what they are.
> {: .prompt-info }

---

## 5. Semantic Encoders

Semantic encoders map natural language — or structured objects — into representations that reflect meaning, not just identity.

The central question they answer: how similar is A to B?

| Encoder type  | Input                       | Output       | Similarity computation         |
| ------------- | --------------------------- | ------------ | ------------------------------ |
| Bi-encoder    | single text                 | dense vector | $f(q)^\top f(d)$ (dot product) |
| Cross-encoder | text pair $(q, d)$          | scalar score | full attention over the pair   |
| Poly-encoder  | query + multiple candidates | weighted sum | intermediate between the two   |

### 5.1 Bi-Encoders

A bi-encoder encodes query and document independently:

$$
s(q, d) = f_\theta(q)^\top f_\theta(d)
$$

Because $f_\theta(d)$ can be precomputed for all documents, retrieval scales to billions of candidates via approximate nearest neighbor search.

**Example**: query _"early symptoms of diabetes"_ → 768-dim vector $q$. Document _"Fatigue and frequent urination are early signs of high blood sugar"_ → 768-dim vector $d$, computed once and stored. Similarity = $q^\top d = 0.87$. Nothing in this score knows that "early" in the query aligns with "early signs" in the document — the match is purely geometric, between two independently computed vectors.

The constraint is strong: each representation must be independently sufficient. Cross-document reasoning is impossible — the model cannot attend to tokens in $d$ while encoding $q$.

> **Bias–variance:** The independence constraint is a structural bias — relationships between query and document terms that only emerge in context cannot be modeled at encoding time. In exchange, precomputable document representations eliminate variance from repeated inference.
> {: .prompt-tip }

### 5.2 Cross-Encoders

A cross-encoder concatenates query and document and processes the pair jointly:

$$
s = f_\theta\bigl([q \,;\, \text{[SEP]} \,;\, d]\bigr)
$$

Full attention operates over both inputs simultaneously, allowing every token in $q$ to interact with every token in $d$. The output is a scalar relevance score.

**Example**: the concatenated input is _"early symptoms of diabetes [SEP] Fatigue and frequent urination are early signs of high blood sugar"_. The token "symptoms" in the query can now directly attend to "signs" in the document; "diabetes" can attend to "blood sugar". The model integrates these alignments and outputs a single score like 0.94 — much more precise than the dot product of two independent vectors, but only computable at query time.

| Property              | Bi-Encoder                    | Cross-Encoder                     |
| --------------------- | ----------------------------- | --------------------------------- |
| Encoding              | $f(q)$, $f(d)$ separately     | $f([q; d])$ jointly               |
| Precomputation        | yes — index documents offline | no — must recompute per pair      |
| Latency at query time | fast (ANN lookup)             | slow (full forward pass per pair) |
| Expressivity          | limited                       | high                              |
| Typical role          | first-stage retrieval         | second-stage reranking            |

In production retrieval systems, both are used in sequence: a bi-encoder retrieves a candidate set, a cross-encoder reranks it.

> **Bias–variance:** Cross-encoders remove the independence constraint, eliminating the structural bias of bi-encoders. The cost is mandatory per-pair inference and higher sensitivity to query–document distributional shift.
> {: .prompt-tip }

> Cross-encoders do not produce embeddings — they produce scores. The encoding is not a point in space; it is a function of a pair.
> {: .prompt-info }

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
