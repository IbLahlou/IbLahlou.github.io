---
title: Acoustic Operations and Foundations of Speech Processing
description: Core signal processing operations that underpin all speech recognition and synthesis systems.
date: 2025-11-18
categories:
  - Machine Learning
  - Data Science
  - Signal Processing
  - Speech Processing
tags:
  - DSP
  - Acoustics
  - Fourier Transform
  - Filtering
  - Convolution
pin: true
math: true
mermaid: true
image:
  path: /assets/img/panels/panel12@4x.png
---

# Introduction

**ASR (Automatic Speech Recognition)** systems transform audio waveforms into text. At the heart of this process lies the **acoustic model**—a statistical mapping from audio features to sound units called **phone**.


### What's a Phone

The minimal, perceptually isolable segment of sound—the atomic, boundary-defined unit that any hearing system (human or machine) treats as a single entity. From Greek phōnḗ (“sound/voice”), it stays deliberately pre-linguistic, pre-musical, and pre-semantic: the raw quantum from which every audible stream is built, universally accepted in speech tech, music information retrieval, bioacoustics, and forensic audio. Core contemporary representations of the phone (what people actually use in 2025 models)

- Wavelets – localized oscillatory atoms; perfect reconstruction + multi-resolution sparsity (denoising, compression, transient capture)
- Shapelets – mined discriminative subsequences; shape-focused, highly interpretable for classification
- MFCCs / Log-Mel frames – 20–40 ms perceptual coefficients; still the workhorse of legacy and lightweight systems
- CQ-Transform or Gammatone bins – constant-Q or auditory-filterbank responses; preferred for music and environmental sound
- Self-supervised embeddings (Wav2Vec 2.0, HuBERT, Audio-ALiBi, etc.) – dense vectors learned from raw audio; current state-of-the-art for almost everything
- Neural codec tokens (EnCodec, SoundStream, DAC) – discrete 50–1500 Hz tokens; the new standard for generative and ultra-low-bitrate pipelines
- Sinusoidal + residual tracks – high-quality parametric modeling for voice conversion and music

From this neutral phone, only three classic domain-specific abstractions emerge:

- Phoneme – bundle of phones that are contrastive in a given language
- Grapheme – written symbol(s) conventionally linked to a phoneme
- Note – phone whose dominant attribute is stable perceived pitch (musical domain)


**Example Illustration**

When a singer or a piano plays the first four notes of “Do–Ré–Mi–Fa”, your ears receive exactly four raw phones: four short, perceptually distinct bursts of acoustic energy with stable pitch.

That’s all that physically exists.

Everything else is just the question we decide to ask those same four phones:
- What was the word that being sung
- What pitches are these, regardless of words?
- What are the exact pitches , frequencies and how it is structure in form of data ?

The table below shows, row by row, how the exact same four phones travel through every major abstraction layer we use today


| What you actually hear (four raw phones) | Solfege syllable (human sings) | Grapheme (how we write the syllable)      | Phonemic transcription (IPA, language lens) | Note abstraction (music lens)                                     | Typical model that extracts this layer |
| ---------------------------------------- | ------------------------------ | ----------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------- |
| 4 distinct, stable pitched sounds        | Do  –  Ré  –  Mi  –  Fa        | Do  –  Ré  –  Mi  –  Fa                   | /do/  –  /ʁe/  –  /mi/  –  /fa/             | Do (C)  –  Ré (D)  –  Mi (E)  –  Fa (F)                           | –                                      |
| Same four phones                         | Do  –  Ré  –  Mi  –  Fa        | Do  –  Ré  –  Mi  –  Fa                   | /do/  –  /ʁe/  –  /mi/  –  /fa/             | C4  –  D4  –  E4  –  F4 (fixed-Do) or movable-Do position 1-2-3-4 | CREPE, Basic Pitch, mt3                |
| Same four phones                         | Do  –  Ré  –  Mi  –  Fa        | C   –  D   –  E   –  F   (staff notation) | –                                           | C4  –  D4  –  E4  –  F4                                           | Shazam, SoundHound, Spotify song ID    |
| Same four phones                         | Do  –  Ré  –  Mi  –  Fa        | –                                         | –                                           | midi: 60  –  62  –  64  –  65                                     | Any MIDI transcription model           |
| Same four phones                         | Do  –  Ré  –  Mi  –  Fa        | –                                         | –                                           | freq: 261.63 – 293.66 – 329.63 – 349.23 Hz                        | YIN, pYIN, SPICE                       |
| Same four phones (neural codec view)     | Do  –  Ré  –  Mi  –  Fa        | –                                         | –                                           | token seq: [234][567][891]…                                       | EnCodec / Descript Audio Codec         |
| Same four phones (embedding view)        | Do  –  Ré  –  Mi  –  Fa        | –                                         | –                                           | 1024-dim vectors × 4                                              | HuBERT / AudioLM                       |

The crucial mapping in one line:

**Do Ré Mi Fa** (sung syllables)  
→ **/do ʁe mi fa/** (phonemic, language treats them as spoken words)  
→ **Do, Ré, Mi, Fa** (graphemic, solfège spelling)  
→ **C4 D4 E4 F4** or simply **Do Ré Mi Fa** (note abstraction, music treats the exact same phones as pure pitch classes)


### The Acoustic Model Pipeline

```
Raw audio (48–96 kHz PCM)  
→ Phone segmentation (onset, boundary, or change-point detection)  
→ Short-time framing & windowing (10–50 ms frames, 10–25 ms hop)  
→ Time-frequency representation  
    ├─ STFT (most common)  
    ├─ Constant-Q Transform (CQT) – music & pitched sources  
    ├─ Continuous/Discrete Wavelet Transform – transients & non-stationary signals  
    └─ Gammatone or ERB filterbanks – auditory modeling  
→ Perceptual compression  
    └─ Mel / Bark / ERB scaling + logarithmic energy (standard)  
→ Representation layer (multiple parallel paths coexist in 2025)  
    ├─ DCT → MFCCs / GFCCs (legacy, lightweight devices)  
    ├─ Convolutional / Transformer front-ends → dense embeddings (Wav2Vec 2.0, HuBERT, Audio-ALiBi, BEATs)  
    ├─ Vector quantization → discrete neural tokens (EnCodec, SoundStream, Descript Audio Codec, DAC)  
    └─ Parametric decomposition → sinusoids + residual (DDSP, Diff-PTC)  
→ Temporal sequence modeling  
    └─ Conformer, Transformer, Mamba/State-Space, or CRNN architectures  
→ Task-specific symbolic decoder (final divergence point)  
    ├─ Phonemes → Graphemes → Text (speech & lyrics)  
    ├─ Notes / Chords / Pitch classes → MIDI or music notation (instrumental & vocal music)  
    ├─ Solfege or scale degrees (Do–Ré–Mi systems)  
    ├─ Sound-event / scene labels (AudioSet, VGGSound, industrial monitoring)  
    ├─ Bioacoustic call types (bird, whale, bat, anuran)  
```
This unified pipeline is effectively domain-agnostic through the sequence-modeling stage. The same pretrained front-end (e.g., HuBERT Large, Audio-ALiBi-XXL, or EnCodec 48 kHz) can be fine-tuned or directly used for speech recognition, music transcription, environmental sound classification, or generative synthesis with only the decoder and loss function changed. Classic DSP remains the foundation; contemporary neural architectures have internalized and extended it rather than replaced it.

This article covers the mathematical operations in this pipeline. Understanding these fundamentals is essential—neural networks don't replace DSP (Digital Signal Processing), they build upon it.

---

## 0. Digital Signal Processing Foundations Recap

These operations are covered in detail in previous posts. Here's the quick reference for speech processing:

**Convolution** $(x * h)[n]$ 

 — how filters and systems transform signals. Convolution in time = multiplication in frequency (FFT—Fast Fourier Transform—speedup).

**Correlation** $R_{xx}[n]$ 

— measures self-similarity at different lags. Peaks in autocorrelation reveal pitch period.

**DFT (Discrete Fourier Transform)** $X[k] = \sum_n x[n] e^{-i2\pi kn/N}$ 

— decomposes signal into frequency components. Magnitude = "how much", phase = "when".

**Windowing** 

— When we extract a finite frame from a continuous signal, the abrupt edges create artificial discontinuities. The DFT assumes periodicity, so these sharp cuts cause **spectral leakage**—energy spreading into adjacent frequency bins where it shouldn't be.

A window function tapers the frame smoothly to zero at the edges:

| Window      | Main Lobe Width | Side Lobe Level  | Use Case                     |
| ----------- | --------------- | ---------------- | ---------------------------- |
| Rectangular | Narrowest       | Highest (-13 dB) | Maximum frequency resolution |
| Hamming     | Medium          | Low (-43 dB)     | General speech analysis      |
| Hann        | Medium          | Lower (-31 dB)   | Spectral analysis            |
| Blackman    | Widest          | Lowest (-58 dB)  | When leakage must be minimal |

**Trade-off:** Narrower main lobe = better frequency resolution. Lower side lobes = less leakage. You can't optimize both—this is the **time-frequency uncertainty principle**.

> For speech processing, **Hamming window** is the standard choice: good balance between frequency resolution and leakage suppression.
{: .prompt-tip }

**STFT (Short-Time Fourier Transform)** $X[m,k]$ — sliding window DFT producing spectrograms. For speech: 20-30ms frames, 10ms hop. Time-frequency uncertainty: $\Delta t \cdot \Delta f \geq 1/4\pi$.

---

## 1. Filtering - Frequency Selection

With these foundational operations established, we can now build speech processing systems. The first step is often **filtering**—selectively passing or blocking certain frequencies to prepare the signal for analysis.

<!-- TODO: Add images when available
<!-- ![Filtering Dark Mode](../assets/img/graphics/post_12/dark/img1_filtering.png){: .dark } -->
<!-- ![Filtering Light Mode](../assets/img/graphics/post_12/light/img1_filtering.png){: .light } -->
_Figure 1.0: Filter types and their frequency responses_
-->

### Filter Types

| Type      | Passes                 | Blocks          | Use Case                  |
| --------- | ---------------------- | --------------- | ------------------------- |
| Low-pass  | $f < f_c$              | $f > f_c$       | Anti-aliasing, smoothing  |
| High-pass | $f > f_c$              | $f < f_c$       | Remove DC, pre-emphasis   |
| Band-pass | $f_1 < f < f_2$        | else            | Filterbank channels       |
| Band-stop | $f < f_1$ or $f > f_2$ | $f_1 < f < f_2$ | Notch filter (remove hum) |

### FIR vs IIR

Digital filters come in two fundamental types, distinguished by whether they use feedback.

**FIR (Finite Impulse Response):** A filter whose output depends only on current and past inputs—no feedback.

$$y[n] = \sum_{k=0}^{M} b_k \cdot x[n-k]$$

**Intuition:** Output is a weighted sum of the current and past $M$ input samples only. If you feed an impulse (single spike), the output dies after $M$ samples—hence "finite."

| Property            | Explanation                                    |
| ------------------- | ---------------------------------------------- |
| Always stable       | No feedback → no risk of runaway oscillation   |
| Linear phase        | Symmetric coefficients preserve waveform shape |
| Higher order needed | Need many taps for sharp cutoffs               |

**IIR (Infinite Impulse Response):** A filter with feedback—output depends on past outputs too.

$$y[n] = \sum_{k=0}^{M} b_k \cdot x[n-k] - \sum_{k=1}^{N} a_k \cdot y[n-k]$$

**Intuition:** Output depends on past inputs AND past outputs (feedback). An impulse can theoretically ring forever—hence "infinite." The feedback creates resonances that achieve sharp frequency responses with fewer coefficients.

| Property         | Explanation                                                                   |
| ---------------- | ----------------------------------------------------------------------------- |
| Can be unstable  | Feedback can cause output to explode if poles outside unit circle             |
| Lower order      | Feedback provides "free" filtering; 2nd-order IIR ≈ 50th-order FIR            |
| Phase distortion | Different frequencies delayed differently (problematic for some applications) |

> IIR filters can become unstable if not designed carefully. Always verify that all poles are inside the unit circle in the z-plane.
{: .prompt-warning }

**In speech processing:**

- **FIR:** Mel filterbanks (linear phase preserves temporal structure)
- **IIR:** Pre-emphasis (simple 1st-order, low latency)

### Pre-emphasis Filter

$$y[n] = x[n] - \alpha x[n-1], \quad \alpha \approx 0.97$$

**Transfer function:** $H(z) = 1 - \alpha z^{-1}$

**Frequency response:** $\lvert H(e^{i\omega}) \rvert = \sqrt{1 + \alpha^2 - 2\alpha\cos\omega}$

Boosts high frequencies ~6 dB/octave.

### Why Pre-emphasis?

The glottal source (vocal cord vibration) has a natural spectral tilt: energy decreases ~20 dB/decade at higher frequencies. Pre-emphasis compensates for this, giving high-frequency formants more weight in analysis. Without it, low frequencies would dominate MFCC computation.

---

## 2. Cepstral Analysis

Now that we can filter and shape the spectrum, we need a way to **separate** the two main components of speech: the excitation source (vocal cords) and the vocal tract filter. The cepstrum provides exactly this capability.

<!-- ![Cepstrum Dark Mode](../assets/img/graphics/post_12/dark/img7_cepstrum.png){: .dark } -->
<!-- ![Cepstrum Light Mode](../assets/img/graphics/post_12/light/img7_cepstrum.png){: .light } -->
_Figure 2.0: Cepstrum separating source and filter_

### Homomorphic Deconvolution

Speech = source _ filter: $s[n] = e[n] _ h[n]$

In frequency: $S(f) = E(f) \cdot H(f)$

Take log: $\log S = \log E + \log H$

**Cepstrum:** inverse DFT of log magnitude spectrum

$$c[n] = \mathcal{F}^{-1}\{\log \lvert X[k] \rvert\}$$

### Quefrency Domain

**Quefrency** is the independent variable in the cepstral domain—it has units of time (samples or milliseconds) but represents "rate of change in the spectrum." The name is an anagram of "frequency," following the cepstrum/spectrum wordplay.

- Low quefrency: slow spectral variations (vocal tract = formants)
- High quefrency: fast spectral variations (pitch harmonics)

### Liftering

Keep only low-quefrency components:

$$\hat{c}[n] = c[n] \cdot l[n]$$

where $l[n]$ is a low-pass lifter.

### Cepstrum Intuition

**Etymology:** "Cepstrum" is an anagram of "spectrum"—we're analyzing the spectrum of a spectrum.

**Separation principle:** The vocal tract (slow-varying formants) appears at low quefrencies. The pitch harmonics (fast-varying) appear at high quefrencies. Liftering removes pitch, leaving vocal tract shape—the basis for speaker-independent recognition.

**Connection to MFCCs:** MFCCs are essentially cepstral coefficients computed on a mel-warped spectrum. The DCT decorrelates the log mel energies, producing a compact representation of the spectral envelope.

---

## 3. Mel-Frequency Analysis

The cepstrum works on linear frequency. But human hearing doesn't perceive frequencies linearly—we're more sensitive to differences at low frequencies than high. The **mel scale** models this perception, leading to **MFCCs (Mel-Frequency Cepstral Coefficients)**—the most widely used features in speech recognition.

<!-- ![Mel Scale Dark Mode](../assets/img/graphics/post_12/dark/img8_mel.png){: .dark } -->
<!-- ![Mel Scale Light Mode](../assets/img/graphics/post_12/light/img8_mel.png){: .light } -->
_Figure 3.0: Mel filterbank on linear frequency axis_

### Mel Scale

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

$$f = 700 \cdot \left(10^{m/2595} - 1\right)$$

**Perceptual motivation:** Equal mel intervals = equal perceived pitch intervals.

### Mel Filterbank

Triangular filters uniformly spaced in mel domain:

$$
H_m[k] = \begin{cases}
0 & k < f[m-1] \\
\frac{k - f[m-1]}{f[m] - f[m-1]} & f[m-1] \leq k < f[m] \\
\frac{f[m+1] - k}{f[m+1] - f[m]} & f[m] \leq k < f[m+1] \\
0 & k \geq f[m+1]
\end{cases}
$$

### Filterbank Energies

$$E_m = \sum_{k=0}^{N/2} \lvert X[k] \rvert^2 \cdot H_m[k]$$

### MFCC Computation

1. Compute power spectrum: $\lvert X[k] \rvert^2$
2. Apply mel filterbank: $E_m$
3. Log compress: $\log E_m$
4. DCT: $c_i = \sum_{m=1}^{M} \log E_m \cdot \cos\left(\frac{\pi i (m-0.5)}{M}\right)$

### Why DCT for MFCCs?

**Decorrelation:** Mel filterbank outputs are correlated (adjacent filters overlap). DCT produces uncorrelated coefficients, beneficial for diagonal-covariance GMMs.

**Energy compaction:** Most speech information concentrates in the first 12-13 coefficients. Higher coefficients represent fine spectral detail (often discarded).

### Dynamic Features: Deltas and Delta-Deltas

Static MFCCs capture spectral shape at a single instant. Speech is inherently dynamic—phoneme transitions carry critical information.

**Delta coefficients** (velocity): First derivative of MFCCs across time
$$\Delta c_t = \frac{\sum_{n=1}^{N} n(c_{t+n} - c_{t-n})}{2\sum_{n=1}^{N} n^2}$$

**Delta-delta coefficients** (acceleration): Second derivative, computed the same way on deltas.

| Coefficient Type | Captures          | Example                     |
| ---------------- | ----------------- | --------------------------- |
| Static MFCC      | Spectral envelope | Vowel identity              |
| Delta            | Rate of change    | Consonant-vowel transitions |
| Delta-delta      | Acceleration      | Emphasis, speaking rate     |

**Standard feature vector:** 39 dimensions per frame

- 13 static (12 MFCCs + energy)
- 13 delta
- 13 delta-delta

> The 39-dimensional MFCC+delta+delta-delta feature vector has been the de facto standard for speech recognition for decades. Even with modern neural approaches, it remains a strong baseline.
{: .prompt-info }

This captures both "what sound" and "how it's changing"—essential for distinguishing coarticulated phonemes.

---

## 4. Discrete Cosine Transform (DCT)

The **DCT (Discrete Cosine Transform)** is a transform similar to the DFT but uses only cosine functions, producing real-valued coefficients. It's widely used in compression (JPEG, MP3) because it concentrates signal energy into fewer coefficients than the DFT.

<!-- ![DCT Dark Mode](../assets/img/graphics/post_12/dark/img9_dct.png){: .dark } -->
<!-- ![DCT Light Mode](../assets/img/graphics/post_12/light/img9_dct.png){: .light } -->
_Figure 4.0: DCT basis functions and energy compaction_

### Definition (DCT-II)

$$C[k] = \sum_{n=0}^{N-1} x[n] \cdot \cos\left(\frac{\pi k (2n+1)}{2N}\right)$$

### Why DCT?

1. **Real-valued:** No complex numbers
2. **Energy compaction:** Most energy in first few coefficients
3. **Decorrelation:** Approximates the KLT (Karhunen-Loève Transform, the optimal decorrelating transform) for Markov-1 signals

### DCT vs DFT

| Property   | DFT           | DCT                 |
| ---------- | ------------- | ------------------- |
| Values     | Complex       | Real                |
| Assumes    | Periodic      | Symmetric extension |
| Boundary   | Discontinuity | Smooth              |
| Compaction | Good          | Better              |

---

## 5. Linear Prediction (LPC)

MFCCs capture spectral shape through filterbanks. **LPC (Linear Predictive Coding)** takes a different approach: it models the vocal tract as an **all-pole filter** and finds coefficients that best predict the signal. This yields another powerful representation—one that's particularly useful for speech coding and formant analysis.

**Formants** are the resonance frequencies of the vocal tract (labeled F1, F2, F3...). They determine vowel identity—for example, the difference between /i/ ("ee") and /a/ ("ah") is primarily in F1 and F2 positions.

<!-- ![LPC Dark Mode](../assets/img/graphics/post_12/dark/img10_lpc.png){: .dark } -->
<!-- ![LPC Light Mode](../assets/img/graphics/post_12/light/img10_lpc.png){: .light } -->
_Figure 5.0: Linear prediction as all-pole filter modeling_

### The Model

Predict current sample from past samples:

$$\hat{x}[n] = -\sum_{k=1}^{p} a_k \cdot x[n-k]$$

Prediction error: $e[n] = x[n] - \hat{x}[n]$

### All-Pole Filter

$$H(z) = \frac{1}{1 + \sum_{k=1}^{p} a_k z^{-k}} = \frac{1}{A(z)}$$

Models vocal tract transfer function (resonances = formants).

### Solving for Coefficients

Minimize mean squared error:

$$E = \sum_n e^2[n] = \sum_n \left(x[n] + \sum_{k=1}^{p} a_k x[n-k]\right)^2$$

Take derivatives, set to zero → **Yule-Walker equations:**

$$\sum_{k=1}^{p} a_k R[i-k] = -R[i], \quad i = 1, \ldots, p$$

where $R[k]$ is autocorrelation.

The **Yule-Walker equations** (named after statisticians George Udny Yule and Gilbert Walker) form a linear system that relates the LPC coefficients to the autocorrelation of the signal. The resulting matrix is **Toeplitz**—a special structure where each descending diagonal contains the same value. This structure enables efficient algorithms.

### Levinson-Durbin Algorithm

Solving Yule-Walker directly requires $O(p^3)$ operations (matrix inversion). The Levinson-Durbin algorithm exploits the **Toeplitz structure** of the autocorrelation matrix to solve it in $O(p^2)$.

**Key insight:** The solution for order $i$ can be built from order $i-1$. We compute coefficients recursively:

**Algorithm steps:**

1. **Initialize:** $E_0 = R[0]$ (signal energy)

2. **For each order $i = 1, 2, \ldots, p$:**

   Compute reflection coefficient:
   $$k_i = \frac{R[i] + \sum_{j=1}^{i-1} a_j^{(i-1)} R[i-j]}{E_{i-1}}$$

   Update coefficients:
   $$a_i^{(i)} = k_i$$
   $$a_j^{(i)} = a_j^{(i-1)} + k_i \cdot a_{i-j}^{(i-1)}, \quad j = 1, \ldots, i-1$$

   Update prediction error:
   $$E_i = (1 - k_i^2) E_{i-1}$$

3. **Output:** Final coefficients $a_1, \ldots, a_p$

**Reflection coefficients $k_i$:**

These have a physical interpretation—they represent the reflection at each "stage" of a lattice filter (like acoustic reflections in a tube model of the vocal tract).

**Stability guarantee:** If $|k_i| < 1$ for all $i$, the filter is stable. This is always true when computed from valid autocorrelation (positive definite).

> Unlike general IIR filter design, Levinson-Durbin always produces stable filters when starting from a valid autocorrelation sequence—no need for manual stability checks.
{: .prompt-info }

### Applications

- Speech coding (LPC-10, CELP)
- Formant estimation
- Speaker recognition

---

## 6. Fundamental Frequency (F0) Estimation

So far we've focused on the vocal tract (formants, spectral envelope). But the other critical component is the **excitation source**—specifically, the fundamental frequency or pitch.

**F0 (Fundamental Frequency)** is the rate at which the vocal cords vibrate during voiced speech—it determines the perceived pitch. F0 carries prosodic information: intonation, stress, emotion. Estimating it reliably is essential for many applications.

<!-- ![Pitch Detection Dark Mode](../assets/img/graphics/post_12/dark/img11_pitch.png){: .dark } -->
<!-- ![Pitch Detection Light Mode](../assets/img/graphics/post_12/light/img11_pitch.png){: .light } -->
_Figure 6.0: Pitch detection methods_

### Autocorrelation Method

Find first major peak in autocorrelation:

$$R[k] = \sum_n x[n] \cdot x[n+k]$$

Pitch period $T_0$ = lag of first peak after $R[0]$.

F0 = $f_s / T_0$

### Cepstral Method

Peak in cepstrum at quefrency = pitch period.

### RAPT / YAAPT / DIO

<!--
RESEARCH & EXPLAIN:
- Robust algorithms for real-world signals
- Handling unvoiced segments
- Octave errors
-->

### Typical Ranges

- Male: 80-200 Hz
- Female: 150-350 Hz
- Child: 200-500 Hz

---

## 7. Modulation and Demodulation

Speech can be viewed as a slowly-varying envelope (amplitude modulation) riding on rapidly-varying carriers (formants). Extracting these modulations provides yet another perspective on the signal—one that connects to neural processing of speech and alternative feature representations.

<!-- ![Modulation Dark Mode](../assets/img/graphics/post_12/dark/img12_modulation.png){: .dark } -->
<!-- ![Modulation Light Mode](../assets/img/graphics/post_12/light/img12_modulation.png){: .light } -->
_Figure 7.0: AM, FM, and the analytic signal_

### Amplitude Modulation

$$y(t) = x(t) \cdot \cos(2\pi f_c t)$$

Envelope: $\lvert x(t) \rvert$

### Hilbert Transform and Analytic Signal

$$\hat{x}(t) = \mathcal{H}\{x(t)\} = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t-\tau} d\tau$$

**Analytic signal:** $z(t) = x(t) + i\hat{x}(t)$

**Instantaneous amplitude:** $A(t) = \lvert z(t) \rvert$

**Instantaneous frequency:** $f(t) = \frac{1}{2\pi} \frac{d\phi(t)}{dt}$

### Applications in Speech

- Envelope extraction for ASR features
- F0 estimation via instantaneous frequency
- Modulation spectrum analysis

## 8. Acoustic Model Architectures

The acoustic model maps feature sequences to phoneme sequences. Two main approaches:

### Hidden Markov Models (HMMs)

An **HMM (Hidden Markov Model)** is a statistical model where the system transitions between hidden states, and each state produces observable outputs with some probability. For speech: the hidden states are phoneme sub-units, and the observations are acoustic features.

Traditional approach modeling temporal variability:

- Each phoneme = sequence of HMM states (typically 3: onset, middle, offset)
- **Emission probabilities:** GMMs (Gaussian Mixture Models) model the probability of observing features in each state. A **GMM** represents a distribution as a weighted sum of multiple Gaussian (bell-curve) distributions.
- **Transition probabilities:** Model phoneme duration

**Strengths:** Interpretable, handles variable-length sequences naturally.

**Weaknesses:** GMMs assume feature independence, limited modeling capacity.

### Deep Neural Networks (DNNs)

A **DNN (Deep Neural Network)** is a neural network with multiple hidden layers. Evolution of architectures:

| Era | Architecture | Approach |
| --- | ------------ | -------- |
| 2012+ | DNN-HMM hybrid | DNN replaces GMM for emission probabilities |
| 2015+ | LSTM/GRU | Recurrent networks with CTC loss |
| 2017+ | Transformer | Attention-based, parallel training |
| 2020+ | Self-supervised | Pre-trained representations |

### Modern Approach: Self-Supervised Speech Embeddings

Traditional MFCCs are hand-crafted features. Modern systems learn representations directly from raw audio using self-supervised learning.

**Wav2Vec 2.0** (Facebook/Meta, 2020): Learns speech representations by predicting masked portions of the audio. Pre-trained on 60k hours of unlabeled speech, then fine-tuned on small labeled datasets.

> Wav2Vec 2.0 achieves strong ASR results with just 10 minutes of labeled data—a massive reduction from traditional systems requiring thousands of hours.
{: .prompt-info }

**HuBERT** (Hidden-Unit BERT): Similar approach but uses offline clustering to create pseudo-labels for masked prediction.

**Whisper** (OpenAI, 2022): Trained on 680k hours of weakly-supervised data. Robust to accents, background noise, and technical language.

> Whisper is particularly useful for real-world applications due to its robustness to noise and ability to handle multiple languages without explicit language identification.
{: .prompt-tip }

These models output **embeddings**—dense vector representations that capture phonetic, speaker, and linguistic information. They can replace or augment traditional MFCC pipelines:

```
Traditional: Audio → MFCCs → Acoustic Model → Text
Modern:      Audio → Wav2Vec/Whisper → Fine-tuning → Text
```

**Why embeddings work:** Self-supervised pre-training on massive unlabeled data learns universal speech representations. Fine-tuning adapts these to specific tasks with minimal labeled data.

### Example: LSTM Acoustic Model

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input, Model

def build_acoustic_model(num_features, num_hidden, num_phonemes):
    input_features = Input(shape=(None, num_features))
    x = LSTM(num_hidden, return_sequences=True)(input_features)
    output_phonemes = Dense(num_phonemes, activation='softmax')(x)
    model = Model(inputs=input_features, outputs=output_phonemes)
    return model

# Typical configuration
num_features = 39   # 13 MFCCs + 13 deltas + 13 delta-deltas
num_hidden = 256    # LSTM units
num_phonemes = 40   # English phoneme set

model = build_acoustic_model(num_features, num_hidden, num_phonemes)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Input:** (batch, time_steps, 39) — sequence of MFCC frames
**Output:** (batch, time_steps, 40) — phoneme probabilities per frame

---

## References

<!--
ADD YOUR OWN:
- DSP textbooks you used
- Tutorials that helped
-->

1. Oppenheim & Schafer - Discrete-Time Signal Processing
2. Rabiner & Schafer - Digital Processing of Speech Signals
3. Quatieri - Discrete-Time Speech Signal Processing
4. Smith - Mathematics of the DFT
