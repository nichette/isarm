# Reference 01: Deep Learning — Autoencoders (Ch. 14)
**Source:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
**Chapter 14: Autoencoders** — pp. 493–540.
**DOI:** 10.7551/mitpress/11808.001.0001

---

## ISARM_MAP: [Autoencoder Architecture] → `src/train_autoencoder.py::DeepAutoencoder`

### 14.1 Undercomplete Autoencoders
> "An autoencoder is a neural network that is trained to copy its input to its output... The code layer h has dimensionality smaller than the input, so the autoencoder is forced to capture the most salient features of the input distribution." (p. 494)

**ISARM Application:** Bottleneck dimension `z=16` compresses 45-dim preprocessed workforce vectors. Forces learning of "normal workforce manifold" — anomalies (flight risks) lie off-manifold.

### 14.2 Regularized Autoencoders
> "Regularized autoencoders... can learn something useful even when the encoder and decoder are linear... Sparsity regularization... encourages the model to learn features that are active for only a small fraction of inputs." (p. 502)

**ISARM Application:** `Dropout(0.2)` + `BatchNorm1d` in encoder/decoder prevents identity mapping memorization. No explicit sparsity penalty — reconstruction error itself is the signal.

### 14.3 Representational Power, Layer Size and Depth
> "Deep autoencoders... can compress data more effectively than shallow ones... The universal approximation theorem guarantees that a feedforward network with one hidden layer can represent any continuous function, but depth exponentially reduces the number of units required." (p. 498)

**ISARM Architecture:** 45 → 64 → 32 → **16** → 32 → 64 → 45 (symmetric deep encoder/decoder). Depth chosen per this principle.

### 14.4 Denoising Autoencoders
> "The denoising autoencoder... forces the autoencoder to learn a vector field pointing toward the manifold... This makes it more robust to small perturbations." (p. 508)

**ISARM Note:** Not explicitly used (clean synthetic data), but injection of structured anomalies (Types A/B/C) mimics denoising objective — model must reconstruct "true" manifold from corrupted regions.

### 14.5 Contractive Autoencoders
> "The contractive autoencoder adds an explicit regularizer... the Frobenius norm of the Jacobian of the encoder activations with respect to the input... This forces the model to be insensitive to input variations tangent to the manifold." (p. 510)

**ISARM Relevance:** Jacobian regularization would improve latent stability for RL injection — future work.

### 14.7 Applications of Autoencoders
> "Anomaly detection... is one of the most common applications of autoencoders... The anomaly score is typically the reconstruction error... Points with high reconstruction error are anomalies." (p. 524)

**ISARM Implementation:**

python
Reconstruction MSE per sample

mse = criterion(recon, batch).mean(dim=1) # (batch_size,)
Threshold = 95th percentile of TRAINING NORMAL errors

threshold = np.percentile(train_errors[y_train == 0], 95)

Directly follows this prescription.

---

## Key Equations (p. 494, 524)

| Eq. | Formula | ISARM Use |
|-----|---------|-----------|
| 14.1 | $h = f_\theta(x) = \sigma(Wx + b)$ | Encoder forward pass |
| 14.2 | $\hat{x} = g_{\theta'}(h) = \sigma(W'h + b')$ | Decoder forward pass |
| 14.3 | $L(x, \hat{x}) = \|x - \hat{x}\|^2$ | MSE Loss (reduction='none') |
| 14.20 | $\text{score}(x) = L(x, \hat{x})$ | Anomaly score = reconstruction error |

---

## Critical Design Decisions from Text

| Decision | Text Guidance | ISARM Choice |
|----------|---------------|--------------|
| Bottleneck dim | "Smaller than input... captures salient features" | 16 (vs 45 input) |
| Activation | "ReLU... avoids saturation" | ReLU + BatchNorm |
| Loss | "MSE for continuous inputs" | `nn.MSELoss(reduction='none')` |
| Threshold | "Quantile of normal data" | 95th %ile of train normals |
| Regularization | "Dropout, weight decay" | Dropout(0.2/0.1) + AdamW(wd=1e-5) |
