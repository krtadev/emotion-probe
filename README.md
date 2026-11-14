# Does the model read the user's emotion?

A linear probe on Qwen2.5-7B-Instruct's hidden state reads a user's **implicit** emotion
(frustrated / anxious / happy / neutral, no emotion words in the text) with **94% accuracy on
situations it never saw, even when topic gives nothing away**: the same 40 situations are each
written in all four emotions. A word-count classifier gets 58% on that set, and shuffled labels about 20%.

On an easier first set of 200 messages, where topic and emotion overlap, the probe scores 100%
(layer 19). That number is inflated by the overlap: the probe trained there recognises only 38% of
neutral messages on the matched set, because it had partly learned "neutral = factual question".

**Interactive write-up:** https://krtadev.github.io/emotion-probe/ (source in `docs/index.html`)

## Contents

```
emotion_probe.py          full pipeline: activations -> per-layer probes -> controls -> figures + results.json
emotion_probe.ipynb       the same pipeline as a Colab notebook (A100 / L4)
requirements.txt
data/messages.tsv         200 messages, "label<TAB>text", 50 per class, emotion words removed
results/results_colab_run.json
                          numbers from the 13 Nov 2026 Colab A100 run (used by the page)
docs/index.html           interactive results page, served by GitHub Pages
```

`results_colab_run.json` holds accuracy per layer (real and shuffled), the TF-IDF baseline,
class-direction cosines, held-out probability of the true class per message (rounded to 0.01),
and 2-D PCA coordinates for layers 4/9/14/19/23/28 (rounded to 0.1). Running the script writes
the full-precision `results/results.json`, `results/activations_last_token.npy` (float16) and
`figures/*.png`.

## Run it

```bash
pip install -r requirements.txt
python emotion_probe.py                         # needs a CUDA GPU with >= 24 GB
MODEL=meta-llama/Llama-3.1-8B-Instruct python emotion_probe.py   # any chat model works
```

Colab: open `emotion_probe.ipynb`, set the runtime to A100, upload `data/messages.tsv` into
`data/`, then Run all. About 1 minute to download the model, about 75 s for activations and
about 90 s for probes. **Disconnect and delete the runtime afterwards.**

## Method

- **Representation:** residual stream at the last prompt token (assistant turn opened), all
  29 layers, giving `X: [200, 29, 3584]`.
- **Probe:** `StandardScaler -> PCA(128) -> LogisticRegression(C=0.1)`, 5-fold stratified CV.
  PCA is fit inside each fold. Without the PCA step, CPU training takes more than 10 minutes.
- **Controls:**
  - the same probe on permuted labels;
  - TF-IDF (1–2 gram) logistic regression on the raw text.
- **Sanity check:** layer 0 = chance. The last token is the same template token for every
  message.

## Results on the first, easier set (Colab A100 run)

| layer | probe | shuffled |
|---|---|---|
| 0 | 25.0% | 25.0% |
| 1 | 86.5% | 21.5% |
| 9 | 95.0% | 23.5% |
| 14 | 99.0% | 28.5% |
| **19** | **100.0%** | 23.5% |
| 28 | 98.5% | 24.5% |
| TF-IDF baseline | 78.0% | |

## Contrast test (topic and message type held fixed)

`data/contrast.tsv` contains 40 situations, each written in all four emotions. Every emotion is
about half questions or requests. There is no "!", "?" or emoji, and no emotion words. Run it with
`python contrast_test.py`.

| test | best layer | accuracy |
|---|---|---|
| probe, held-out situations (GroupKFold) | 22 | **94.4%** |
| same, shuffled labels | | ~20% |
| TF-IDF, held-out situations | | 57.5% |
| message length only | | 38.1% |
| original probe → contrast set (transfer) | 22 | 79.4% |

On the transfer test, frustrated, anxious and happy are recognised 92–95% of the time, but only
38% of neutral messages are. The original probe had partly learned "neutral = factual question",
so neutral statements about loaded situations get read as anxious. Within the matched set,
emotion is still decodable when topic can't help. That signal builds later in the network
(layers 14–22) than the 86%-at-layer-1 seen on the easy set suggests.

## Caveats and next steps

- Small (200), single-author (LLM-written), clean cues, so 100% mostly means the set is easy.
- Topic is a likely confound: anxious messages skew towards health and deadlines, neutral ones are
  factual questions. The next set should hold the situation fixed and vary only the emotion.
- This shows the model *represents* the emotion, not that it *uses* it. Next:
  1. Score each reply for whether it adapts, and compare that with the probe score.
  2. Steer along the probe direction and see whether adaptation follows.
