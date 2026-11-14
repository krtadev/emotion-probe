"""
Can a linear probe read the user's (implicit) emotion from an LLM's hidden state?

Run on a GPU with >= 24 GB (A100 / L4). On a Colab A100 the full run takes ~3-4 minutes
after the ~1 minute model download.

    pip install -r requirements.txt
    python emotion_probe.py            # writes results/ and figures/

Pipeline
  1. 200 user messages (data/messages.tsv), 50 each of frustrated / anxious / happy / neutral,
     with explicit emotion words removed.
  2. Each message goes through the model's chat template with the assistant turn opened.
     We keep the residual-stream hidden state at the LAST prompt token for every layer
     (embeddings + 28 blocks for Qwen2.5-7B) -> X: [200, 29, 3584].
  3. Probe per layer: StandardScaler -> PCA(128) -> LogisticRegression(C=0.1), 5-fold stratified CV.
     PCA is fit inside each fold (no leakage). Without the PCA step the probes take >10 min on CPU.
  4. Controls: same probe on shuffled labels; TF-IDF (1-2 gram) logistic regression on the raw text.
"""
import json, os, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
T0 = time.time()

# ---------------------------------------------------------------- data
rows = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(HERE, "data", "messages.tsv"), encoding="utf-8") if l.strip()]
labels_txt = [r[0] for r in rows]
texts = [r[1] for r in rows]
classes = sorted(set(labels_txt))
y = np.array([classes.index(l) for l in labels_txt])
print(len(texts), "messages", {c: labels_txt.count(c) for c in classes})

# ---------------------------------------------------------------- activations
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

feats = []
with torch.no_grad():
    for t in texts:
        p = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
        hs = model(**tok(p, return_tensors="pt").to("cuda"), output_hidden_states=True).hidden_states
        feats.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy())
X = np.stack(feats)
n_layers = X.shape[1]
np.save(os.path.join(HERE, "results", "activations_last_token.npy"), X.astype(np.float16))
print("activations:", X.shape, f"({time.time() - T0:.0f}s)")
del model
torch.cuda.empty_cache()

# ---------------------------------------------------------------- probes
cv = StratifiedKFold(5, shuffle=True, random_state=0)
mk = lambda: make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(max_iter=3000, C=0.1))
y_shuf = np.random.default_rng(0).permutation(y)
real = [cross_val_score(mk(), X[:, l], y, cv=cv).mean() for l in range(n_layers)]
shuf = [cross_val_score(mk(), X[:, l], y_shuf, cv=cv).mean() for l in range(n_layers)]
text_acc = cross_val_score(make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000)),
                           texts, y, cv=cv).mean()
best = int(np.argmax(real))
print(f"best layer {best}: {real[best]:.3f} | shuffled mean {np.mean(shuf):.3f} | TF-IDF {text_acc:.3f}")

# ---------------------------------------------------------------- figures
COL = {"anxious": "#2f6fb6", "frustrated": "#c4633f", "happy": "#d6a21e", "neutral": "#aeaca3"}
MRK = {"anxious": "o", "frustrated": "s", "happy": "^", "neutral": "D"}
plt.rcParams.update({"figure.facecolor": "#faf9f5", "axes.facecolor": "#faf9f5", "font.size": 10,
                     "axes.edgecolor": "#e8e6dc", "grid.color": "#e8e6dc", "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(real, "-o", c="#d97757", ms=4, lw=2, label="probe, real labels")
ax.plot(shuf, "-", c="#c2c0b6", lw=2, label="probe, shuffled labels")
ax.axhline(text_acc, ls="--", c="#4a86c8", label=f"word-count baseline {text_acc:.2f}")
ax.axhline(1 / len(classes), ls=":", c="#8f8d86", label="chance")
ax.set(xlabel="layer", ylabel="5-fold accuracy", ylim=(0, 1.05), title="Where does the model read the user's emotion?")
ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=.6); plt.tight_layout()
plt.savefig(os.path.join(HERE, "figures", "fig1_accuracy_by_layer.png"), dpi=150); plt.close()

show = sorted(set(np.linspace(0, n_layers - 1, 7).astype(int).tolist() + [best]))
fig, axs = plt.subplots(2, 4, figsize=(16, 7.5)); pca_out = {}
for ax, l in zip(axs.flat, show):
    Z = PCA(2).fit_transform(StandardScaler().fit_transform(X[:, l])); pca_out[l] = Z
    for c in classes:
        m = np.array(labels_txt) == c
        ax.scatter(Z[m, 0], Z[m, 1], s=18, c=COL[c], marker=MRK[c], label=c, edgecolors="none")
    ax.set_title(f"layer {l} · acc {real[l]:.2f}"); ax.set_xticks([]); ax.set_yticks([])
for ax in list(axs.flat)[len(show):]: ax.axis("off")
axs.flat[0].legend(frameon=False, fontsize=9)
fig.suptitle("User-emotion clusters across depth (PCA of last-token activations)"); plt.tight_layout()
plt.savefig(os.path.join(HERE, "figures", "fig2_pca_by_layer.png"), dpi=150); plt.close()

pred = cross_val_predict(mk(), X[:, best], y, cv=cv)
P = cross_val_predict(mk(), X[:, best], y, cv=cv, method="predict_proba")
cm = confusion_matrix(y, pred, normalize="true")
Xb = StandardScaler().fit_transform(X[:, best])
means = np.stack([Xb[y == i].mean(0) for i in range(len(classes))]); means -= means.mean(0)
means /= np.linalg.norm(means, axis=1, keepdims=True); cos = means @ means.T

# ---------------------------------------------------------------- results
R = {"model": MODEL, "n": len(texts), "classes": classes, "best_layer": best, "best_acc": round(real[best], 3),
     "real": [round(a, 3) for a in real], "shuffled": [round(a, 3) for a in shuf], "text_baseline": round(text_acc, 3),
     "chance": round(1 / len(classes), 3), "confusion": np.round(cm, 3).tolist(), "cosine": np.round(cos, 3).tolist(),
     "p_true": np.round(P[np.arange(len(y)), y], 3).tolist(),
     "misses": [{"text": texts[i], "true": classes[y[i]], "pred": classes[pred[i]]} for i in np.where(pred != y)[0]],
     "pca": {str(l): np.round(Z, 2).tolist() for l, Z in pca_out.items()}, "labels": labels_txt, "texts": texts}
json.dump(R, open(os.path.join(HERE, "results", "results.json"), "w"))
print(f"done in {time.time() - T0:.0f}s -> results/results.json, figures/")
