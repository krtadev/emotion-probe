"""
Contrast test: does the emotion probe survive when topic and message type are held fixed?

data/contrast.tsv has 40 situations, each written four ways (frustrated / anxious / happy / neutral),
with roughly half of each class phrased as a question or request and no "!", "?" or emoji.
  A) transfer: probe trained on the original 200 messages, tested on the 160 contrast messages
  B) grouped CV: GroupKFold(5) over situations, so the test situations are never seen in training
  baselines: TF-IDF (same grouping), message length only, shuffled labels
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "\n".join("|".join(l.rstrip("\n").split("\t", 1)) for l in open(os.path.join(HERE, "data", "messages.tsv"), encoding="utf-8") if l.strip())
CONTRAST = "\n".join("|".join(l.rstrip("\n").split("\t")) for l in open(os.path.join(HERE, "data", "contrast.tsv"), encoding="utf-8") if l.strip())
import json, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
T0 = time.time()
MODEL = "Qwen/Qwen2.5-7B-Instruct"
orig = [l.split("|", 1) for l in DATA.strip().split("\n")]
con = [l.split("|", 2) for l in CONTRAST.strip().split("\n")]
classes = ["anxious", "frustrated", "happy", "neutral"]
o_txt = [r[1] for r in orig]; o_y = np.array([classes.index(r[0]) for r in orig])
c_txt = [r[2] for r in con]; c_y = np.array([classes.index(r[1]) for r in con]); c_g = np.array([int(r[0]) for r in con])
print(len(o_txt), "original,", len(c_txt), "contrast,", len(set(c_g)), "situations")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
def acts(texts):
    out = []
    with torch.no_grad():
        for t in texts:
            p = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
            hs = model(**tok(p, return_tensors="pt").to("cuda"), output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy())
    return np.stack(out)
XO, XC = acts(o_txt), acts(c_txt)
print("activations", XO.shape, XC.shape, f"{time.time()-T0:.0f}s")
del model; torch.cuda.empty_cache()

L = XO.shape[1]
mk = lambda k=100: make_pipeline(StandardScaler(), PCA(k, random_state=0), LogisticRegression(max_iter=3000, C=0.1))
gkf = GroupKFold(5)
# A) transfer: train on the original 200, test on the topic-matched 160
transfer = []
for l in range(L):
    clf = mk().fit(XO[:, l], o_y); transfer.append(float((clf.predict(XC[:, l]) == c_y).mean()))
# B) within contrast, held-out situations (every situation has all four emotions, so topic cannot help)
grouped = [float(cross_val_score(mk(), XC[:, l], c_y, cv=gkf, groups=c_g).mean()) for l in range(L)]
y_sh = np.random.default_rng(0).permutation(c_y)
grouped_shuf = [float(cross_val_score(mk(), XC[:, l], y_sh, cv=gkf, groups=c_g).mean()) for l in range(L)]
# baselines on the contrast set, same grouping
tf = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000))
tfidf_grouped = float(cross_val_score(tf(), c_txt, c_y, cv=gkf, groups=c_g).mean())
tfidf_transfer = float((tf().fit(o_txt, o_y).predict(c_txt) == c_y).mean())
wc = np.array([[len(t.split()), len(t)] for t in c_txt], float)
length_grouped = float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), wc, c_y, cv=gkf, groups=c_g).mean())
bt, bg = int(np.argmax(transfer)), int(np.argmax(grouped))
cm_t = confusion_matrix(c_y, mk().fit(XO[:, bt], o_y).predict(XC[:, bt]), normalize="true")
pred_g = cross_val_predict(mk(), XC[:, bg], c_y, cv=gkf, groups=c_g)
cm_g = confusion_matrix(c_y, pred_g, normalize="true")
miss = [{"sit": int(c_g[i]), "true": classes[c_y[i]], "pred": classes[pred_g[i]], "text": c_txt[i]} for i in np.where(pred_g != c_y)[0]]
R = {"transfer": [round(a, 3) for a in transfer], "grouped": [round(a, 3) for a in grouped], "grouped_shuf": [round(a, 3) for a in grouped_shuf],
     "tfidf_grouped": round(tfidf_grouped, 3), "tfidf_transfer": round(tfidf_transfer, 3), "length_grouped": round(length_grouped, 3),
     "best_transfer_layer": bt, "best_grouped_layer": bg, "cm_transfer": np.round(cm_t, 2).tolist(), "cm_grouped": np.round(cm_g, 2).tolist(), "misses": miss}
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(R, open(os.path.join(HERE, "results", "contrast_results.json"), "w"))
print(f"done {time.time()-T0:.0f}s"); print(json.dumps({k: v for k, v in R.items() if k != "misses"}))
