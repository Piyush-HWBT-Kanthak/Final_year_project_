"""
Step 2: fine-tuned SapBERT + simple models -> hybrid, evaluated on the KI-DDI official split.

RUN (nothing else to do; it downloads the 4 small data files itself):
    pip install scikit-learn numpy transformers
    pip install torch            # must be a CUDA build so your RTX 3050 is used
    python step2_hybrid.py

Defaults are sized for a 4 GB GPU (fp16 + gradient checkpointing, batch 4 x accum 2).
Rough time on a 3050: ~20-40 min per seed. Use --seeds 1 --epochs 5 for a quick first try.
If you get a CUDA out-of-memory error:  python step2_hybrid.py --bs 2 --accum 4 --max-len 256

HONESTY RULES BUILT IN
  * The test set is used ONLY for the final numbers. Epoch, C, and fusion weight are chosen on validation.
  * Two tracks are reported:
      TEXT-ONLY  : uses the dialogue text only (realistic, no annotated symptoms)
      ORACLE     : also uses the dataset's ANNOTATED symptom lists (not available in a real deployment)
  * It tells you whether the improvement is statistically supported. It cannot guarantee a win.
Dataset terms (KI-DDI repo readme): research / non-commercial use, attribute the authors.
"""
import argparse, json, os, random, re, sys, time, urllib.request

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RAW = "https://raw.githubusercontent.com/NLP-RL/KI-DDI/main/"
FILES = ["data_dict.p", "full_dialog/final_combined_all_dialog.p",
         "full_dialog/new_dialog_id_list.p", "full_dialog/new_dialog_symptom_map.p"]
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
PAPER = {"acc": 0.6410, "macro_f1": 0.6035, "top3": 0.868, "top5": 0.9401}  # KI-DDI, as reported
N_CLASSES = 90


# ----------------------------------------------------------------------------- data
def ensure_data(root):
    import pickle
    for f in FILES:
        path = os.path.join(root, f)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {f} ...")
        urllib.request.urlretrieve(RAW + f, path)
    L = lambda p: pickle.load(open(os.path.join(root, p), "rb"))
    d = L("data_dict.p")
    fd, ids, smap = L(FILES[1]), L(FILES[2]), L(FILES[3])
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    lookup = {norm(r["dialog"]): i for r, i in zip(fd, ids)}
    out = {}
    for part in ["train", "valid", "test"]:
        xs = d[f"{part}_data"]
        y = np.array(d[f"{part}_label"]).astype(int)
        did = [lookup[norm(x)] for x in xs]
        out[part] = (xs, y, did)
    assert not (set(out["train"][2]) & set(out["test"][2])), "train/test overlap"
    return out, smap


def prep(s):  # light cleanup of the speaker tags; [SEP] and [MASK] are kept
    return s.replace("[PAT] :", "Patient:").replace("[DOC] :", "Doctor:")


# ----------------------------------------------------------------------------- metrics
def full_proba(model, X):  # LR probabilities mapped into a fixed 90-class array
    P = np.zeros((X.shape[0], N_CLASSES))
    P[:, model.classes_] = model.predict_proba(X)
    return P


def topk(P, y, k):
    top = np.argsort(-P, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))


def metrics(P, y):
    pred = P.argmax(1)
    return {"acc": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "top3": topk(P, y, 3), "top5": topk(P, y, 5)}


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))


def boot_ci(correct, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    s = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return np.percentile(s, [2.5, 97.5])


def paired_diff(c_a, c_b, n=5000, seed=1):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(c_a), len(c_a))
        d.append(c_a[i].mean() - c_b[i].mean())
    d = np.array(d)
    return float(c_a.mean() - c_b.mean()), np.percentile(d, [2.5, 97.5]), float((d > 0).mean())


# ----------------------------------------------------------------------------- neural model
def build_tokenizer_and_model(dry, texts):
    from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel, BertTokenizerFast
    if not dry:
        return AutoTokenizer.from_pretrained(MODEL_NAME), AutoModel.from_pretrained(MODEL_NAME)
    # offline tiny model, only to test the code path (no download)
    import tempfile
    toks = sorted({t for s in texts for t in re.findall(r"\w+|[^\w\s]", prep(s).lower())})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [t for t in toks if t not in ("[", "]")]
    vf = os.path.join(tempfile.mkdtemp(), "vocab.txt")
    open(vf, "w").write("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=vf, do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, max_position_embeddings=512)
    return tok, BertModel(cfg)


def train_one_seed(seed, data, args, device, log):
    import torch
    from torch import nn
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    tr_x, tr_y, _ = data["train"]; va_x, va_y, _ = data["valid"]; te_x, te_y, _ = data["test"]
    tok, enc = build_tokenizer_and_model(args.dry_run, tr_x + va_x + te_x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(enc.config.hidden_size, N_CLASSES)

        def forward(self, ids, mask):
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]  # [CLS]
            return self.head(self.drop(h))

    net = Net().to(device)
    if not args.dry_run:
        net.enc.gradient_checkpointing_enable()
    use_amp = device.type == "cuda"
    try:
        from torch.amp import GradScaler
        scaler = GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def batchify(texts, idx):
        b = tok([prep(texts[i]) for i in idx], padding=True, truncation=True,
                max_length=args.max_len, return_tensors="pt")
        return b["input_ids"].to(device), b["attention_mask"].to(device)

    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": args.lr},
                             {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    steps_per_epoch = int(np.ceil(len(tr_x) / (args.bs * args.accum)))
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    @torch.no_grad()
    def predict(texts):
        net.eval()
        out = []
        for i in range(0, len(texts), 16):
            ids, mask = batchify(texts, list(range(i, min(i + 16, len(texts)))))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                lg = net(ids, mask)
            out.append(torch.softmax(lg.float(), -1).cpu().numpy())
        return np.concatenate(out)

    best, best_state, bad = (-1, 9e9), None, 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        order = rng.permutation(len(tr_x))
        micro = [order[i:i + args.bs] for i in range(0, len(order), args.bs)]
        tot_loss = 0.0
        opt.zero_grad(set_to_none=True)
        for k, idx in enumerate(micro):
            ids, mask = batchify(tr_x, idx)
            y = torch.tensor(tr_y[idx], device=device)
            try:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = crit(net(ids, mask).float(), y) / args.accum
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                log("CUDA out of memory. Re-run with:  python step2_hybrid.py --bs 2 --accum 4 --max-len 256")
                sys.exit(1)
            tot_loss += loss.item() * args.accum
            if (k + 1) % args.accum == 0 or k == len(micro) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        Pv = predict(va_x)
        score = (accuracy_score(va_y, Pv.argmax(1)), -nll(Pv, va_y))
        log(f"  seed {seed} epoch {ep + 1:2d}/{args.epochs}  train_loss={tot_loss / len(micro):.3f}  "
            f"valid_acc={score[0]:.4f}  valid_nll={-score[1]:.3f}  ({(time.time() - t0) / 60:.1f} min)")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log("  early stop"); break
    net.load_state_dict(best_state)
    Pv, Pt = predict(va_x), predict(te_x)
    del net, opt, enc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return Pv, Pt


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./kiddi_data")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--out", default="step2_results.json")
    ap.add_argument("--dry-run", action="store_true", help="tiny random model on CPU, only tests the code")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    log = lambda m: print(m, flush=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not (args.dry_run or args.allow_cpu):
        sys.exit("No CUDA GPU found. Install a CUDA build of torch (https://pytorch.org/get-started/locally/), "
                 "or run on Google Colab. (Use --allow-cpu to force CPU; it will be very slow.)")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        log(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")

    data, smap = ensure_data(args.data_dir)
    tr_x, tr_y, tr_id = data["train"]; va_x, va_y, va_id = data["valid"]; te_x, te_y, te_id = data["test"]
    log(f"split: train={len(tr_x)} valid={len(va_x)} test={len(te_x)}")

    # ---- simple models (chosen on validation only) ----
    clean = lambda s: re.sub(r"\[(PAT|DOC|SEP|MASK)\]", "", s)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(map(clean, tr_x)); Xva = vec.transform(map(clean, va_x)); Xte = vec.transform(map(clean, te_x))
    allsym = sorted({s for v in smap.values() for s in v}); sidx = {s: i for i, s in enumerate(allsym)}

    def multihot(ids):
        M = np.zeros((len(ids), len(allsym)), dtype=np.float32)
        for r, i in enumerate(ids):
            for s in smap[i]:
                M[r, sidx[s]] = 1
        return M
    Str, Sva, Ste = multihot(tr_id), multihot(va_id), multihot(te_id)

    def fit_lr(A, B, grid):
        best = None
        for C in grid:
            m = LogisticRegression(C=C, max_iter=3000).fit(A, tr_y)
            a = accuracy_score(va_y, m.predict(B))
            if best is None or a > best[0]:
                best = (a, C, m)
        return best[2], best[1]
    m_txt, C_txt = fit_lr(Xtr, Xva, [1, 10, 100])
    m_sym, C_sym = fit_lr(Str, Sva, [0.3, 1, 10, 100])
    PV = {"tfidf": full_proba(m_txt, Xva), "sym": full_proba(m_sym, Sva)}
    PT = {"tfidf": full_proba(m_txt, Xte), "sym": full_proba(m_sym, Ste)}

    # ---- neural model, several seeds ----
    nn_v, nn_t = [], []
    seeds = [123 + i for i in range(args.seeds)]
    for s in seeds:
        log(f"\n=== training seed {s} ===")
        pv, pt = train_one_seed(s, data, args, device, log)
        nn_v.append(pv); nn_t.append(pt)
        log(f"  seed {s}: test acc (for the record) = {accuracy_score(te_y, pt.argmax(1)):.4f}")

    # ---- fusion; weight chosen on validation NLL ----
    def best_w(Pa, Pb):
        grid = np.linspace(0, 1, 11)
        return float(grid[int(np.argmin([nll(w * Pa + (1 - w) * Pb, va_y) for w in grid]))])

    systems = {}  # name -> list of test-probability arrays (one per seed)  [track]
    def add(name, track, plist):
        systems[name] = (track, plist)

    add("SapBERT fine-tuned (alone)", "TEXT-ONLY", nn_t)
    add("TF-IDF text LR (alone)", "TEXT-ONLY", [PT["tfidf"]])
    add("Hybrid: SapBERT + TF-IDF (equal avg)", "TEXT-ONLY", [(p + PT["tfidf"]) / 2 for p in nn_t])
    add("Annotated-symptom LR (alone)", "ORACLE", [PT["sym"]])
    add("Hybrid: SapBERT + symptoms (equal avg)", "ORACLE", [(p + PT["sym"]) / 2 for p in nn_t])
    add("Hybrid: SapBERT + symptoms + TF-IDF (avg)", "ORACLE", [(p + PT["sym"] + PT["tfidf"]) / 3 for p in nn_t])
    ws = [best_w(pv, PV["sym"]) for pv in nn_v]
    add(f"Hybrid: SapBERT + symptoms (val-tuned w~{np.mean(ws):.1f})", "ORACLE",
        [w * pt + (1 - w) * PT["sym"] for w, pt in zip(ws, nn_t)])

    # ---- report ----
    res = {}
    log("\n" + "=" * 100)
    log(f"TEST RESULTS (n={len(te_y)}); mean +- std over {len(seeds)} seeds for neural systems")
    log("=" * 100)
    for track in ["TEXT-ONLY", "ORACLE"]:
        log(f"\n[{track}]" + ("  (uses annotated symptoms; not available in a real deployment)" if track == "ORACLE" else ""))
        log("%-52s %-15s %-15s %-8s %-8s" % ("system", "accuracy", "macro-F1", "top3", "top5"))
        for name, (tr, plist) in systems.items():
            if tr != track:
                continue
            ms = [metrics(P, te_y) for P in plist]
            f = lambda k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms]))
            a, f1, t3, t5 = f("acc"), f("macro_f1"), f("top3"), f("top5")
            log("%-52s %.4f+-%.4f   %.4f+-%.4f   %.4f   %.4f" % (name, a[0], a[1], f1[0], f1[1], t3[0], t5[0]))
            res[name] = {"track": track, "acc": a, "macro_f1": f1, "top3": t3[0], "top5": t5[0]}
    log(f"\nPublished KI-DDI (as reported): acc={PAPER['acc']:.4f}  macro-F1={PAPER['macro_f1']:.4f}  "
        f"top3={PAPER['top3']:.3f}  top5={PAPER['top5']:.4f}")

    # ---- is the improvement real? (seed-averaged probabilities, paired bootstrap) ----
    log("\nSTATISTICAL CHECK (seed-averaged probabilities, paired bootstrap on the test set)")
    avg = lambda name: np.mean(systems[name][1], axis=0)
    correct = lambda P: (P.argmax(1) == te_y).astype(float)
    pairs = [("Hybrid: SapBERT + TF-IDF (equal avg)", "TF-IDF text LR (alone)"),
             ("Hybrid: SapBERT + TF-IDF (equal avg)", "SapBERT fine-tuned (alone)"),
             ("Hybrid: SapBERT + symptoms (equal avg)", "Annotated-symptom LR (alone)"),
             ("Hybrid: SapBERT + symptoms (equal avg)", "SapBERT fine-tuned (alone)")]
    for a_name, b_name in pairs:
        d, ci, pr = paired_diff(correct(avg(a_name)), correct(avg(b_name)))
        verdict = "SUPPORTED" if ci[0] > 0 else "NOT supported (interval includes 0)"
        log(f"  {a_name}  vs  {b_name}\n     accuracy diff = {d:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
            f"P(hybrid better)={pr:.2f}  -> {verdict}")
        res[f"diff::{a_name}::{b_name}"] = {"diff": d, "ci95": ci.tolist(), "p_better": pr}
    for name in ["Hybrid: SapBERT + TF-IDF (equal avg)", "Hybrid: SapBERT + symptoms (equal avg)"]:
        lo, hi = boot_ci(correct(avg(name)))
        log(f"  {name}: seed-avg accuracy 95% CI [{lo:.3f}, {hi:.3f}]  (published KI-DDI = {PAPER['acc']:.3f})")

    log("\nHOW TO READ THIS (be honest in the paper):")
    log("  * Test set is small: one accuracy point is only ~2.7 dialogues; intervals are wide.")
    log("  * Claim 'beats KI-DDI' only for the TEXT-ONLY track unless you also use annotated/predicted symptoms")
    log("    the same way KI-DDI does (check its paper). ORACLE rows use gold symptoms.")
    log("  * Put the simple baselines (TF-IDF, symptom LR) in your table: reviewers will ask for them.")
    log("  * If an improvement is 'NOT supported', say so and report it as comparable, not better.")
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    np.savez_compressed("step2_probs.npz", test_y=te_y, valid_y=va_y,
                        **{f"nn_test_{s}": p for s, p in zip(seeds, nn_t)},
                        **{f"nn_valid_{s}": p for s, p in zip(seeds, nn_v)},
                        tfidf_test=PT["tfidf"], sym_test=PT["sym"], tfidf_valid=PV["tfidf"], sym_valid=PV["sym"])
    log(f"\nsaved {args.out} and step2_probs.npz")


if __name__ == "__main__":
    main()