#!/usr/bin/env python
"""Build speaker-disjoint splits for the 'destiny' wake-word spotter.

All audio lives in the TRAIN split (the 557K on-disk wavs are train rows).
We therefore carve held-out sets out of train.csv by SPEAKER hash so that
training and every held-out set are speaker-disjoint.

Outputs (JSON) in destiny_wakeword/splits/:
  pos_train.json   destiny positives for training
  pos_test.json    held-out destiny positives (disjoint speakers)
  neg_train.json   OOS negatives for training, capped per word
  oos_test.json    held-out OOS negatives (disjoint speakers), by category
"""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "splits")
os.makedirs(OUT, exist_ok=True)

CMD_WORDS = ["on","off","go","stop","yes","no","up","down","left","right",
             "zero","one","two","three","four","five","six","seven","eight","nine",
             "cat","dog","bird","bed","house","tree","wow"]
NEAR_MISS = ["desert","distance","destination","destinies","distant","destined",
             "despite","design","desire","faint","faintly","district","despair",
             "descent","deserved","dessert","destroys","destroy"]
NEAR_MISS = [w for w in NEAR_MISS if w != "destiny"]

def stem_of(r):
    return r["WORD"] + "_" + os.path.basename(r["LINK"]).replace(".opus", ".wav")[:-4]

def load(manifest):
    df = pd.read_csv(manifest).dropna(subset=["LINK","WORD","SPEAKER"])
    df = df[df["VALID"] == True].copy()
    df["stem"] = df.apply(stem_of, axis=1)
    return df

_WAVSET = None
def wavset():
    global _WAVSET
    if _WAVSET is None:
        import glob
        _WAVSET = set(os.path.basename(w)[:-4] for w in glob.glob(os.path.join(DATA,"wavs","*.wav")))
    return _WAVSET

def on_disk(stem):
    return stem in wavset()

def pick(df, word_set, cap, rng):
    rows = []
    for w in word_set:
        sub = df[(df["WORD"]==w) & (df["stem"].map(on_disk))]
        if len(sub)==0: continue
        rows.append(sub.sample(n=min(cap,len(sub)), random_state=rng))
    return pd.concat(rows) if rows else pd.DataFrame(columns=df.columns)

def to_records(df, category):
    return [{"stem":r.stem,"word":r.WORD,"speaker":r.SPEAKER,"category":category}
            for r in df.itertuples()]

def main():
    rng = 12345
    import random as _rnd
    _rnd.seed(rng)
    tr = load(os.path.join(DATA,"splits","train.csv"))
    print(f"train rows: {len(tr)}, on-disk destiny: {int(((tr['WORD']=='destiny')&(tr['stem'].map(on_disk))).sum())}")

    # ---- destiny: split speakers into train / held-out (70/30) ----
    dest = tr[(tr["WORD"]=="destiny") & (tr["stem"].map(on_disk))].copy()
    spk = sorted(dest["SPEAKER"].unique())
    _rnd.shuffle(spk)
    n_hold = max(1, int(len(spk)*0.30))
    hold_spk = set(spk[:n_hold]); train_spk = set(spk[n_hold:])
    pos_te = dest[dest["SPEAKER"].isin(hold_spk)]
    pos_tr = dest[dest["SPEAKER"].isin(train_spk)].sample(n=min(1500,len(dest[dest['SPEAKER'].isin(train_spk)])), random_state=rng)

    # ---- OOS negatives (non-destiny), capped per word, speaker-disjoint ----
    CAP = 40
    non_dest = tr[(tr["WORD"]!="destiny") & (tr["stem"].map(on_disk))]
    neg_tr = pick(non_dest[non_dest["SPEAKER"].isin(train_spk)], CMD_WORDS+NEAR_MISS, CAP, rng)
    neg_te = pick(non_dest[non_dest["SPEAKER"].isin(hold_spk)], CMD_WORDS+NEAR_MISS, CAP, rng)

    def categorize(w):
        if w in CMD_WORDS: return "command"
        if w in NEAR_MISS: return "near_miss"
        return "random"
    neg_tr["category"] = neg_tr["WORD"].map(categorize)
    neg_te["category"] = neg_te["WORD"].map(categorize)

    # random-other-word negatives
    used = set(CMD_WORDS)|set(NEAR_MISS)|{"destiny"}
    rp_tr = non_dest[(~non_dest["WORD"].isin(used)) & (non_dest["SPEAKER"].isin(train_spk))]
    rp_te = non_dest[(~non_dest["WORD"].isin(used)) & (non_dest["SPEAKER"].isin(hold_spk))]
    rw_tr = list(rp_tr["WORD"].value_counts().head(60).index)
    rw_te = list(rp_te["WORD"].value_counts().head(60).index)
    rand_tr = pick(rp_tr, rw_tr, CAP, rng); rand_tr["category"]="random"
    rand_te = pick(rp_te, rw_te, CAP, rng); rand_te["category"]="random"
    neg_tr = pd.concat([neg_tr, rand_tr]); neg_te = pd.concat([neg_te, rand_te])

    # final speaker-disjointness guard
    train_speakers = set(pos_tr["SPEAKER"]) | set(neg_tr["SPEAKER"])
    pos_te_c = pos_te[~pos_te["SPEAKER"].isin(train_speakers)]
    neg_te_c = neg_te[~neg_te["SPEAKER"].isin(train_speakers)]

    def dump(name, recs):
        json.dump(recs, open(os.path.join(OUT,name),"w"))
        print(f"{name}: {len(recs)}")
    dump("pos_train.json", to_records(pos_tr,"destiny"))
    dump("pos_test.json", to_records(pos_te_c,"destiny"))
    dump("neg_train.json", to_records(neg_tr,""))
    dump("oos_test.json", to_records(neg_te_c,""))

    print("\n=== SUMMARY ===")
    print(f"pos_train: {len(pos_tr)} (spk {pos_tr['SPEAKER'].nunique()})")
    print(f"pos_test:  {len(pos_te_c)} (spk {pos_te_c['SPEAKER'].nunique()})")
    print(f"neg_train: {len(neg_tr)}  {neg_tr['category'].value_counts().to_dict()}")
    print(f"oos_test:  {len(neg_te_c)}  {neg_te_c['category'].value_counts().to_dict()}")
    print(f"oos_test words: {neg_te_c['WORD'].nunique()}")
    ov = len(set(pos_te_c['SPEAKER']) & train_speakers)
    print(f"speaker overlap (should be 0): {ov}")

if __name__=="__main__":
    main()
