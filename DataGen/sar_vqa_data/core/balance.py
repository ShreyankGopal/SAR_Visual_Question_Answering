"""balance.py: answer balancing per (source, category), rules chosen in the config.

config:
  balance:
    regional_vqa:    {rule: yes_no}           # equal Yes / No; other answer keys kept
    sar_comparative: {rule: ab}               # correct answer first / second 50:50
    orientation:     {rule: cap_share, share: 0.35}
    object_counting: {rule: zero_cap, share: 0.2}
Categories without a rule are left as generated.
"""
import numpy as np
import pandas as pd


def seed(rng):
    return int(rng.integers(1e9))


def cap_share(g, share, rng):
    """Downsample so no answer_key exceeds `share` of the category."""
    c = g.answer_key.value_counts()
    cap = int(c.max())
    while cap > 1 and cap > share * np.minimum(c, cap).sum():
        cap -= 1
    return pd.concat([x.sample(min(len(x), cap), random_state=seed(rng)) for _, x in g.groupby("answer_key")])


def equalise(g, keys, rng):
    """Downsample rows whose answer_key is in `keys` to equal counts; other rows kept."""
    sel, rest = g[g.answer_key.isin(keys)], g[~g.answer_key.isin(keys)]
    n = sel.answer_key.value_counts().reindex(keys, fill_value=0).min()
    keep = [sel[sel.answer_key == k].sample(n, random_state=seed(rng)) for k in keys]
    return pd.concat(keep + [rest])


def zero_cap(g, share, rng):
    z, nz = g[g.answer_key == "zero"], g[g.answer_key != "zero"]
    max_z = int(share / (1 - share) * len(nz))
    return pd.concat([nz, z.sample(min(len(z), max_z), random_state=seed(rng))])


def balance(df, cfg, rng):
    rules = cfg.get("balance", {})
    out = []
    for (src, cat), g in df.groupby(["source", "category"]):
        r = rules.get(cat)
        if r is None:
            pass
        elif r["rule"] == "yes_no":
            g = equalise(g, ["Yes", "No"], rng)
        elif r["rule"] == "ab":
            g = equalise(g, ["first", "second"], rng)
        elif r["rule"] == "cap_share":
            g = cap_share(g, r["share"], rng)
        elif r["rule"] == "zero_cap":
            g = zero_cap(g, r["share"], rng)
        else:
            raise ValueError(f"unknown balance rule {r['rule']} for {cat}")
        out.append(g)
    return pd.concat(out)
