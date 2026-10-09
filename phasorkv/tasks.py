"""Zero-shot multiple-choice tasks with lm-eval-harness's prompts and scoring.

Each choice is scored by the summed log-prob of its continuation given the
context; acc takes the argmax, acc_norm divides by the continuation's length in
characters first. WinoGrande uses lm-eval's partial scoring: the option fills
the blank in the context and the shared rest of the sentence is scored.
"""
import re

import torch
from datasets import load_dataset

TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "openbookqa")
REPORT = {"piqa": "acc", "arc_easy": "acc", "arc_challenge": "acc_norm",
          "hellaswag": "acc_norm", "winogrande": "acc", "openbookqa": "acc_norm"}


def _hs_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def load_task(name, limit=None):
    """-> list of (contexts: list[str], continuations: list[str], gold: int); one context per
    choice for winogrande, the same context repeated otherwise."""
    out = []
    if name == "piqa":
        for ex in load_dataset("baber/piqa", split="validation"):
            ctx = f"Question: {ex['goal']}\nAnswer:"
            out.append(([ctx, ctx], [" " + ex["sol1"], " " + ex["sol2"]], int(ex["label"])))
    elif name in ("arc_easy", "arc_challenge"):
        sub = "ARC-Easy" if name == "arc_easy" else "ARC-Challenge"
        for ex in load_dataset("allenai/ai2_arc", sub, split="test"):
            ctx = f"Question: {ex['question']}\nAnswer:"
            ch = ex["choices"]
            out.append(([ctx] * len(ch["text"]), [" " + t for t in ch["text"]],
                        ch["label"].index(ex["answerKey"])))
    elif name == "hellaswag":
        for ex in load_dataset("Rowan/hellaswag", split="validation"):
            ctx = _hs_pre(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
            out.append(([ctx] * 4, [" " + _hs_pre(e) for e in ex["endings"]], int(ex["label"])))
    elif name == "winogrande":
        for ex in load_dataset("allenai/winogrande", "winogrande_xl", split="validation"):
            s = ex["sentence"]
            i = s.index("_")
            cont = " " + s[i + 1:].strip()
            out.append(([s[:i] + ex["option1"], s[:i] + ex["option2"]], [cont, cont],
                        int(ex["answer"]) - 1))
    elif name == "openbookqa":
        for ex in load_dataset("allenai/openbookqa", "main", split="test"):
            ch = ex["choices"]
            out.append(([ex["question_stem"]] * len(ch["text"]), [" " + t for t in ch["text"]],
                        ch["label"].index(ex["answerKey"])))
    else:
        raise ValueError(name)
    return out[:limit] if limit else out


def build_requests(tok, tasks, limit):
    """Tokenize every (context, continuation) pair; returns seqs, spans, and per-question index."""
    seqs, spans, meta = [], [], []
    for name in tasks:
        for qi, (ctxs, conts, gold) in enumerate(load_task(name, limit)):
            first = len(seqs)
            for ctx, cont in zip(ctxs, conts):
                c_ids = tok(ctx).input_ids
                w_ids = tok(ctx + cont).input_ids
                n_c = len(c_ids)
                # sentencepiece can merge across the boundary; keep at least one target token
                while n_c > 1 and w_ids[:n_c] != c_ids[:n_c]:
                    n_c -= 1
                n_c = min(n_c, len(w_ids) - 1)
                seqs.append(torch.tensor(w_ids))
                spans.append((n_c, len(w_ids)))
            meta.append(dict(task=name, first=first, n=len(ctxs), gold=gold,
                             lens=[len(c) for c in conts]))
    return seqs, spans, meta


def accuracy(meta, logprob):
    """Per task: acc and acc_norm."""
    res = {}
    for m in meta:
        lp = logprob[m["first"]:m["first"] + m["n"]]
        acc = int(int(lp.argmax()) == m["gold"])
        norm = lp / lp.new_tensor(m["lens"])
        accn = int(int(norm.argmax()) == m["gold"])
        r = res.setdefault(m["task"], [0, 0, 0])
        r[0] += acc
        r[1] += accn
        r[2] += 1
    return {t: dict(acc=a / n, acc_norm=b / n, n=n) for t, (a, b, n) in res.items()}
