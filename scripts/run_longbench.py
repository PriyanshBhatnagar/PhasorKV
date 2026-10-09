"""LongBench (v1, English) with fake-quantized latent KV caches.

Follows LongBench's pred.py / eval.py: the same prompts, middle truncation to
--max-length tokens, chat formatting except for trec / triviaqa / samsum / lsht /
lcc / repobench-p, greedy decoding with the per-task generation length (samsum also
stops at a newline), and the official metrics (metrics.py, fetched from the
LongBench repository). A latent method replaces k_proj / v_proj in every layer,
so each token's key and value are compressed once, when written, during prefill
and decoding alike, exactly as a latent cache would hold them.

Methods: bf16, or <basis variant>:<method>, e.g. diag-wo-alpha:starq.sink4@0.85
(results/<tag>/latent_<variant> from ablation_prep.py; .gs needs layer_sens.json).

  python scripts/run_longbench.py --model lmsys/longchat-7b-v1.5-32k --tag longchat7b \\
      --methods bf16,diag-wo-alpha:starq.sink4@0.85 --limit 50
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request
import zipfile

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phasorkv.kvmethods import LatentPatch, compression

TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
         "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
NO_CHAT = ("trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p")
FIRST_LINE = ("trec", "triviaqa", "samsum", "lsht")
METRIC = {"narrativeqa": "qa_f1_score", "qasper": "qa_f1_score", "multifieldqa_en": "qa_f1_score",
          "hotpotqa": "qa_f1_score", "2wikimqa": "qa_f1_score", "musique": "qa_f1_score",
          "gov_report": "rouge_score", "qmsum": "rouge_score", "multi_news": "rouge_score",
          "trec": "classification_score", "triviaqa": "qa_f1_score", "samsum": "rouge_score",
          "passage_retrieval_en": "retrieval_score", "passage_count": "count_score",
          "lcc": "code_sim_score", "repobench-p": "code_sim_score"}
GROUP = {"single-doc QA": TASKS[0:3], "multi-doc QA": TASKS[3:6], "summarization": TASKS[6:9],
         "few-shot": TASKS[9:12], "synthetic": TASKS[12:14], "code": TASKS[14:16]}
LB = "https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/"
VICUNA = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")


def longbench_files(cache):
    """Official prompts, generation lengths and metrics, plus the English data."""
    os.makedirs(cache, exist_ok=True)
    for f in ("config/dataset2prompt.json", "config/dataset2maxlen.json", "metrics.py"):
        dst = os.path.join(cache, os.path.basename(f))
        if not os.path.exists(dst):
            urllib.request.urlretrieve(LB + f, dst)
    spec = importlib.util.spec_from_file_location("lb_metrics", os.path.join(cache, "metrics.py"))
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    if not os.path.exists(os.path.join(cache, "data")):
        zipfile.ZipFile(hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")).extractall(cache)
    prompts = json.load(open(os.path.join(cache, "dataset2prompt.json")))
    maxlen = json.load(open(os.path.join(cache, "dataset2maxlen.json")))
    return prompts, maxlen, metrics


def build_chat(tok, prompt, model_name):
    if "longchat" in model_name or "vicuna" in model_name:
        return f"{VICUNA} USER: {prompt} ASSISTANT:", True
    if tok.chat_template:
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True), False      # template carries BOS
    return prompt, True


def score(task, preds, metrics):
    fn = getattr(metrics, METRIC[task])
    tot = 0.0
    for p in preds:
        pred = p["pred"].lstrip("\n").split("\n")[0] if task in FIRST_LINE else p["pred"]
        tot += max(fn(pred, gt, all_classes=p["all_classes"]) for gt in p["answers"])
    return round(100 * tot / len(preds), 2)


def install(model, tag_root, method, n_kv, d, n_q, device):
    """Patch every layer for a latent method; returns the patch and an undo function."""
    variant, m = method.split(":")
    kind, frac = m.split("@")
    patch = LatentPatch(kind, float(frac))
    lat = os.path.join(tag_root, f"latent_{variant}")
    L = len(model.model.layers)
    if patch.global_sens:
        tables = {i: {k: v.clone() for k, v in torch.load(os.path.join(lat, f"L{i}.pt"), mmap=True).items()
                      if k in ("DK", "DV")} for i in range(L)}
        sens = {int(l): v for l, v in json.load(open(os.path.join(tag_root, "layer_sens.json")))["layers"].items()}
        patch.plan_global(tables, sens, n_kv, d)
    undos = [patch.patch(i, layer, torch.load(os.path.join(lat, f"L{i}.pt")), n_kv, d, n_q, device)
             for i, layer in enumerate(model.model.layers)]
    return patch, lambda: [u() for u in undos]


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--limit", type=int, default=0, help="first N samples per task (0: all)")
    p.add_argument("--max-length", type=int, default=31500, help="prompt tokens after middle truncation")
    p.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "..", "results", "longbench_data"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    prompts, maxlen, metrics = longbench_files(args.cache)
    root = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    out_root = os.path.join(root, "longbench" + (f"_n{args.limit}" if args.limit else ""))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device,
                                                 attn_implementation="sdpa").eval()
    cfg = model.config
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    nl_id = tok.encode("\n", add_special_tokens=False)[-1]
    scores_path = os.path.join(out_root, "scores.json")

    for method in [m for m in args.methods.split(",") if m]:
        patch, undo = (None, lambda: None) if method == "bf16" else \
            install(model, root, method, n_kv, d, n_q, args.device)
        mdir = os.path.join(out_root, method.replace(":", "__"))
        os.makedirs(mdir, exist_ok=True)
        res = {}
        for task in args.tasks.split(","):
            data = [json.loads(l) for l in open(os.path.join(args.cache, "data", f"{task}.jsonl"))]
            data = data[:args.limit] if args.limit else data
            path = os.path.join(mdir, f"{task}.jsonl")
            done = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
            t0 = time.time()
            for obj in data[len(done):]:
                prompt = prompts[task].format(**obj)
                ids = tok(prompt, truncation=False).input_ids
                if len(ids) > args.max_length:
                    half = args.max_length // 2
                    prompt = tok.decode(ids[:half], skip_special_tokens=True) + \
                        tok.decode(ids[-half:], skip_special_tokens=True)
                add_special = True
                if task not in NO_CHAT:
                    prompt, add_special = build_chat(tok, prompt, args.model)
                enc = tok(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special).to(args.device)
                ctx = enc.input_ids.shape[-1]
                kw = dict(max_new_tokens=maxlen[task], num_beams=1, do_sample=False)
                if task == "samsum":
                    eos = model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    kw.update(min_length=ctx + 1, eos_token_id=eos + [nl_id])
                out = model.generate(**enc, **kw)[0]
                pred = tok.decode(out[ctx:], skip_special_tokens=True)
                rec = dict(pred=pred, answers=obj["answers"], all_classes=obj["all_classes"],
                           length=obj["length"], ctx_tokens=ctx)
                done.append(rec)
                with open(path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            res[task] = score(task, done, metrics)
            print(f"[{method}] {task:22s} {res[task]:6.2f}  (n={len(done)}, {time.time() - t0:.0f}s)", flush=True)
        res["avg"] = round(sum(res[t] for t in TASKS if t in res) / len([t for t in TASKS if t in res]), 2)
        for g, ts in GROUP.items():
            if all(t in res for t in ts):
                res[g] = round(sum(res[t] for t in ts) / len(ts), 2)
        if patch is not None:
            res["compression"] = compression(patch, cfg.num_hidden_layers, n_kv, d)
        allres = json.load(open(scores_path)) if os.path.exists(scores_path) else {}
        allres[method] = res
        json.dump(allres, open(scores_path, "w"), indent=1)
        print(f"[{method}] avg {res['avg']:.2f}  " + "  ".join(f"{g}={res[g]:.2f}" for g in GROUP if g in res), flush=True)
        undo()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
