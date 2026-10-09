"""Layerwise scoring of many sequences under many KV methods at once.

Each sequence carries a target span; the runner returns the summed log-prob of
the target tokens and their count, which covers perplexity windows (target =
every token after the first) and multiple-choice tasks (target = the answer
continuation). Sequences are right-padded inside length-sorted batches; with
causal attention a real token never sees a pad, and every method is told each
batch's lengths so token-axis quantizers can ignore the pads.

Post-RoPE methods hook in through a registered attention function that
quantizes key/value states right before SDPA, which is where a real cache
would store them.
"""
import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def _phasor_attention(module, query, key, value, attention_mask, **kwargs):
    fq = getattr(module, "kv_fq", None)
    if fq is not None:
        kq, vq = fq(key, value, getattr(module, "kv_lengths", None))
        key, value = kq.to(key.dtype), vq.to(value.dtype)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("phasor_fq", _phasor_attention)


class Batch:
    def __init__(self, idx, ids, lengths, spans):
        self.idx, self.ids, self.lengths, self.spans = idx, ids, lengths, spans


def make_batches(seqs, spans, max_tokens=16384, pad_id=0):
    """seqs: list of 1-D LongTensors; spans: (start, end) target token positions per sequence."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    batches, cur = [], []
    for i in order:
        L = len(seqs[cur[0]]) if cur else len(seqs[i])
        if cur and (len(cur) + 1) * L > max_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    out = []
    for b in batches:
        L = len(seqs[b[0]])
        ids = torch.full((len(b), L), pad_id, dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, :len(seqs[i])] = seqs[i]
        lengths = torch.tensor([len(seqs[i]) for i in b])
        out.append(Batch(b, ids, lengths, [spans[i] for i in b]))
    return out


class Runner:
    def __init__(self, model, device="cuda:0", prep_loader=None):
        self.model, self.device = model, torch.device(device)
        cfg = model.config
        self.n_q, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.d = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_q
        self.layers = model.model.layers
        self.rotary = model.model.rotary_emb.to(self.device)
        self.prep_loader = prep_loader
        model.config._attn_implementation = "phasor_fq"

    @torch.no_grad()
    def score(self, batches, patches, skip_layers=()):
        """Returns {patch.name: (logprob_sum [N], n_target [N])} for N sequences."""
        hs = self._forward(batches, patches, skip_layers)
        return {p.name: self._targets(batches, hs[p.name]) for p in patches}

    @torch.no_grad()
    def score_kl(self, batches, patches, ref):
        """Mean per-token KL(ref || patch) of the next-token distributions over the target
        spans, for every patch other than `ref` (a patch name in `patches`)."""
        hs = self._forward(batches, patches)
        return {p.name: self._kl(batches, hs[p.name], hs[ref]) for p in patches if p.name != ref}

    @torch.no_grad()
    def _forward(self, batches, patches, skip_layers=()):
        emb = self.model.model.embed_tokens
        hs = {p.name: [emb(b.ids) for b in batches] for p in patches}
        for i, layer in enumerate(self.layers):
            layer.to(self.device)
            prep = self.prep_loader(i) if (self.prep_loader and i not in skip_layers) else None
            for p in patches:
                undo = (lambda: None) if i in skip_layers else \
                    p.patch(i, layer, prep, self.n_kv, self.d, self.n_q, self.device)
                attn = layer.self_attn
                for bi, b in enumerate(batches):
                    full = bool((b.lengths == b.ids.shape[1]).all())
                    lengths = None if full else b.lengths.to(self.device)
                    attn.kv_lengths = lengths
                    for m in (attn.k_proj, attn.v_proj):
                        if hasattr(m, "lengths"):
                            m.lengths = lengths
                    h = hs[p.name][bi].to(self.device)
                    L = h.shape[1]
                    pos = torch.arange(L, device=self.device)[None]
                    pe = self.rotary(h, pos)
                    hs[p.name][bi] = layer(h, attention_mask=None, position_embeddings=pe,
                                           use_cache=False).cpu()
                undo()
                attn.kv_lengths = None
            del prep
            layer.cpu()
            torch.cuda.empty_cache()
        return hs

    @torch.no_grad()
    def _kl(self, batches, hs, hs_ref, chunk=512):
        norm = self.model.model.norm.to(self.device)
        head = self.model.lm_head.to(self.device)
        tot, n = 0.0, 0
        try:
            for b, h, hr in zip(batches, hs, hs_ref):
                for r in range(len(b.idx)):
                    s, e = b.spans[r]
                    for c in range(s - 1, e - 1, chunk):
                        sl = slice(c, min(c + chunk, e - 1))
                        lp = torch.log_softmax(head(norm(h[r, sl].to(self.device))).float(), -1)
                        lr = torch.log_softmax(head(norm(hr[r, sl].to(self.device))).float(), -1)
                        tot += (lr.exp() * (lr - lp)).sum().item()
                        n += lp.shape[0]
        finally:
            norm.cpu(), head.cpu()
        return tot / n

    @torch.no_grad()
    def _targets(self, batches, hs, chunk=1024):
        norm = self.model.model.norm.to(self.device)
        head = self.model.lm_head.to(self.device)
        N = sum(len(b.idx) for b in batches)
        lp, cnt = torch.zeros(N, dtype=torch.float64), torch.zeros(N, dtype=torch.long)
        try:
            for b, h in zip(batches, hs):
                for r, i in enumerate(b.idx):
                    s, e = b.spans[r]
                    x = norm(h[r, s - 1:e - 1].to(self.device))
                    y = b.ids[r, s:e].to(self.device)
                    tot = 0.0
                    for c in range(0, x.shape[0], chunk):
                        logits = head(x[c:c + chunk]).float()
                        tot += -torch.nn.functional.cross_entropy(logits, y[c:c + chunk], reduction="sum").item()
                    lp[i], cnt[i] = tot, e - s
        finally:
            norm.cpu(), head.cpu()
        return lp, cnt
