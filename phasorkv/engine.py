"""Layer-at-a-time execution of a HF Llama-family model.

The model stays on the CPU and one decoder layer at a time is moved to the GPU,
with every window's hidden states carried from layer to layer. Peak GPU memory
is one layer plus a batch of activations, so an 8B model runs next to other
tenants on a 24 GB card, and each layer's k_proj can be swapped before its
turn without touching the rest.
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_name: str, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="cpu", attn_implementation="sdpa")
    model.eval()
    cfg = model.config
    assert not getattr(cfg, "attention_bias", False), "k_proj bias is not handled"
    return model, tok


def head_dims(model):
    cfg = model.config
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_attention_heads, cfg.num_key_value_heads, d


# ---------------------------------------------------------------------------
# data (same sources and slices as STAR-KV's eval.py)
# ---------------------------------------------------------------------------

def token_stream(name: str, split: str, tok) -> torch.Tensor:
    if name == "wikitext2":
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        return tok("\n\n".join(data["text"]), return_tensors="pt").input_ids[0]
    if name == "c4":
        assert split == "validation"
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        return tok(" ".join(data[:1100]["text"]), return_tensors="pt").input_ids[0]
    raise ValueError(name)


def c4_heldout(tok, n: int, seqlen: int) -> torch.Tensor:
    """C4 validation documents after the 1100 used for PPL, so they never overlap."""
    data = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        split="validation",
    )
    ids = tok(" ".join(data[2000:4000]["text"]), return_tensors="pt").input_ids[0]
    return consecutive_windows(ids, n, seqlen)


def consecutive_windows(ids: torch.Tensor, n: int, seqlen: int) -> torch.Tensor:
    n = min(n, ids.numel() // seqlen)
    return ids[: n * seqlen].reshape(n, seqlen)


def random_windows(ids: torch.Tensor, n: int, seqlen: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seqlen, (n,), generator=g)
    return torch.stack([ids[s : s + seqlen] for s in starts])


# ---------------------------------------------------------------------------
# layerwise execution
# ---------------------------------------------------------------------------

class Layerwise:
    def __init__(self, model, device="cuda:0", batch: int = 4):
        self.model, self.device, self.batch = model, torch.device(device), batch
        self.layers = model.model.layers
        self.rotary = model.model.rotary_emb.to(self.device)
        self._pe = {}

    def position_embeddings(self, T: int, dtype):
        key = (T, dtype)
        if key not in self._pe:
            pos = torch.arange(T, device=self.device)[None]
            dummy = torch.empty(1, T, 1, device=self.device, dtype=dtype)
            self._pe[key] = self.rotary(dummy, pos)
        return self._pe[key]

    @torch.no_grad()
    def embed(self, windows: torch.Tensor) -> torch.Tensor:
        """Token windows [N, T] -> hidden states [N, T, D] on the CPU."""
        return self.model.model.embed_tokens(windows)

    def batches(self, h: torch.Tensor):
        for i in range(0, h.shape[0], self.batch):
            yield slice(i, i + self.batch), h[i : i + self.batch].to(self.device, non_blocking=True)

    @torch.no_grad()
    def run_layer(self, layer, h: torch.Tensor, nope=None) -> torch.Tensor:
        """Push every window through one decoder layer already on the GPU; returns CPU states.

        nope: bool [head_dim/2], frequencies this layer scores without RoPE
        (bandN's slow block), applied to q and k alike as its cache does."""
        out = torch.empty_like(h)
        for sl, hb in self.batches(h):
            cos, sin = self.position_embeddings(hb.shape[1], hb.dtype)
            if nope is not None and nope.any():
                m = torch.cat([nope, nope]).to(self.device)
                cos = torch.where(m, cos[:, :1], cos)      # position 0: the attention scale
                sin = torch.where(m, torch.zeros_like(sin), sin)
            out[sl] = layer(hb, attention_mask=None, position_embeddings=(cos, sin), use_cache=False).cpu()
        return out

    @torch.no_grad()
    def nll(self, h: torch.Tensor, windows: torch.Tensor, chunk: int = 512) -> tuple[float, int]:
        """Summed next-token NLL and token count from final hidden states.

        One window and `chunk` tokens at a time: a full window of Llama-3 logits
        (128k vocab) in fp32 is 1 GB."""
        norm = self.model.model.norm.to(self.device)
        head = self.model.lm_head.to(self.device)
        total, count = 0.0, 0
        try:
            for i in range(h.shape[0]):
                hw = norm(h[i, :-1].to(self.device))
                labels = windows[i, 1:].to(self.device)
                for c in range(0, hw.shape[0], chunk):
                    logits = head(hw[c : c + chunk]).float()
                    total += torch.nn.functional.cross_entropy(
                        logits, labels[c : c + chunk], reduction="sum").item()
                count += labels.numel()
        finally:
            norm.cpu(), head.cpu()
        return total, count
