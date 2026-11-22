"""
Bits-per-byte (and bits-per-character) on the Kannada eval set for several tokenizer
variants of OLMo-2-7B, before any continued pretraining.

Variants:
  - base:          OLMo's tokenizer with the regex fix + NFC, no new tokens
  - bpe:           base + the first N learned merges (unconstrained or syllable-constrained file)
  - unigram_byte:  hybrid - byte-level Unigram for Kannada pre-tokens, OLMo BPE for the rest
  - unigram_char:  hybrid - character-level Unigram for Kannada runs, OLMo BPE for the rest

New embedding rows (input embeddings AND lm_head, which OLMo 2 does not tie) are initialized
as the mean of the rows of each new token's sub-tokens under OLMo's ORIGINAL BPE.
The model is loaded once; the original rows are restored before every variant, so
variants never inherit each other's rows.

Run:  python -u bpb_compare.py
Results are written to OUT_PATH after each variant; rerunning skips variants already done.
"""

import copy
import json
import math
import os
import pickle
import re
import unicodedata

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ============================================================================ config (EDIT)
MODEL_DIR = "/workspace/language_adaptation/olmo2-7b-stage1"
KN_EVAL = "/workspace/language_adaptation/eval/kn_eval_texts.jsonl"
OUT_PATH = "/workspace/language_adaptation/eval/bpb_tokenizer_variants.json"

tok_path = "/workspace/language_adaptation/trained_toks"

VARIANTS = [
    {"name": "base",              "kind": "bpe",          "merges": None, "n": 0},
    {"name": "bpe_4000",          "kind": "bpe",          "merges": f"{tok_path}/new_merges.pkl",      "n": 4000},
    {"name": "bpe_excl_4000",     "kind": "bpe",          "merges": f"{tok_path}/new_merges_excl.pkl", "n": 4000},
    {"name": "unigram_byte_4000", "kind": "unigram_byte", "path": f"{tok_path}/unigram_4000.json"},
    {"name": "unigram_char_4000", "kind": "unigram_char", "path": f"{tok_path}/unigram_4000_char.json"},
]

MAX_DOCS = None      # e.g. 300 for a quick check; None = the full eval set
MAX_LEN = 4096       # keep identical across all runs
SLICE_LEN = 1024     # memory only; does not affect results


# ============================================================================ byte-level helpers
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_ENC = bytes_to_unicode()
BYTE_DEC = {v: k for k, v in BYTE_ENC.items()}


def to_bytelevel(s):
    """Unicode string -> OLMo byte-level string (space becomes 'Ġ')."""
    return "".join(BYTE_ENC[b] for b in s.encode("utf-8"))


def bl_bytes(p):
    """Byte-level string -> raw bytes."""
    return bytes(BYTE_DEC[c] for c in p)


def is_kannada_bl(w):
    """Byte-level string contains a Kannada character (lead bytes E0 B2 / E0 B3)."""
    return "à²" in w or "à³" in w


def is_kn_char(ch):
    return "\u0c80" <= ch <= "\u0cff"


KN_RUN = re.compile(r"[\u0c80-\u0cff]+|[^\u0c80-\u0cff]+")


def split_leading(text):
    """'(ಮನೆ' -> ('(', 'ಮನೆ'); ' ಮನೆ' and 'ಮನೆ' are left whole."""
    if len(text) > 1 and text[0] != " " and not is_kn_char(text[0]):
        return text[0], text[1:]
    return "", text


def split_runs(text):
    """' ಕನ್ನಡabc' -> [(' ಕನ್ನಡ', True), ('abc', False)]; a leading space stays with the next run."""
    lead = ""
    if text.startswith(" "):
        lead, text = " ", text[1:]
    runs = []
    for i, r in enumerate(KN_RUN.findall(text)):
        runs.append(((lead if i == 0 else "") + r, is_kn_char(r[0])))
    return runs


# ============================================================================ base tokenizer
print("Loading OLMo's tokenizer and applying the regex fix")
_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
TJ = json.loads(_tok.backend_tokenizer.to_str())
_split = TJ["pre_tokenizer"]["pretokenizers"][0]
_old = _split["pattern"]["Regex"]
_new = _old.replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+")
assert _new != _old, "regex pattern not found"
_split["pattern"]["Regex"] = _new
TJ["normalizer"] = {"type": "NFC"}

BASE = Tokenizer.from_str(json.dumps(TJ))
ORIG_VOCAB = BASE.get_vocab()                 # includes the added/special tokens
BPE0 = BASE.model                             # OLMo's original BPE model (original merges)
FIRST_NEW_ID = max(ORIG_VOCAB.values()) + 1   # 100278 for OLMo 2

_check = "ನಮಸ್ಕಾರ ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ"
assert [_check[s:e] for _, (s, e) in BASE.pre_tokenizer.pre_tokenize_str(_check)] == \
       ["ನಮಸ್ಕಾರ", " ಕನ್ನಡ", " ಭಾಷೆಯಲ್ಲಿ"], "pre-tokenizer still splits Kannada words"


# ============================================================================ variant builders
# Each builder returns (encode, pieces, new_tokens):
#   encode(text) -> list of token IDs
#   pieces(text) -> list of byte-level piece strings (for the round-trip check)
#   new_tokens   -> {byte-level token string: new ID}

def build_bpe(merges_path, n):
    tj = copy.deepcopy(TJ)
    vocab, merge_list = tj["model"]["vocab"], tj["model"]["merges"]
    as_str = isinstance(merge_list[0], str)
    new_tokens, next_id = {}, FIRST_NEW_ID
    if merges_path and n:
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        for (a, b), _ in merges[:n]:
            if a + b not in vocab:
                vocab[a + b] = next_id
                new_tokens[a + b] = next_id
                next_id += 1
            merge_list.append(f"{a} {b}" if as_str else [a, b])
    tk = Tokenizer.from_str(json.dumps(tj))
    encode = lambda text: tk.encode(text, add_special_tokens=False).ids
    pieces = lambda text: tk.encode(text, add_special_tokens=False).tokens
    return encode, pieces, new_tokens


def hybrid_encoder(segment, new_tokens):
    def pieces(text):
        text = unicodedata.normalize("NFC", text)
        out = []
        for w, _ in BASE.pre_tokenizer.pre_tokenize_str(text):
            out.extend(segment(w))
        return out

    def encode(text):
        return [ORIG_VOCAB[p] if p in ORIG_VOCAB else new_tokens[p] for p in pieces(text)]

    return encode, pieces


def bpe0(s_bl):
    return [t.value for t in BPE0.tokenize(s_bl)]


def build_unigram_byte(path):
    u = Tokenizer.from_file(path)
    vocab = u.get_vocab()
    new_tokens, next_id = {}, FIRST_NEW_ID
    for p, _ in sorted(vocab.items(), key=lambda x: x[1]):
        if p not in ORIG_VOCAB:                 # every piece the model can emit needs an ID
            new_tokens[p] = next_id
            next_id += 1

    def segment(w):
        if is_kannada_bl(w):
            return [t.value for t in u.model.tokenize(w)]
        return bpe0(w)

    encode, pieces = hybrid_encoder(segment, new_tokens)
    return encode, pieces, new_tokens


def build_unigram_char(path):
    u = Tokenizer.from_file(path)
    unk_id = u.token_to_id("<unk>")
    vocab = u.get_vocab()
    new_tokens, next_id = {}, FIRST_NEW_ID
    for p, i in sorted(vocab.items(), key=lambda x: x[1]):
        if i == unk_id:
            continue
        bl = to_bytelevel(p)
        if bl not in ORIG_VOCAB and bl not in new_tokens:
            new_tokens[bl] = next_id
            next_id += 1

    def segment(w):
        if not is_kannada_bl(w):
            return bpe0(w)
        text = bl_bytes(w).decode("utf-8")      # pre-tokens always hold whole characters
        lead, rest = split_leading(text)
        out = bpe0(to_bytelevel(lead)) if lead else []
        for run, is_kn in split_runs(rest):
            if is_kn:
                toks = u.model.tokenize(run)
                if any(t.id == unk_id for t in toks):   # unseen character: byte-level BPE for this run
                    out += bpe0(to_bytelevel(run))
                else:
                    out += [to_bytelevel(t.value) for t in toks]
            else:
                out += bpe0(to_bytelevel(run))
        return out

    encode, pieces = hybrid_encoder(segment, new_tokens)
    return encode, pieces, new_tokens


def build_variant(v):
    if v["kind"] == "bpe":
        return build_bpe(v.get("merges"), v.get("n", 0))
    if v["kind"] == "unigram_byte":
        return build_unigram_byte(v["path"])
    if v["kind"] == "unigram_char":
        return build_unigram_char(v["path"])
    raise ValueError(f"unknown kind: {v['kind']}")


def check_roundtrip(pieces, texts, k=200):
    """Pieces must reproduce the NFC text byte for byte."""
    for t in texts[:k]:
        got = b"".join(bl_bytes(p) for p in pieces(t))
        if got != unicodedata.normalize("NFC", t).encode("utf-8"):
            raise AssertionError("round-trip failed: pieces do not reproduce the text")


# ============================================================================ model + initialization
def prepare_model(model, new_tokens, orig_in, orig_out):
    """Restore original rows, resize, and initialize new rows from sub-token means."""
    orig_rows = orig_in.shape[0]
    need = max([orig_rows] + [i + 1 for i in new_tokens.values()])
    model.resize_token_embeddings(need, pad_to_multiple_of=128, mean_resizing=False)

    w_in = model.get_input_embeddings().weight
    w_out = model.get_output_embeddings().weight
    dev, dt = w_in.device, w_in.dtype

    with torch.no_grad():
        w_in[:orig_rows] = orig_in.to(dev, dt)
        w_out[:orig_rows] = orig_out.to(dev, dt)

        # Padding rows beyond the original matrix: copy one of OLMo's own unused spare rows,
        # which it has learned never to predict, so they don't take probability mass.
        if w_in.shape[0] > orig_rows:
            w_in[orig_rows:] = orig_in[-1].to(dev, dt)
            w_out[orig_rows:] = orig_out[-1].to(dev, dt)

        for tok_str, i in new_tokens.items():
            sub = [t.id for t in BPE0.tokenize(tok_str)]
            w_in[i] = orig_in[sub].float().mean(0).to(dev, dt)
            w_out[i] = orig_out[sub].float().mean(0).to(dev, dt)


@torch.no_grad()
def measure(model, encode, texts):
    n_rows = model.get_output_embeddings().weight.shape[0]
    total_nll, n_bytes, n_chars, n_tok = 0.0, 0, 0, 0
    for t in tqdm(texts, desc="BPB", leave=False):
        ids = encode(t)
        if ids and max(ids) >= n_rows:
            raise AssertionError(f"token ID {max(ids)} outside the embedding matrix ({n_rows} rows)")
        ids = torch.tensor(ids, dtype=torch.long)
        n_tok += len(ids)
        for s in range(0, len(ids) - 1, MAX_LEN):
            chunk = ids[s : s + MAX_LEN + 1].unsqueeze(0).to(model.device)
            if chunk.shape[1] < 2:
                continue
            inp, tgt = chunk[:, :-1], chunk[:, 1:]
            hidden = model.model(inp).last_hidden_state
            for j in range(0, inp.shape[1], SLICE_LEN):
                logits = model.lm_head(hidden[:, j : j + SLICE_LEN]).float()
                total_nll += F.cross_entropy(logits[0], tgt[0, j : j + SLICE_LEN], reduction="sum").item()
        n_bytes += len(t.encode("utf-8"))
        n_chars += len(t)
    return {
        "bits_per_byte": total_nll / (math.log(2) * n_bytes),
        "bits_per_char": total_nll / (math.log(2) * n_chars),
        "tokens": n_tok,
        "chars_per_token": n_chars / n_tok,
    }


def save_results(results):
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, OUT_PATH)


# ============================================================================ main
def main():
    with open(KN_EVAL, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    if MAX_DOCS:
        texts = texts[:MAX_DOCS]
    print(f"{len(texts)} Kannada eval documents")

    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            results = json.load(f)

    print("Loading the model")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    orig_in = model.get_input_embeddings().weight.detach().cpu().clone()
    orig_out = model.get_output_embeddings().weight.detach().cpu().clone()
    print(f"Original embedding rows: {orig_in.shape[0]:,}; first new token ID: {FIRST_NEW_ID:,}")

    for v in VARIANTS:
        name = v["name"]
        if name in results:
            print(f"[{name}] already done: {results[name]['bits_per_byte']:.4f} BPB, skipping")
            continue

        print(f"\n[{name}] building tokenizer")
        encode, pieces, new_tokens = build_variant(v)
        check_roundtrip(pieces, texts)
        print(f"[{name}] {len(new_tokens):,} new tokens; round-trip OK")

        prepare_model(model, new_tokens, orig_in, orig_out)
        metrics = measure(model, encode, texts)
        metrics.update({
            "new_tokens": len(new_tokens),
            "n_docs": len(texts),
            "max_len": MAX_LEN,
            "init": "sub-token mean (input + output embeddings)",
            "variant": v,
        })
        results[name] = metrics
        save_results(results)
        print(f"[{name}] BPB {metrics['bits_per_byte']:.4f} | bits/char {metrics['bits_per_char']:.4f} | "
              f"chars/token {metrics['chars_per_token']:.3f}")

    print("\nSummary")
    for name, m in results.items():
        print(f"  {name:22} BPB {m['bits_per_byte']:.4f}   bits/char {m['bits_per_char']:.4f}   "
              f"new tokens {m['new_tokens']:,}")


if __name__ == "__main__":
    main()
