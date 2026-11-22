# %pip install -q transformers datasets lm-eval accelerate

import os
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/root/.cache/huggingface/datasets"


from datasets import load_dataset
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer


import json
from tqdm import tqdm
import math, torch
import importlib.metadata as md


model_dir = "/workspace/language_adaptation/olmo2-7b-stage1-step928646-tokens3896B"


# #### View English benchmarks


with open("/workspace/language_adaptation/eval/baseline_en_benchmarks.json") as f:
    results = json.load(f)


headline = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "mmlu": "acc"}

for task, m in headline.items():
    print(f"{task:15s} {m:9s} {results[task][f'{m},none']:.4f}")


print(f"{'task':40s} {'metric':12s} {'value':>8s} {'stderr':>8s}")

for task, metrics in sorted(results.items()):
    for key, val in metrics.items():
        if key.endswith("_stderr") or "," not in key or not isinstance(val, (int, float)):
            continue
        name = key.split(",")[0]          # e.g. "acc_norm,none" -> "acc_norm"
        err = metrics.get(f"{name}_stderr,none")
        err_s = f"{err:.4f}" if isinstance(err, (int, float)) else ""
        print(f"{task:40s} {name:12s} {val:8.4f} {err_s:>8s}")



# #### Loading English dataset


from itertools import islice


# Loading the fineweb-edu dataset from HF (streaming)
en_stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                         split="train", streaming=True)
en_stream = en_stream.shuffle(seed=42, buffer_size=10_000)



#en_texts = [x["text"][:8000] for x in islice(en_stream, 2500) if len(x["text"]) > 200][:2000]
en_texts = [x["text"][:8000] for x in tqdm(islice(en_stream, 2500)) if len(x["text"]) > 200][:2000] 
print(len(en_texts))


# Writing the English dataset to disk
os.makedirs("/workspace/language_adaptation/eval", exist_ok=True)
with open("/workspace/language_adaptation/eval/en_eval_texts.jsonl", "w", encoding="utf-8") as f:
    for t in en_texts:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


# Loading from disk


# with open("/workspace/language_adaptation/eval/en_eval_texts.jsonl", encoding="utf-8") as f:
#     en_texts = [json.loads(line)["text"] for line in f]



# #### Loading tokenizer, dataset, and calculating tokenizer_stats, bits_per_byte

# Loading the tokenizer
tok = AutoTokenizer.from_pretrained(model_dir)


# Disable caching as it leads to permission errors on /workspace directories
datasets.disable_caching()



def tokenizer_stats(tok, texts):
    n_tok = n_words = n_chars = n_frag = 0
    for t in tqdm(texts):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tok += len(ids)
        n_words += len(t.split())
        n_chars += len(t)
        n_frag += sum("\ufffd" in tok.decode([i]) for i in ids)
    return {
        "fertility (tokens/word)": n_tok / n_words,
        "chars/token": n_chars / n_tok,
        "fragment rate": n_frag / n_tok,
    }


tok_stats = tokenizer_stats(tok, en_texts)

# fertility (tokens/word): 1.2804896737381426
# chars/token: 4.7985698471112705
# fragment rate: 0.0013623353179945945


EVAL_DIR = "/workspace/language_adaptation/eval"

def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: never leaves a half-written file


eval_config = {
    "model": "allenai/OLMo-2-1124-7B", "revision": "stage1-step928646-tokens3896B", "dtype": "bf16",
    "tokenizer": "original", 
    "en_eval": "/workspace/language_adaptation/eval/en_eval_texts.jsonl", "bpb_max_len": 4096,
    "lm_eval": {"num_fewshot": 0, "limit": None},
    "versions": {p: md.version(p) for p in ["torch", "transformers", "datasets", "lm_eval"]},
}


lang, texts, tag = "en", en_texts, "baseline"   # tag = which tokenizer/model variant


save_json({"config": eval_config, "lang": lang, "n_docs": len(texts),
           "tokenizer_stats": tok_stats}, f"{EVAL_DIR}/{tag}_{lang}_tokenizer.json")
print("Tokenizer stats saved")


model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="bfloat16", device_map="cuda")


@torch.no_grad()
def bits_per_byte(model, tok, texts, max_len=4096, slice_len=1024):
    total_nll, total_bytes = 0.0, 0
    for t in tqdm(texts):
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        for s in range(0, len(ids) - 1, max_len):
            chunk = ids[s : s + max_len + 1].unsqueeze(0).to(model.device)
            if chunk.shape[1] < 2:
                continue
            inp, tgt = chunk[:, :-1], chunk[:, 1:]
            hidden = model.model(inp).last_hidden_state          # (1, L, d)
            for j in range(0, inp.shape[1], slice_len):
                logits = model.lm_head(hidden[:, j : j + slice_len]).float()
                total_nll += torch.nn.functional.cross_entropy(
                    logits[0], tgt[0, j : j + slice_len], reduction="sum"
                ).item()
        total_bytes += len(t.encode("utf-8"))
    return total_nll / (math.log(2) * total_bytes)


bpb = bits_per_byte(model, tok, en_texts)
print(bpb)


save_json({"config": eval_config, "lang": lang, "n_docs": len(texts),
           "bits_per_byte": bpb}, f"{EVAL_DIR}/{tag}_{lang}_bpb.json")
print(f"{lang} BPB: {bpb:.4f}")


# Tokenizer stats on the English eval set:
# fertility (tokens/word): 1.2804896737381426
# chars/token: 4.7985698471112705
# fragment rate: 0.0013623353179945945
