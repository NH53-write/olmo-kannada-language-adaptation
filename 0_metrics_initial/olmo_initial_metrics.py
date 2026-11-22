import os
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/root/.cache/huggingface/datasets"

from datasets import load_dataset
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
from tqdm import tqdm

model_dir = "/workspace/language_adaptation/olmo2-7b-stage1-step928646-tokens3896B"

# Loading the tokenizer
tok = AutoTokenizer.from_pretrained(model_dir)

# Load dataset
kn = load_dataset(
    "ai4bharat/sangraha",
    data_dir="verified/kan",
    split="train",
    cache_dir="/workspace/language_adaptation/dataset",
)

print("Dataset:\n",kn)                  # shows columns and row count
print(kn[0]["text"][:500])

# Disable caching as it leads to permission errors on /workspace directories
datasets.disable_caching()

kn_texts = []

with open("/workspace/language_adaptation/eval/kn_eval.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        kn_texts.append(data["text"])

print(f"Successfully read {len(kn_texts)} lines.")   # Should be 1997


def tokenizer_stats(tok, texts):
    n_tok = n_words = n_chars = n_frag = 0
    for t in texts:
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

print("Tokenizer stats:\n",tokenizer_stats(tok, kn_texts))

#### English benchmarks

import lm_eval
from lm_eval.models.huggingface import HFLM

# Loading the model
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="bfloat16", device_map="cuda")

# Moving the model to the GPU
model = model.to("cuda")

# Creating a HF language model
lm = HFLM(pretrained=model, tokenizer=tok, batch_size="auto")

res = lm_eval.simple_evaluate(
    model=lm,
    tasks=["hellaswag", "arc_easy", "mmlu"],
)

with open("/workspace/language_adaptation/eval/baseline_en_benchmarks.json", "w") as f:
    json.dump(res["results"], f, indent=2, default=str)

