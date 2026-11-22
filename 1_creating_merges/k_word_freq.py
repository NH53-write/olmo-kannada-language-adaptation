from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tokenizers import Tokenizer

import json

import unicodedata, pickle, os
from collections import Counter
from tqdm import tqdm

from itertools import pairwise


## Loading the tokenizer
model_dir = f"/Users/nirajh/Code/Language_adaptation/olmo2-7b-stage1"

print(f"Loading the tokenizer from the following path:\n{model_dir}\n")
tok = AutoTokenizer.from_pretrained(model_dir)


## Loading the dataset
dataset_dir = "/Users/nirajh/Code/Language_adaptation/sangraha/train_split"
print(f"Loading the kannada dataset from the following path:\n{dataset_dir}\n")

# Load the dataset 
kn_train = load_dataset(
    "ai4bharat/sangraha",
    data_dir="verified/kan",
    cache_dir=dataset_dir,
    download_mode="reuse_cache_if_exists"
)


## Changing the regex pattern in the pre tokenizer, so that kannada words are not split incorrectly
tj = json.loads(tok.backend_tokenizer.to_str())
split = tj["pre_tokenizer"]["pretokenizers"][0]
old = split["pattern"]["Regex"]
new = old.replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+")
assert new != old, "pattern not found; print(old) and adjust"
split["pattern"]["Regex"] = new

# Checking its output
kn_tok = Tokenizer.from_str(json.dumps(tj))
text = "ನಮಸ್ಕಾರ ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ"
print([text[s:e] for _, (s, e) in kn_tok.pre_tokenizer.pre_tokenize_str(text)])


eng_txt_path = "/Users/nirajh/Code/Language_adaptation/test_eng_page.txt"

with open(eng_txt_path, "r", encoding="utf-8") as file:
    en_text = file.read()


# Confirming that this doesn't adversely affect English tokens
en_toks = 0
diff = 0
for t in en_text:
    if tok.encode(t, add_special_tokens=False) != kn_tok.encode(t, add_special_tokens=False).ids:
        diff += 1
    en_toks += 1

print("\nEnglish docs with different tokenization (after changing the regex pattern in the pretokenizer):", diff)   # should be 0 or very close
print(f"\nNumber of english docs processed with both tokenizers: {en_toks}")


## Tokenizing the input dataset

print("\nTokenizing the kannada train dataset...")
word_freq = Counter()
pre = kn_tok.pre_tokenizer

ds = kn_train["train"]
for start in tqdm(range(0, len(ds), 1000)):
    for t in ds[start : start + 1000]["text"]:
        t = unicodedata.normalize("NFC", t)
        word_freq.update(w for w, _ in pre.pre_tokenize_str(t))   # byte-level strings, e.g. 'à²ķà²¨...'

print(len(word_freq), "unique pre-tokens in the kannada dataset")       # Output: 23277887 unique pre-tokens
print(word_freq.most_common(10))


MIN_FREQ = 3   # drop words seen fewer than 3 times; record this in your config

print(f"Unique pre-tokens before filtering: {len(word_freq):,}")
filtered = Counter({w: c for w, c in word_freq.items() if c >= MIN_FREQ})
print(f"Unique pre-tokens after filtering (freq >= {MIN_FREQ}): {len(filtered):,}")
print(f"Share of all occurrences kept: {sum(filtered.values()) / sum(word_freq.values()):.2%}")


output_path = "/Users/nirajh/Code/Language_adaptation/merges_work/"
os.makedirs(output_path, exist_ok=True)

final_path = os.path.join(output_path, "word_freq.pkl")
tmp_path = final_path + ".tmp"

payload = {
    "word_freq": filtered,
    "min_freq": MIN_FREQ,
    "n_unique_before_filter": len(word_freq),
    "n_occurrences_before_filter": sum(word_freq.values()),
}

with open(tmp_path, "wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.flush()
    os.fsync(f.fileno())


print("Wrote tmp file to disk")

os.replace(tmp_path, final_path)

print("Replaced tmp file with the final file")

# verify by reading it back
with open(final_path, "rb") as f:
    check = pickle.load(f)
assert len(check["word_freq"]) == len(filtered)
print(f"Wrote and verified {final_path} ({os.path.getsize(final_path) / 1e6:.1f} MB)")

