from tokenizers import Tokenizer, models, trainers, normalizers, decoders, pre_tokenizers
from transformers import AutoTokenizer
from datasets import load_dataset

import json, copy, sys, pickle, os
from tqdm.notebook import tqdm
import unicodedata
from math import ceil

model_dir = "/Users/nirajh/Code/Language_adaptation/olmo2-7b-stage1"


#### Loading the tokenizer and modifying its regex pattern

# Loading the tokenizer
tok = AutoTokenizer.from_pretrained(model_dir)

## Changing the regex pattern in the pre tokenizer, so that kannada words are split correctly
tj = json.loads(tok.backend_tokenizer.to_str())        # Getting the tokenizer's vocab, merges, special tokens, and pre-processing configs
split = tj["pre_tokenizer"]["pretokenizers"][0]        # Fetches the first pre-tokenizer (they are usually chained together under a Sequence)
old = split["pattern"]["Regex"]
new = old.replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+")
assert new != old, "pattern not found; print(old) and adjust"
split["pattern"]["Regex"] = new

# Checking its output
kn_tok = Tokenizer.from_str(json.dumps(tj))
text = "ನಮಸ್ಕಾರ ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ"
print([text[s:e] for _, (s, e) in kn_tok.pre_tokenizer.pre_tokenize_str(text)])


#### Loading the new merges list

# Loading the new merges list
final_path = "/Users/nirajh/Code/Language_adaptation/merges_files/new_merges.pkl"

with open(final_path, "rb") as f:
    merges_ = pickle.load(f)


#### Loading the eval set

kn_texts = []

with open("/Users/nirajh/Code/Language_adaptation/eval/kn_eval_texts.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        kn_texts.append(data["text"])

print(f"Successfully read {len(kn_texts)} lines.")   # Should be 1997


en_texts = []

with open("/Users/nirajh/Code/Language_adaptation/eval/en_eval_texts.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        en_texts.append(data["text"])

print(f"Successfully read {len(en_texts)} lines.")   # Should be 2000 lines


#### Loading the dataset

dataset_dir = "/Users/nirajh/Code/Language_adaptation/sangraha/train_split"
print(f"Loading the kannada dataset from the following path:\n{dataset_dir}\n")

# Load the dataset 
kn_train = load_dataset(
    "ai4bharat/sangraha",
    data_dir="verified/kan",
    cache_dir=dataset_dir,
    download_mode="reuse_cache_if_exists"
)

#### Unigram function

def train_unigram(vocab_size, text_iter, length=None):
    u = Tokenizer(models.Unigram())
    u.normalizer = normalizers.NFC()
    u.pre_tokenizer = kn_tok.pre_tokenizer            # same regex fix + byte-level step
    u.decoder = decoders.ByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # all 256 bytes: nothing is ever unknown
        show_progress=True
    )
    u.train_from_iterator(text_iter, trainer=trainer, length=length)
    return u


def text_batches(ds, batch_size=1000):
    for i in range(0, len(ds), batch_size):
        yield ds[i : i + batch_size]["text"]

# kn_train is >3.6M in size; the below line is selecting 300k
ds = kn_train["train"].shuffle(seed=42).select(range(300_000))   # sample; record the size in your config

tok_6000 = train_unigram(6000, text_batches(ds), length=len(ds))
print("Vocabulary size: ",tok_6000.get_vocab_size())


out_dir = "/Users/nirajh/Code/Language_adaptation/tokenizers/unigram"
os.makedirs(out_dir, exist_ok=True)

tok_6000.save(os.path.join(out_dir, "unigram_6000.json"))

