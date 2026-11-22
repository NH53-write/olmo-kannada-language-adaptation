from tokenizers import Tokenizer, models, trainers, normalizers, decoders, pre_tokenizers, Regex
from transformers import AutoTokenizer
from datasets import load_dataset

import json, copy, sys, pickle, os
from tqdm import tqdm
import unicodedata
from math import ceil

model_dir = "/Users/nirajh/Code/Language_adaptation/olmo2-7b-stage1"


#### Loading the tokenizer and modifying its regex pattern

# Loading the tokenizer
tok = AutoTokenizer.from_pretrained(model_dir)

## Changing the regex pattern in the pre tokenizer, so that kannada words are split correctly

tj = json.loads(tok.backend_tokenizer.to_str())        # Getting the tokenizer's vocab, merges, special tokens, and pre-processing configs
split = tj["pre_tokenizer"]["pretokenizers"][0]        # Fetches the first step of the pre-tokenizer (the steps are usually chained together as a Sequence)
# This is the Split step: tj["pre_tokenizer"]["pretokenizers"][0], which contains the regex pattern.
# The ByteLevel step is tj["pre_tokenizer"]["pretokenizers"][1] (these two steps are a part of a Sequence). 
# It runs after the split and converts each chunk's UTF-8 bytes into the stand-in characters (Ġ, à² and so on).
# Since the ByteLevel step was never changed earlier, the kn_tok.pretokenizer produced byte-level strings. 


old = split["pattern"]["Regex"]
new = old.replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+")
assert new != old, "pattern not found; print(old) and adjust"
split["pattern"]["Regex"] = new


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

# Creating a pre tokenizer that splits on characters instead of bytes

split_cfg = tj["pre_tokenizer"]["pretokenizers"][0]          # the regex-fixed Split
char_split = pre_tokenizers.Split(                      # Creating a new tokenizer that doesn't have the ByteLevel step
    Regex(split_cfg["pattern"]["Regex"]),
    behavior=split_cfg["behavior"].lower(),                  # "Isolated" -> "isolated"
    invert=split_cfg["invert"],                 # Copying OLMo's config for invert
)
# The above code uses pre_tokenizers.Split(). There is also pre_tokenizers.ByteLevel().
# A tokenizer with both steps can be built as: pre_tokenizers.Sequence([pre_tokenizers.Split(), pre_tokenizers.ByteLevel()])


text = "ನಮಸ್ಕಾರ ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ"
kn_tok = Tokenizer.from_str(json.dumps(tj))
print("Pre tokenizer without ByteLevel step: \n",char_split.pre_tokenize_str(text))         # [('ನಮಸ್ಕಾರ', (0, 7)), (' ಕನ್ನಡ', (7, 13))]
print("Pre tokenizer with ByteLevel step: \n",kn_tok.pre_tokenizer.pre_tokenize_str(text))  # same chunks, but as byte-level strings ('Ġà²ķ...')

KANNADA_CHARS = [chr(c) for c in range(0x0C80, 0x0D00) if unicodedata.name(chr(c), None)]       
# The above list returns 91 characters. Apart from vowels and consonants, it includes ottaksharas and digits. There are even some rare/obsolete/vedic/scholarly-use characters.
# The if condition in the list removes code points within the Kannada Unicode block that are unassigned. If there's no Unicode character name, that character is unassigned.
# The range in the loop returns 128 characters


def train_unigram(vocab_size, text_iter, length=None):
    u = Tokenizer(models.Unigram())
    u.normalizer = normalizers.NFC()
    u.pre_tokenizer = char_split                              # regex split only, no ByteLevel
    u.decoder = decoders.Fuse()                               # decoding = concatenate pieces

    # The above two lines have replaced the following two:
    # u.pre_tokenizer = kn_tok.pre_tokenizer            # same regex fix + byte-level step
    # u.decoder = decoders.ByteLevel()
    
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        initial_alphabet=KANNADA_CHARS,         # Passing Kannada characters instead of the byte-level alphabet
        unk_token="<unk>",                      # Need an unk token as we don't have byte-level chars anymore
        special_tokens=["<unk>"],
        show_progress=True,

        # vocab_size=vocab_size,
        # initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # all 256 bytes: nothing is ever unknown
        # show_progress=True
    )
    u.train_from_iterator(text_iter, trainer=trainer, length=length)
    return u


def text_batches(ds, batch_size=1000):
    for i in range(0, len(ds), batch_size):
        yield ds[i : i + batch_size]["text"]

# # kn_train is >3.6M in size
ds = kn_train["train"].shuffle(seed=42)      # If this is passed direclty to train_unigram, it leads to the error "The vocabulary is not large enough to contain all chars"

# The insufficient vocab error occurs because the HF Unigram trainer treats every distinct character seen in the training text as required.
# The full training text contains non-Kannada characters as well, which increases the number of distinct characters beyond the specified vocabulary size of 4000.

# Filtering out non-Kannada characters from the training text to avoid exceeding the vocabulary size 
KN_OK = set(KANNADA_CHARS) | {" "}

def is_kannada_chunk(c):
    return any("\u0c80" <= ch <= "\u0cff" for ch in c) and all(ch in KN_OK for ch in c)
# The any() condition is checking for the same range of characters that the loop used to generated KANNADA_CHARS checked
# Since range() is not end inclusive, the character after \u0cff was used there.


def split_leading(c):
    """
    '(ಮನೆ' -> ('(', 'ಮನೆ'); ' ಮನೆ' and 'ಮನೆ' are left whole.
    This function separates leading non-Kannada characters from a chunk, if present.
    """
    if len(c) > 1 and c[0] != " " and not ("\u0c80" <= c[0] <= "\u0cff"):
        return c[0], c[1:]
    return "", c
# We use split_leading() as the pre-tokenizer's regex pattern allows one non-letter character at the start of chunk.
# A split_trailing() function is not needed as the chunk ends at the last letter  or mark. Regex will not match a trailing punctuation or digit character.


def kannada_chunk_batches(ds, batch_size=1000):
    for i in range(0, len(ds), batch_size):
        chunks = []
        for t in ds[i : i + batch_size]["text"]:
            for c, _ in char_split.pre_tokenize_str(unicodedata.normalize("NFC", t)):
                _, rest = split_leading(c)
                if is_kannada_chunk(rest):
                    chunks.append(rest)
        yield chunks

tok_4000 = train_unigram(4000, kannada_chunk_batches(ds))
# tok_4000 = train_unigram(4000, text_batches(ds), length=len(ds))
print(tok_4000.get_vocab_size())               # should be 4000


out_dir = "/Users/nirajh/Code/Language_adaptation/tokenizers/unigram"
os.makedirs(out_dir, exist_ok=True)

tok_4000.save(os.path.join(out_dir, "unigram_4000_char.json"))
