from collections import Counter
from itertools import pairwise
from collections import defaultdict

from tqdm import tqdm
import pickle
import json
import re

import gc
import copy
import os

from tokenizers import Tokenizer
from transformers import AutoTokenizer


#### Loading the tokenizer

model_dir = f"/Users/nirajh/Code/Language_adaptation/olmo2-7b-stage1"

print(f"Loading the tokenizer from the following path:\n{model_dir}\n")
tok = AutoTokenizer.from_pretrained(model_dir)


#### Modifying tokenizer

# The model's tokenizer splits on any ottakshara or virama/halanta
text = "ನಮಸ್ಕಾರ ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ"
print([text[s:e] for _, (s, e) in tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)])


## Changing the regex pattern in the pre tokenizer, so that kannada words are split correctly
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


# A note on the characters in the regex patterns:
# - \p{L} (Letter): Matches any Unicode letter from any alphabet or language (including ASCII letters).
# - \p{N} (Number): Matches any kind of numeric character in Unicode—not just standard ASCII digits (0–9), but also Roman numerals, superscript numbers, and numbers from other scripts (like Arabic-Indic digits). 
# - \p{M} (Mark): Matches a mark, which is a character that combines with another character (such as accent marks, diacritics, or vowel signs in scripts like Devanagari or Hebrew) rather than standing alone. 


#### Loading the word frequencies file

merges_path = "/Users/nirajh/Code/Language_adaptation/merges_work/"

with open(merges_path + "word_freq.pkl", "rb") as file:
    word_freq_dict = pickle.load(file)

word_freq = word_freq_dict['word_freq']
print(len(word_freq), "unique pre-tokens in the kannada dataset")       # Output: 23277887 unique pre-tokens

# Getting the model's pretokens for each word (display word); this applies merges that the model has already learned
tok_model = kn_tok.model
word_tokens = {w: [t.value for t in tok_model.tokenize(w)] for w in tqdm(word_freq, desc="Tokenizing to get each word's pretokens")}


print("Printing a few of these word-value pairs:")

ind = 0
for w, v in word_tokens.items():
    print(f"{w}, {v}")
    ind += 1
    if ind>=5:
        break


#### Checking what these Kannada words are

# To do that, convert the byte-level stand-in characters back to raw bytes, then decode the bytes as UTF-8.

from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
            #print(f"This b is not in bs: {b}; it's char is: '{chr(b)}'")
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_decoder = {v: k for k, v in bytes_to_unicode().items()}        # This inverts the byte encoder

def to_bytes(s):
    return bytes(byte_decoder[c] for c in s)         # 's' is a byte-level token string like 'Ġà²ķ'. byte_decoder() returns the byte value from the lookup table

def to_text(s):
    return to_bytes(s).decode("utf-8", errors="replace")        # Interprets the recovered bytes as UTF-8 to get real text back

for w, c in word_freq.most_common(15):
    print(f"{to_text(w)!r:25} {c:>12,}")


#### Removing pretokens that are not Kannada words

KANNADA = re.compile(r"[\u0C80-\u0CFF]")

kn_word_freq = Counter({w: c for w, c in tqdm(word_freq.items()) if KANNADA.search(to_text(w))})

print(f"Pre-tokens kept: {len(kn_word_freq):,} of {len(word_freq):,}")
print(f"Occurrences kept: {sum(kn_word_freq.values()) / sum(word_freq.values()):.2%}")

for w, c in kn_word_freq.most_common(10):
    print(f"{to_text(w)!r:25} {c:>12,}")

# Checking the pretokenized display characters and their corresponding kannada characters
for w, v in kn_word_freq.most_common(10):
    print(f'''{word_tokens[w]!r:70} {to_text(w)}''')


### Merging the most frequent token pairs

def create_pair_counts(word_freq, word_tok:dict):

    # A counter to tally the number of times a pair ocurrs
    pair_counts = Counter()

    # Creating a list of the words in which each pair occurs
    pairs_to_words = defaultdict(list)

    for idx, (display_word, count) in enumerate(tqdm(word_freq.items())):
        
        dc_list = word_tok[display_word]
        local_pairs = Counter(pairwise(dc_list))
        weighted_pairs = Counter({pair: c * count for pair, c in local_pairs.items()})
        pair_counts.update(weighted_pairs)

        # Updating the words in which a pair is seen
        for p in local_pairs.keys():
            pairs_to_words[p].append(display_word)

    return pair_counts, pairs_to_words


print("Creating counts of token pairs")
pair_counts_op, pairs_to_words_op = create_pair_counts(kn_word_freq, word_tokens)


def merge(tokens_, top_):

    new_toks = []
    i = 0
    merged_tokens = ''.join(top_)

    while i < len(tokens_):
        if (i<(len(tokens_)-1)) and (tokens_[i]==top_[0]) and (tokens_[i+1]==top_[1]):
            new_toks.append(merged_tokens)
            i += 2
        else:
            new_toks.append(tokens_[i])
            i += 1

    return new_toks


def create_merges(pair_counts_:Counter, word_freq_:Counter, word_toks_:dict, pairs_to_words_:defaultdict, check_flag:bool = False):

    # Getting the top pair
    top_pair = max(pair_counts_, key=lambda k: pair_counts_[k])

    # Getting a count of the number of times the top pair occurs
    merge_count = pair_counts_[top_pair]

    #for w in pairs_to_words_[top_pair]:
    for w in dict.fromkeys(pairs_to_words_[top_pair]):   # unique words, original order, a copy

        # Looking up the tokens in this word
        toks = word_toks_[w]

        # Counting the number of times each token pair occurs in the current word
        old_pairs = Counter(pairwise(toks))

        for p, c in old_pairs.items():
            pair_counts_[p] -= c * word_freq_[w]        # Multiplying frequency by count as a pair may occur more than once in the current word

            if pair_counts_[p] == 0:
                del pair_counts_[p]                 # Removing this pair from the Counter as its count is zero; This is faster than using 'pair_counts_ = +pair_counts_' outside this for loop, which copies the entire counter
                pairs_to_words_.pop(p, None)        # The current pair p is no longer present in any word, so its word list is stale

            elif check_flag and pair_counts_[p] < 0:
                raise AssertionError(f"Negative count for {p}: {pair_counts_[p]}")

        # Updating the tokens list for the current word directly in the dictionary
        word_toks_[w] = merge(toks, top_pair)

        for p in pairwise(word_toks_[w]):
            pair_counts_[p] += word_freq_[w]        # Adding the pair count for the new token pairs in this word; no need to multiply here as this loop is iterating over a list and not a counter

            if (p not in old_pairs) or (p not in pairs_to_words_):          # Only adding pairs that are newly formed (checking for that in two different ways)
                pairs_to_words_[p].append(w)                           # This word has one or more different pairs now; therefore, adding the word to those pairs' lists
        
        # By using pairwise() and the above two loops, we avoid deleting items based on their index (since the index shifts on deleting and it's hard to keep track for multiple occurrences of a pair in the same word)
        # The first pairwise for loop could be simpler, but that would make the second more more complex; this implementation keeps both fairly simple

    # Also removing the top pair from the pairs to words mapping
    pairs_to_words_.pop(top_pair, None)

    if check_flag:
        # Checking if the pair counts and the list of pairs match (if they do, it means the pair counts don't have to be rebuilt in every iteration of this for loop)
        new_pair_counts, _ = create_pair_counts(word_freq_, word_toks_)
        if new_pair_counts!= pair_counts_:
            print(f"Old pair counts: {len(pair_counts_)} \nNew pair counts: {len(new_pair_counts)}")
            raise AssertionError
    
    return top_pair, merge_count, pair_counts_, word_toks_, pairs_to_words_


# Making copies of the pair_counts and pair_to_words variables
pair_counts_wip = copy.deepcopy(pair_counts_op)
pairs_to_words = copy.deepcopy(pairs_to_words_op)


def save_merges(merges_):

    output_path = "/Users/nirajh/Code/Language_adaptation/merges_work/"
    os.makedirs(output_path, exist_ok=True)

    final_path = os.path.join(output_path, "new_merges.pkl")
    tmp_path = final_path + ".tmp"

    with open(tmp_path, "wb") as f:
        pickle.dump(merges_, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())

    print("Wrote tmp file to disk")
    os.replace(tmp_path, final_path)

    print("Replaced tmp file with the final file")

    # verify by reading it back
    with open(final_path, "rb") as f:
        check = pickle.load(f)

    assert len(check) == len(merges)

    print(f"Wrote and verified {final_path} ({os.path.getsize(final_path) / 1e6:.1f} MB)")


# Setting the number of merges to be done
num_merges = 16000
merges = []

for idx in tqdm(range(num_merges), desc="Merging token pairs"):

    merged_pair, merged_count, pair_counts_wip, word_tokens, pairs_to_words = create_merges(
        pair_counts_=pair_counts_wip, 
        word_freq_=kn_word_freq, 
        word_toks_=word_tokens,
        pairs_to_words_=pairs_to_words,
        check_flag=True if idx<=2 else False
    )
    merges.append((merged_pair, merged_count))

    # Saving the merges file 
    if (idx == 100) or ((idx + 1) % 500 == 0):
        print("Saving the merges list after {idx+1} merges")
        save_merges(merges)

print("Saving the final merges file")
save_merges(merges)

