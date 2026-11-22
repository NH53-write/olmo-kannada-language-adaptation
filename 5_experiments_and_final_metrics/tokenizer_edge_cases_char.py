"""
Edge-case comparisons between the expanded BPE tokenizer and the BPE + Unigram hybrid.

Every comparison works on pre-tokens produced by the SAME regex-fixed pre-tokenizer
(kn_tok.pre_tokenizer) after NFC normalization, so the only difference between the
two tokenizers is how each pre-token is segmented.

A "segmenter" is a function: byte-level pre-token string -> list of byte-level piece strings.

"""

import math
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

# ---------------------------------------------------------------------------
# Byte-level helpers (same table as GPT-2 / OLMo's byte-level BPE)
# ---------------------------------------------------------------------------

@lru_cache()
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~")+1))
            + list(range(ord("¡"), ord("¬")+1))
            + list(range(ord("®"), ord("ÿ")+1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


BYTE_ENCODER = bytes_to_unicode()

def to_bytelevel(s):
    """Unicode string -> OLMo byte-level string (space becomes 'Ġ')."""
    return "".join(BYTE_ENCODER[b] for b in s.encode("utf-8"))


byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

def to_bytes(s):
    return bytes(byte_decoder[c] for c in s)

def to_text(s):
    return to_bytes(s).decode("utf-8", errors="replace")

def is_kannada(w):
    """Byte-level string contains a Kannada character (lead bytes E0 B2 / E0 B3)."""
    return "à²" in w or "à³" in w


# ---------------------------------------------------------------------------
# Segmenters
# ---------------------------------------------------------------------------

def bpe_segmenter(bpe_tok):
    """
    Expanded BPE tokenizer (a transformers tokenizer built with build_tokenizer).
    This includes the new merges that were learnt.
    """
    model = bpe_tok.backend_tokenizer.model
    return lambda w: [t.value for t in model.tokenize(w)]


def split_leading(text):
    """'(ಮನೆ' -> ('(', 'ಮನೆ'); ' ಮನೆ' and 'ಮನೆ' are left whole."""
    if len(text) > 1 and text[0] != " " and not ("\u0c80" <= text[0] <= "\u0cff"):
        return text[0], text[1:]
    return "", text


def hybrid_char_segmenter(u_char, base_bpe_model):
    unk_id = u_char.token_to_id("<unk>")
    def seg(w):
        if not is_kannada(w):
            return [t.value for t in base_bpe_model.tokenize(w)]
        lead, rest = split_leading(to_text(w))
        toks = u_char.model.tokenize(rest)
        if any(t.id == unk_id for t in toks):                # check the ID, not the value; the unknown token's value is the original text, not the string "<unk>"; Therefore, t.value == "<unk>" doesn't fire
            return [t.value for t in base_bpe_model.tokenize(w)]
        # When model.tokenize meets an unknown character, the token it returns carries the original text ('a') as its value and the unknown token's ID
        # That's why the token's ID should be checked (to determine if it is an unknown token) and not the token's value

        lead_pieces = [t.value for t in base_bpe_model.tokenize(to_bytelevel(lead))] if lead else []
        return lead_pieces + [to_bytelevel(t.value) for t in toks]
    return seg


def cached(seg):
    """Memoize a segmenter: the same words repeat constantly in running text."""
    cache = {}

    def f(w):
        r = cache.get(w)
        if r is None:
            r = cache[w] = seg(w)
        return r

    return f


def pretokens(text, pre_tokenizer):
    text = unicodedata.normalize("NFC", text)
    return [w for w, _ in pre_tokenizer.pre_tokenize_str(text)]


def kannada_word_counts(texts, pre_tokenizer):
    """Occurrence counts of Kannada pre-tokens in a list of documents."""
    words = Counter()
    for t in texts:
        for w in pretokens(t, pre_tokenizer):
            if is_kannada(w):
                words[w] += 1
    return words


# ---------------------------------------------------------------------------
# New-token inventories (for fair size matching and dead-token counts)
# ---------------------------------------------------------------------------

def bpe_new_tokens(bpe_tok, first_new_id=100278):
    return {t for t, i in bpe_tok.get_vocab().items() if i >= first_new_id}
    # The above if condition ensures that only the IDs that are not in the base BPE vocab are returned.


def unigram_char_new_tokens(u_char, orig_vocab):
    """Kannada pieces (converted to byte-level) that OLMo's vocab doesn't have."""
    out = set()
    for p in u_char.get_vocab():
        if p == "<unk>":                # Here, 'p' is from the vocabulary and is not a string that the tokenizer couldn't tokenize; therefore, it's string is "<unk>"
            continue
        bl = to_bytelevel(p)
        if is_kannada(bl) and bl not in orig_vocab:
            out.add(bl)
    return out


def fragment_rate(texts, pre_tokenizer, segmenter, kannada_only=False):
    n = frag = 0
    for t in texts:
        for w in pretokens(t, pre_tokenizer):
            if kannada_only and not is_kannada(w):
                continue
            for p in segmenter(w):
                n += 1
                frag += "\ufffd" in to_text(p)
    return frag / n

# ---------------------------------------------------------------------------
# 1. Morphological consistency: is there a token boundary where a suffix starts?
# ---------------------------------------------------------------------------

# Surface endings, starting on a consonant so the boundary falls between characters.
# This is a heuristic starting list: edit it to match the morphology you care about.

# List of Kannada ಸಂಧಿಗಳು: https://kannadagrammar.com/sandhigalu-in-kannada-vyakarana-kannada-grammar/
# Typing: https://www.easykannadatyping.com/



DEFAULT_SUFFIXES = [
    "ನ್ನು",    # accusative   (ಮರವನ್ನು, ಮನೆಯನ್ನು)
    "ಲ್ಲಿ",     # locative     (ಮರದಲ್ಲಿ, ಮನೆಯಲ್ಲಿ)
    "ಕ್ಕೆ",     # dative       (ಮರಕ್ಕೆ)
    "ಗೆ",      # dative       (ಮನೆಗೆ)
    "ಗಳು",     # plural       (ಮರಗಳು)
    "ಗಳ",      # plural stem / genitive (ಮರಗಳ)
    "ತ್ತಾರೆ",   # verb, present 3rd person plural/honorific
    "ತ್ತದೆ",    # verb, present 3rd person neuter
    "ನ್ನೇ",       
    "ಲ್ಲದೆ"

    # "ಾಗಿ",          # Can't use these suffixes in the suffix_boundary_rate function, as they are likely to be parts of tokens
    # "ಂತೇ",
    # "ಿಂದ",
    # "ಾಲಯ",
    # "ುದ್ಧಿ"
]


def suffix_boundary_rate(texts, pre_tokenizer, segmenters, suffixes=DEFAULT_SUFFIXES, min_stem_chars=2):
    """
    For each Kannada word occurrence that ends in one of `suffixes` (longest match wins),
    check whether the segmentation has a token boundary exactly where the suffix starts.
    
    That is, it checks if some tokens ends exactly where the suffix begins. 
    So it is checking if the tokenizer cuts at the word stem-suffix boundary.
    If a suffix is a part of a word that is a single token (eg: ಮರವನ್ನು is one token), 
    the rate (and hits) goes down.

    It checks for the longest matching suffix, so ಗಳು is matched if present, instead of stopping at ಗಳ.

    Returns {tokenizer: {suffix: (rate, n_occurrences), "ALL": (rate, n)}}.
    """
    words = kannada_word_counts(texts, pre_tokenizer)
    ordered = sorted(suffixes, key=len, reverse=True)
    hits = {name: defaultdict(lambda: [0, 0]) for name in segmenters}

    for w, c in words.items():
        b = to_bytes(w)
        core = b.decode("utf-8", errors="replace").lstrip(" ")
        for s in ordered:
            if core.endswith(s) and len(core) - len(s) >= min_stem_chars:
                cut = len(b) - len(s.encode("utf-8"))      # byte offset where the suffix begins
                for name, seg in segmenters.items():
                    ends, pos = set(), 0
                    for p in seg(w):
                        pos += len(to_bytes(p))
                        ends.add(pos)
                    for key in (s, "ALL"):
                        hits[name][key][0] += c * (cut in ends)
                        hits[name][key][1] += c
                break

    return {name: {k: (h / n if n else float("nan"), n) for k, (h, n) in d.items()}
            for name, d in hits.items()}


# ---------------------------------------------------------------------------
# 2. Fertility by training-frequency bucket (frequent vs rare vs unseen words)
# ---------------------------------------------------------------------------

DEFAULT_BUCKETS = [(10_000, ">=10k"), (1_000, "1k-10k"), (100, "100-1k"), (3, "3-100"), (0, "<3 or unseen")]


def fertility_by_frequency(texts, pre_tokenizer, segmenters, train_freq, buckets=DEFAULT_BUCKETS):
    """
    train_freq: Counter of byte-level pre-tokens from the training data
                (your saved word_freq; words below its min_freq count as unseen).
    Returns {bucket: {"word occurrences": n, "unique words": u, tokenizer: tokens/word, ...}}.
    """
    words = kannada_word_counts(texts, pre_tokenizer)
    out = {label: {"word occurrences": 0, "unique words": 0, **{name: 0 for name in segmenters}}
           for _, label in buckets}

    for w, c in words.items():
        f = train_freq.get(w, 0)
        label = next(lab for lo, lab in buckets if f >= lo)
        row = out[label]
        row["word occurrences"] += c
        row["unique words"] += 1
        for name, seg in segmenters.items():
            row[name] += c * len(seg(w))

    for row in out.values():
        n = row["word occurrences"]
        for name in segmenters:
            row[name] = row[name] / n if n else float("nan")
    return out


# ---------------------------------------------------------------------------
# 3 + 4. Token usage (dead / undertrained tokens) and vocabulary utilization
# ---------------------------------------------------------------------------

def kannada_token_counts(texts, pre_tokenizer, segmenter):
    """Counts of every piece produced for Kannada pre-tokens."""
    counts = Counter()
    for t in texts:
        for w in pretokens(t, pre_tokenizer):
            if is_kannada(w):
                counts.update(segmenter(w))
    return counts


def new_token_usage(token_counts, new_tokens, thresholds=(1, 10, 100)):
    """
    How many of the NEW tokens are rarely used in this sample.
    'used < k' includes tokens never used at all.
    """
    total = len(new_tokens)
    usage = [token_counts.get(t, 0) for t in new_tokens]
    report = {"new tokens": total, "never used": sum(u == 0 for u in usage)}
    for k in thresholds:
        report[f"used < {k}"] = sum(u < k for u in usage)
    n_all = sum(token_counts.values())
    report["share of Kannada tokens that are new"] = (
        sum(token_counts.get(t, 0) for t in new_tokens) / n_all if n_all else float("nan"))
    return report


def utilization(token_counts, alpha=2.5):
    """
    Entropy of the token distribution over Kannada text.
    Renyi efficiency here = Renyi entropy / log2(number of distinct tokens used):
    closer to 1 means the tokenizer uses its tokens more evenly.
    """
    n = sum(token_counts.values())
    probs = [c / n for c in token_counts.values()]
    shannon = -sum(p * math.log2(p) for p in probs)
    renyi = math.log2(sum(p ** alpha for p in probs)) / (1 - alpha)
    v = len(token_counts)
    return {
        "distinct tokens used": v,
        "Shannon entropy (bits)": shannon,
        f"Renyi entropy alpha={alpha} (bits)": renyi,
        "Renyi efficiency": renyi / math.log2(v) if v > 1 else float("nan"),
    }


# ---------------------------------------------------------------------------
# 5. Robustness and side-by-side inspection
# ---------------------------------------------------------------------------

# Seed examples: extend with real conversational / social-media / misspelled text.
ROBUSTNESS_EXAMPLES = [
    "ಮರ ಮರವನ್ನು ಮರದಲ್ಲಿ ಮರದಿಂದ ಮರಗಳು ಮರಗಳಲ್ಲಿ",          # one stem, several case forms
    "ಮನೆ ಮನೆಯನ್ನು ಮನೆಯಲ್ಲಿ ಮನೆಗೆ ಮನೆಗಳು",                # another stem
    "ನಾನು ನಾಳೆ office ಗೆ ಹೋಗುತ್ತೇನೆ",                       # Kannada mixed with English
    "ನನ್ನ phone number ೯೮೮೦೧೨೩೪೫೬",                       # Kannada digits
    "ಏನ್ ಮಾಡ್ತಿದೀಯ",                                       # colloquial spelling
    "ಕನ್\u200cನಡ ಕನ್ನಡ",                                     # with and without ZWNJ
]


def show_segmentations(texts, pre_tokenizer, segmenters):
    """Print each text's segmentation under every tokenizer, pieces separated by ' | '."""
    for text in texts:
        print(text)
        for name, seg in segmenters.items():
            pieces = [to_text(p) for w in pretokens(text, pre_tokenizer) for p in seg(w)]
            print(f"  {name:<8} ({len(pieces):>3} tokens): {' | '.join(pieces)}")
        print()


def robustness_stats(texts, pre_tokenizer, segmenters):
    """Tokens per whitespace word and fragment rate on each robustness example."""
    rows = []
    for text in texts:
        n_words = max(len(text.split()), 1)
        row = {"text": text}
        for name, seg in segmenters.items():
            pieces = [p for w in pretokens(text, pre_tokenizer) for p in seg(w)]
            row[f"{name} tokens/word"] = len(pieces) / n_words
            row[f"{name} fragment rate"] = (
                sum("\ufffd" in to_text(p) for p in pieces) / len(pieces) if pieces else float("nan"))
                # \ufffd is the escape sequence for the Unicode Replacement Character, which is displayed as **** 
                # It appears for corrupted or truncated data, unsupported chars, and encoding mismatches
        rows.append(row)
    return rows
