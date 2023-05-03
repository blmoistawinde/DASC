import re 
from collections import defaultdict, Counter

def recover_sentence(sent, tokenizer_name="fnlp/bart-base-chinese"):
    if "chinese" in tokenizer_name:
        sent = re.sub(r"([a-zA-Z0-9!%&*\(\){}\[\]:;\"\'<>,./?_\-+=]) ([a-zA-Z0-9!%&*\(\){}\[\]:;\"\'<>,./?_\-+=])", r"\1\n\2", sent)
        sent = sent.replace(" ", "").replace("\n", " ")
    else:
        sent = sent.replace("\n", " ")
    return sent

def distinct_metric(seqs):
    # seqs: list of iterables (str / tokens list)
    unigrams_all, bigrams_all, trigrams_all = Counter(), Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        trigrams = Counter(zip(seq, seq[1:], seq[2:]))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)
        trigrams_all.update(trigrams)

    dist1 = len(unigrams_all) / max(sum(unigrams_all.values()), 1)
    dist2 = len(bigrams_all) / max(sum(bigrams_all.values()), 1)
    dist3 = len(trigrams_all) / max(sum(trigrams_all.values()), 1)
    dist = (dist1 + dist2 + dist3) / 3
    return dist

def distinct_2(seqs):
    # seqs: list of iterables (str / tokens list)
    bigrams_all = Counter()
    for seq in seqs:
        bigrams = Counter(zip(seq, seq[1:]))

        bigrams_all.update(bigrams)

    dist2 = len(bigrams_all) / max(sum(bigrams_all.values()), 1)
    return dist2
