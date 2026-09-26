"""What the pretrained text encoder learned: fill-in-the-blank on screening sentences, which words sit near ours,
and how highly each held-out word ranks its own item's training words.

    uv run python scripts/check_text_pretraining.py [runs/text/encoder.pt | untrained]
"""

import statistics
import sys
from itertools import batched
from pathlib import Path

import torch

from kev.device import pick_device
from kev.model.text import TextEncoder
from kev.pretrain.text import RUN
from kev.questions.bank import ITEMS, ITEMS_HELD_OUT
from kev.tokenizer.bpe import BYTES, MASK, PATTERN, Tokenizer, normalize, pad

BLANKS = [
    "he tightened the bolt with a ___.",
    "the police officer drew his ___ and fired.",
    "she cut the paper with a pair of ___.",
    "passengers must place their ___ on the conveyor belt.",
    "the ___ exploded, killing three people.",
    "he was arrested and led away in ___.",
    "is there a ___ in this bag?",
    "he lit the cigarette with a ___.",
]

# A word's vector is its average meaning across these sentences.
FRAMES = ["there is a {} in the bag", "he picked up the {}", "a {} was found", "they sold the {} to him",
          "the {} is on the table"]

# Everyday synonyms, to tell whether pretraining learned meaning at all, apart from our own items.
GENERAL_PAIRS = [("car", "automobile"), ("film", "movie"), ("doctor", "physician"), ("ship", "vessel"), ("child", "kid"),
                 ("rock", "stone"), ("road", "street"), ("sofa", "couch"), ("rubbish", "garbage"), ("shop", "store"),
                 ("jail", "prison"), ("attorney", "lawyer"), ("photograph", "picture"), ("baggage", "luggage"),
                 ("present", "gift"), ("cash", "money")]

QUERIES = ["pistol", "knife", "wrench", "spanner", "luggage", "suitcase", "handset", "beeper", "pager",
           "nightstick", "baton", "ammo", "bullet", "lighter", "scissors", "explosive", "powerbank", "syringe"]


def continuing(tokenizer: Tokenizer, text: str) -> list[int]:
    """Ids for text that follows other words, so it keeps its leading space."""
    return [i for chunk in PATTERN.findall(" " + normalize(text)) for i in tokenizer._chunk(chunk)]


@torch.no_grad()
def encode(model: TextEncoder, sequences: list[list[int]], device: torch.device) -> torch.Tensor:
    ids, mask = pad(sequences)
    return model(ids.to(device), mask.to(device))


@torch.no_grad()
def word_vectors(model: TextEncoder, tokenizer: Tokenizer, words: list[str], device: torch.device) -> torch.Tensor:
    vectors = []
    for chunk in batched(words, 200):
        sequences, spans = [], []
        for word in chunk:
            for frame in FRAMES:
                before, after = frame.split("{}")
                start = tokenizer.encode(before)
                middle = continuing(tokenizer, word)
                sequences.append(start + middle + continuing(tokenizer, after))
                spans.append((len(start), len(start) + len(middle)))
        out = encode(model, sequences, device)
        per_frame = torch.stack([out[k, a:b].mean(dim=0) for k, (a, b) in enumerate(spans)])
        vectors.append(per_frame.view(len(chunk), len(FRAMES), -1).mean(dim=1))
    return torch.cat(vectors)


def shares_a_piece(tokenizer: Tokenizer, word: str, others: set[str]) -> bool:
    pieces = lambda w: {i for i in continuing(tokenizer, w) if tokenizer.vocab[i].strip().isalpha()}
    return any(pieces(word) & pieces(other) for other in others)


def synonym_ranks(tokenizer: Tokenizer, candidates: list[str], vectors: torch.Tensor) -> None:
    """Where each word's synonym lands among all the other words. Synonyms that share a piece look alike even to an
    untrained model, so only the ones that don't show learned meaning."""
    pairs = [(n.word.lower(), {m.word.lower() for m in ITEMS[item]}, "ours")
             for item, nouns in ITEMS_HELD_OUT.items() for n in nouns]
    pairs += [(a, {b}, "general") for a, b in GENERAL_PAIRS]
    groups = {}
    for word, partners, kind in pairs:
        order = [candidates[i] for i in (vectors @ vectors[candidates.index(word)]).argsort(descending=True).tolist()
                 if candidates[i] != word]
        rank, partner = min((order.index(p) + 1, p) for p in partners)
        group = "shares a piece" if shares_a_piece(tokenizer, word, partners) else f"meaning only, {kind}"
        groups.setdefault(group, []).append((word, partner, rank))
    for group, rows in sorted(groups.items()):
        ranks = [r for _, _, r in rows]
        print(f"\n{group}: median rank {statistics.median(ranks):,.0f} of {len(candidates) - 1:,}, "
              f"{sum(r <= 10 for r in ranks)} of {len(ranks)} in the top 10")
        if group.startswith("meaning"):
            print("  " + ", ".join(f"{w}/{p} {r:,}" for w, p, r in rows))


def main() -> None:
    device = pick_device()
    tokenizer = Tokenizer.load()
    which = sys.argv[1] if len(sys.argv) > 1 else str(RUN / "encoder.pt")
    if which == "untrained":
        torch.manual_seed(0)
        model = TextEncoder(len(tokenizer)).to(device).eval()
        print("untrained encoder\n")
    else:
        saved = torch.load(Path(which), map_location=device, weights_only=True)
        model = TextEncoder(saved["vocab"]).to(device).eval()
        model.load_state_dict(saved["model"])
        print(f"{which}, step {saved['step']:,}\n")

    for sentence in BLANKS:
        before, after = sentence.split("___")
        ids = tokenizer.encode(before) + [MASK] + continuing(tokenizer, after)
        at = len(tokenizer.encode(before))
        with torch.no_grad():
            scores = model.predict(encode(model, [ids], device)[0, at]).softmax(dim=-1)
        top = scores.topk(6)
        guesses = "  ".join(f"{tokenizer.decode([i]).strip()} {p:.0%}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"{sentence:55} {guesses}")

    whole_words = [tokenizer.vocab[i].decode("utf-8", errors="ignore") for i in range(BYTES + 256, len(tokenizer))]
    whole_words = [w.strip() for w in whole_words if w.startswith(" ") and w[1:].isalpha() and len(w) > 4]
    ours = [n.word.lower() for table in (ITEMS, ITEMS_HELD_OUT) for nouns in table.values() for n in nouns]
    general = [w for pair in GENERAL_PAIRS for w in pair]
    candidates = list(dict.fromkeys(whole_words + ours + QUERIES + general))
    vectors = word_vectors(model, tokenizer, candidates, device)
    vectors = vectors - vectors.mean(dim=0)  # otherwise every word looks similar to every other
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    print(f"\nnearest of {len(candidates):,} words")
    for query in QUERIES:
        similarity = vectors @ vectors[candidates.index(query)]
        best = [i for i in similarity.argsort(descending=True).tolist() if candidates[i] != query][:8]
        print(f"  {query:11} {', '.join(candidates[i] for i in best)}")

    synonym_ranks(tokenizer, candidates, vectors)


if __name__ == "__main__":
    main()
