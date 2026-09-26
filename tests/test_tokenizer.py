from collections import Counter

import torch

from kev.questions.bank import training_text
from kev.tokenizer.bpe import BYTES, NONE, OPTION, PAD, PATTERN, QUESTION, Tokenizer, count_chunks, merge, normalize, pad, train

ODD = ["Is there a KNIFE?", "what's   in\nhere", "naïve café 🐍 ½ ²", "日本語のテキスト", "é _under_score_", "3D-printed", ""]


def slow_train(chunk_counts: Counter, n: int) -> list[tuple[int, int]]:
    words = [([BYTES + b for b in chunk.encode()], k) for chunk, k in chunk_counts.items()]
    merges = []
    for _ in range(n):
        pairs = Counter()
        for ids, k in words:
            for p in zip(ids, ids[1:]):
                pairs[p] += k
        if not pairs:
            break
        pair = min(pairs, key=lambda p: (-pairs[p], p))
        words = [(merge(ids, pair, BYTES + 256 + len(merges)), k) for ids, k in words]
        merges.append(pair)
    return merges


def test_chunks_cover_every_character():
    for text in ODD:
        text = normalize(text)
        assert "".join(PATTERN.findall(text)) == text


def test_fast_training_learns_the_same_merges_as_recounting():
    counts = count_chunks(training_text()[:400] + ["aaaa bbbb aaaa", "the the the then"])
    assert train(counts, 300) == slow_train(counts, 300)


def test_text_survives_encoding():
    tokenizer = Tokenizer(train(count_chunks(training_text()), 200))
    for text in ODD + training_text()[:50]:
        assert tokenizer.decode(tokenizer.encode(text)) == normalize(text)


def test_special_tokens_cannot_be_typed():
    tokenizer = Tokenizer(train(count_chunks(training_text()), 200))
    ids = tokenizer.question("[q] [o] [none] [pad] [mask] is there a knife?")
    assert ids[0] == QUESTION and min(ids[1:]) >= BYTES
    assert tokenizer.option("knife")[0] == OPTION and tokenizer.none_option() == [OPTION, NONE]


def test_saved_tokenizer_loads_the_same_and_can_be_cut_short(tmp_path):
    tokenizer = Tokenizer(train(count_chunks(training_text()), 200))
    tokenizer.save(tmp_path / "t.json")
    loaded = Tokenizer.load(tmp_path / "t.json")
    assert loaded.merges == tokenizer.merges and loaded.encode("any knives?") == tokenizer.encode("any knives?")
    short = Tokenizer.load(tmp_path / "t.json", merges=50)
    assert len(short) == BYTES + 256 + 50 and short.merges == tokenizer.merges[:50]


def test_words_group_the_same_ids():
    tokenizer = Tokenizer(train(count_chunks(training_text()), 200))
    words = tokenizer.encode_words("do you reckon there are spanners in here?")
    assert [i for w in words for i in w] == tokenizer.encode("do you reckon there are spanners in here?")
    assert tokenizer.decode(words[5]) == " spanners"


def test_pad_marks_real_tokens():
    ids, mask = pad([[1, 7, 9], [2, 5]])
    assert ids.tolist() == [[1, 7, 9], [2, 5, PAD]]
    assert torch.equal(mask, torch.tensor([[True, True, True], [True, True, False]]))
