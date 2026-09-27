"""kev: a scan and a set of typed questions in, a probability for every option out, in one pass.

The text encoder stays frozen so held-out wording keeps its meaning; small readers learn to sum up each question's
and option's word vectors. The question decoder lets questions look at each other and at the scan's patches.
"""

import torch
from torch import nn

from kev.model.blocks import WIDTH, Attention, DecoderBlock
from kev.model.image import ImageEncoder
from kev.model.text import TextEncoder

QUESTION_TYPES = ("yes_no", "one_of", "any_of")  # one answer from yes/no; one answer or none; any number of answers
YES_NO, ONE_OF, ANY_OF = range(len(QUESTION_TYPES))
DECODER_LAYERS = 3


class Reader(nn.Module):
    """Sums up a question's or an option's word vectors into one vector."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.query, std=0.02)
        self.attention = Attention(width)
        self.norm = nn.LayerNorm(width)

    def forward(self, words: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(len(words), -1, -1)
        return self.norm(self.attention(query, mask, context=words))[:, 0]


class Kev(nn.Module):
    def __init__(self, image: ImageEncoder, text: TextEncoder, layers: int = DECODER_LAYERS, width: int = WIDTH):
        super().__init__()
        self.image = image
        self.text = text.requires_grad_(False).eval()
        self.question_reader = Reader(width)
        self.option_reader = Reader(width)
        self.none = nn.Parameter(torch.zeros(width))  # "none of these"
        self.types = nn.Parameter(torch.zeros(len(QUESTION_TYPES), width))
        for table in (self.none, self.types):
            nn.init.normal_(table, std=0.02)
        self.blocks = nn.ModuleList(DecoderBlock(width) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.answer = nn.Linear(width, width)
        self.any_of_bias = nn.Parameter(torch.zeros(()))  # where an any-of score tips from no to yes

    def train(self, mode: bool = True) -> "Kev":
        super().train(mode)
        self.text.eval()
        return self

    def read(self, reader: Reader, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(..., tokens) ids -> (..., width), through the frozen text encoder."""
        lead = ids.shape[:-1]
        ids, mask = ids.reshape(-1, ids.shape[-1]), mask.reshape(-1, mask.shape[-1]).clone()
        mask[:, 0] |= ~mask.any(dim=1)  # padding and "none" have no words; one dummy token keeps attention finite
        with torch.no_grad():
            words = self.text(ids, mask)
        return reader(words, mask).reshape(*lead, -1)

    def forward(self, scan: dict, questions: dict) -> torch.Tensor:
        """Scores (batch, questions, options). Softmax them for yes_no and one_of, sigmoid(score + any_of_bias)
        each for any_of; see probabilities()."""
        memory = self.image(scan["patches"], scan["row"], scan["col"], scan["view"], scan["valid"])
        x = self.read(self.question_reader, questions["ids"], questions["mask"]) + self.types[questions["type"]]
        for block in self.blocks:
            x = block(x, questions["valid"], memory, scan["valid"])
        answers = self.answer(self.norm(x))
        options = self.read(self.option_reader, questions["option_ids"], questions["option_mask"])
        options = torch.where(questions["is_none"][..., None], self.none, options)
        return torch.einsum("bqw,bqow->bqo", answers, options) / answers.shape[-1] ** 0.5

    def probabilities(self, scores: torch.Tensor, questions: dict) -> torch.Tensor:
        scores = scores.masked_fill(~questions["option_valid"], float("-inf"))
        single = scores.softmax(dim=-1)
        each = torch.sigmoid(scores + self.any_of_bias)
        return torch.where((questions["type"] == ANY_OF)[..., None], each, single).nan_to_num(0.0)
