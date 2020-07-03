from typing import Dict

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

from ham.modules.attention import Attention


@Model.register("rnn")
class RNNModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        word_embedding: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        attention: Attention,
        regularizer: RegularizerApplicator = None,
    ):
        super().__init__(vocab, regularizer=regularizer)

        self._word_embedding = word_embedding
        self._encoder = encoder
        self._attention = attention
        self._output = nn.Linear(attention.get_output_dim(), 2)

        self._loss = nn.CrossEntropyLoss()
        self._metrics = {"acc1": CategoricalAccuracy()}

    @overrides
    def forward(
        self, tokens: torch.Tensor, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens)

        embedding = self._word_embedding(tokens)
        hidden = self._encoder(embedding, mask)
        attention = self._attention(hidden, mask)
        logit = self._output(weighted_sum(hidden, attention))

        output_dict: Dict[str, torch.Tensor] = {"logit": logit, "attention": attention}

        if label is not None:
            output_dict["loss"] = self._loss(logit, label)

            for eval_metric in self._metrics.values():
                eval_metric(logit, label)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_scores = {
            name: metric.get_metric(reset) for name, metric in self._metrics.items()
        }
        return metric_scores
