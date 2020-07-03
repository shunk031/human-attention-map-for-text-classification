import json
import logging
from typing import Dict, Iterable

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("yelp")
class YelpDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        cache_directory: str = None,
        max_instances: int = None,
        manual_distributed_sharding: bool = False,
        manual_multi_process_sharding: bool = False,
    ):
        super().__init__(
            lazy=lazy,
            cache_directory=cache_directory,
            max_instances=max_instances,
            manual_distributed_sharding=manual_distributed_sharding,
            manual_multi_process_sharding=manual_multi_process_sharding,
        )

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        with open(file_path, "r") as rf:
            for line in rf.readlines():
                if not line:
                    continue

                items = json.loads(line)
                yield self.text_to_instance(**items)

    @overrides
    def text_to_instance(self, text: str, stars: str) -> Instance:
        tokens = self._tokenizer.tokenize(text)
        tokens = TextField(tokens, self._token_indexers)

        label = LabelField(str(stars))

        fields: Dict[str, Field] = {"tokens": tokens, "label": label}
        return Instance(fields)
