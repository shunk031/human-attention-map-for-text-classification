import logging
from typing import Dict, Iterable

import pandas as pd
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from overrides import overrides
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _to_binary(x) -> int:
    if x == 1 or x == 2:
        return 0
    elif x == 4 or x == 5:
        return 1
    else:
        ValueError(f"Invalid star:{x}")


def _load_yelp_dataset(file_path: str) -> pd.DataFrame:

    logger.info(f"Load dataset from {file_path}")
    df = pd.read_json(file_path, orient="records", lines=True)

    # Asign 1 and 2-star reviews to the negative class,
    # and 4 and 5-star reviews to the positive class.
    df = df[df.stars.isin([1, 2, 4, 5])]
    df["stars"] = df["stars"].apply(_to_binary)

    return df[["text", "stars"]]


def split_to_tng_val_tst(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:

    df_tng, df_tst = train_test_split(df, test_size=0.2, stratify=df.stars)
    df_val, df_tst = train_test_split(df_tst, test_size=0.5, stratify=df_tst.stars)

    tng_ratio = len(df_tng) / len(df)
    val_ratio = len(df_val) / len(df)
    tst_ratio = len(df_tst) / len(df)

    logger.info(
        "Splitted dataset into tng : val : tst "
        f"= {tng_ratio:.3f} : {val_ratio:.3f} : {tst_ratio:.3f}"
    )

    return {"tng": df_tng, "val": df_val, "tst": df_tst}


@DatasetReader.register("yelp")
class YelpDatasetReader(DatasetReader):
    def __init__(
        self,
        dataset_path: str,
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

        df = _load_yelp_dataset(dataset_path)
        self._df = split_to_tng_val_tst(df)

    @overrides
    def _read(self, phase: str) -> Iterable[Instance]:
        df = self._df[phase]

        for i in range(len(df)):
            yield self.text_to_instance(**df.iloc[i].to_dict())

    @overrides
    def text_to_instance(self, text: str, stars: str) -> Instance:
        tokens = self._tokenizer.tokenize(text)
        tokens = TextField(tokens, self._token_indexers)

        label = LabelField(str(stars))

        fields: Dict[str, Field] = {"tokens": tokens, "label": label}
        return Instance(fields)
