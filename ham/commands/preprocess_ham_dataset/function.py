import argparse
import logging
import pathlib
import re
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def get_index_of_human_attention(html: str, num_words_in_review: int) -> List[int]:

    pat = re.compile("<span(.*?)/span>")
    all_span_items = pat.findall(html)

    if html == "{}":
        raise ValueError(
            f'Empty human annotation in "{html}" - This should never print.'
        )

    indexes = []
    if len(all_span_items) == num_words_in_review + 1:
        if (all_span_items[num_words_in_review] == "><") or (
            all_span_items[num_words_in_review] == 'ata-vivaldi-spatnav-clickable="1"><'
        ):
            for i, span_item in enumerate(all_span_items):
                if 'class="active"' in span_item:
                    indexes.append(i)
        else:
            raise ValueError("This should never print.")

    return indexes


def _read_csv(csv_file: pathlib.Path) -> pd.DataFrame:
    logger.info(f"Read file from {csv_file}")
    df = pd.read_csv(csv_file)
    df.columns = ["label", "text", "annotator_answer", "ham_annotation"]
    return df


def _save_to_jsonl(df: pd.DataFrame, output_file: pathlib.Path) -> pd.DataFrame:
    logger.info(f"Save to {output_file}")
    df.to_json(output_file, lines=True, orient="records")


def _preprocess_ham_dataset(df: pd.DataFrame) -> None:

    num_words_in_review = len(df["text"][0].split())
    df["ham_annotation_index"] = df["ham_annotation"].apply(
        lambda x: get_index_of_human_attention(x, num_words_in_review)
    )
    return df


def preprocess_ham_dataset(args: argparse.Namespace) -> None:

    csv_files = [p for p in args.ham_dataset_dir.iterdir() if p.suffix == ".csv"]

    for csv_file in csv_files:
        df = _read_csv(csv_file)
        df = _preprocess_ham_dataset(df)

        _save_to_jsonl(df, args.output_dir / f"{csv_file.stem}.jsonl")
