import argparse
import logging
from typing import Dict

import pandas as pd
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

    # Assign 1 and 2-star reviews to the negative class,
    # and 4 and 5-star reviews to the positive class.
    df = df[df.stars.isin([1, 2, 4, 5])]
    df["stars"] = df["stars"].apply(_to_binary)

    return df[["text", "stars"]]


def _split_to_tng_val_tst(
    df: pd.DataFrame, val_size: float, tst_size: float
) -> Dict[str, pd.DataFrame]:

    df_tng, df_tst = train_test_split(
        df, test_size=(val_size + tst_size), stratify=df.stars
    )

    df_val, df_tst = train_test_split(
        df_tst, test_size=tst_size / (val_size + tst_size), stratify=df_tst.stars
    )

    tng_ratio = len(df_tng) / len(df)
    val_ratio = len(df_val) / len(df)
    tst_ratio = len(df_tst) / len(df)

    logger.info(
        "Splitted dataset into tng : val : tst "
        f"= {tng_ratio:.3f} : {val_ratio:.3f} : {tst_ratio:.3f}"
    )

    return {"tng": df_tng, "val": df_val, "tst": df_tst}


def split_dataset(args: argparse.Namespace) -> None:

    all_ratio = args.tng_ratio + args.val_ratio + args.tst_ratio
    assert (
        all_ratio == 1
    ), f"Invalid ratio for splitting dataset: tng + val + tst = {all_ratio:.1f} != 1.0"

    df = _load_yelp_dataset(args.input_file)
    df_dict = _split_to_tng_val_tst(
        df, val_size=args.val_ratio, tst_size=args.tst_ratio
    )

    # make output directory if not exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for phase in ["tng", "val", "tst"]:
        dataset_fpath = args.output_dir / vars(args).get(f"{phase}_filename")
        logger.info(f"Output {phase} set to {dataset_fpath}")
        df_dict[phase].to_json(dataset_fpath, orient="records", lines=True)
