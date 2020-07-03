import argparse
import pathlib

from allennlp.commands.subcommand import Subcommand

from ham.commands.split_dataset.function import split_dataset


@Subcommand.register("split-dataset")
class SplitDataset(Subcommand):
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        description = "Split dataset into training, validation, and test set."
        subparser = parser.add_parser(
            self.name, description=description, help="Split dataset."
        )
        subparser.add_argument(
            "--input-file",
            type=str,
            help="Path to `yelp_academic_dataset_review.json`.",
        )
        subparser.add_argument(
            "--output-dir",
            type=pathlib.Path,
            default="data",
            help="Output path of splitted dataset.",
        )
        subparser.add_argument(
            "--tng-filename",
            type=str,
            default="tng.jsonl",
            help="Filename of training set.",
        )
        subparser.add_argument(
            "--val-filename",
            type=str,
            default="val.jsonl",
            help="Filename of validation set.",
        )
        subparser.add_argument(
            "--tst-filename",
            type=str,
            default="tst.jsonl",
            help="Filename of test set.",
        )
        subparser.add_argument(
            "--tng-ratio",
            type=float,
            default=0.8,
            help="Ratio of the `train set` when it is divided into subsets",
        )
        subparser.add_argument(
            "--val-ratio",
            type=float,
            default=0.1,
            help="Ratio of the `validation set` when it is divided into subsets",
        )
        subparser.add_argument(
            "--tst-ratio",
            type=float,
            default=0.1,
            help="Ratio of the `test set` when it is divided into subsets",
        )
        subparser.set_defaults(func=split_dataset)

        return subparser
