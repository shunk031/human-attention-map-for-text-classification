import argparse
import pathlib

from allennlp.commands.subcommand import Subcommand
from overrides import overrides

from .function import preprocess_ham_dataset


@Subcommand.register("preprocess-ham-dataset")
class PreprocessHamDataset(Subcommand):
    @overrides
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Run the specified model against a JSON-lines input file."""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Use a trained model to make predictions.",
        )
        subparser.add_argument(
            "--ham-dataset-dir",
            type=pathlib.Path,
            default=pathlib.Path(__file__).absolute().parents[3]
            / "data"
            / "ham-dataset"
            / "raw_data",
            help="Path to the HAM dataset directory",
        )
        subparser.add_argument(
            "--output-dir",
            type=pathlib.Path,
            default=pathlib.Path(__file__).absolute().parents[3] / "data",
        )
        subparser.set_defaults(func=preprocess_ham_dataset)
        return subparser
