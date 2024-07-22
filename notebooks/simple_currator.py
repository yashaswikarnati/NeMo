import argparse
import os
import shutil
from typing import Any



from nemo_curator.modifiers import DocumentModifier

from nemo_curator import ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import RepeatingTopNGramsFilter, WordCountFilter
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.modules import ExactDuplicates
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

class QuotationUnifier(DocumentModifier):
    """
    A simple modifier that unifies the quotation marks in the documents.
    """

    def modify_document(self, text: str) -> str:
        """
        Modifies the given text by replacing left and right single quotes with normal single quotes,
        and replacing left and right double quotes with normal double quotes.

        Args:
            text (str): The text to be modified.

        Returns:
            str: The modified text.
        """
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        return text

def clean_and_unify(dataset: DocumentDataset) -> DocumentDataset:
    """
    Cleans and unifies the given dataset using a set of predefined cleaners.

    Args:
        dataset (DocumentDataset): The dataset to be cleaned and unified.

    Returns:
        DocumentDataset: The cleaned and unified dataset.
    """
    cleaners = Sequential(
        [
            # Unify all the quotation marks
            Modify(QuotationUnifier()),
            # Unify all unicode
            Modify(UnicodeReformatter()),
        ]
    )
    return cleaners(dataset)

def filter_dataset(dataset: DocumentDataset) -> DocumentDataset:
    """
    Filters the given dataset based on various criteria.

    Args:
        dataset (DocumentDataset): The dataset to be filtered.

    Returns:
        DocumentDataset: The filtered dataset.
    """
    filters = Sequential(
        [
            ScoreFilter(
                WordCountFilter(min_words=80),
                text_field="text",
                score_field="word_count",
                score_type=int,
            ),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2),
                text_field="text",
                score_type=float,
            ),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
                text_field="text",
                score_type=float,
            ),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16),
                text_field="text",
                score_type=float,
            ),
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset

def run_curation_pipeline(args: Any, jsonl_dir: str) -> None:
    """
    Run the curation pipeline on the TinyStories dataset.

    Args:
        args (Any): Command-line arguments.
        jsonl_dir (str): Directory path where the JSONL files are stored.
    """
    # Initialize the Dask cluster.
    client = get_client(**ArgumentHelper.parse_client_args(args))
    print(f"Running curation pipeline on '{jsonl_dir}'...")
    files = [
        fp
        for fp in get_all_files_paths_under(jsonl_dir, recurse_subdirectories=False)
        if fp.endswith(".jsonl")
    ]
    print("Reading the data...")
    orig_dataset = DocumentDataset.read_json(files, add_filename=True)
    dataset = orig_dataset

    curation_steps = Sequential(
        [
            clean_and_unify,
            filter_dataset,
        ]
    )
    dataset = curation_steps(dataset)
    print("Executing the pipeline...")
    print(f"Original dataset length: {len(orig_dataset.df)}")
    print(f"After dataprep: {len(dataset.df)}")
    print("Writing the results to disk...")

    # Overwrite existing files in the curated directory.
    out_path = os.path.join(jsonl_dir, "curated")

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)
    dataset.to_json(out_path, write_to_filename=True)
    client.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory path where the JSONL files are stored')
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)    

    run_curation_pipeline(args, args.input_dir)


if __name__ == "__main__":
    main()