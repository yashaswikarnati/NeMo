import time
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.modifiers import DocumentModifier
from nemo_curator import Sequential
from nemo_curator.modules.modify import Modify
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.filters import DocumentFilter
from nemo_curator.modules import ExactDuplicates
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.filters import WordCountFilter,RepeatingTopNGramsFilter
from nemo_curator.utils.config_utils import build_filter_pipeline
from nemo_curator import Score, Filter, ScoreFilter
from nemo_curator import ScoreFilter
import re
from nemo_curator import AddId
from dask.distributed import Client, LocalCluster
import os
import argparse
import warnings
import glob
warnings.filterwarnings("ignore",module="dask.dataframe.core")
from nemo_curator.utils.script_utils import ArgumentHelper
def pre_imports():
    import cudf  # noqa: F401

class TextFilter(DocumentModifier):
    def modify_document(self, text: str) -> str:
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("\n", " ")
        text = text.replace("§", "")
        text = re.sub(r'\b(?<![\d-])(\d{1,2})(?![\d-])\b', '', text) #numbered bullet points but dont remove useful numbers
        text = re.sub(r'[^\w\s\-\./:,\'"]', '', text) #back slashes
        text = re.sub(r'\s+', ' ', text).strip()
        return text
def clean_and_unify(dataset: DocumentDataset) -> DocumentDataset:
    cleaners = Sequential(
        [
            Modify(TextFilter()),
            Modify(UnicodeReformatter()),
        ]
    )
    return cleaners(dataset)

def redact_pii(dataset: DocumentDataset) -> DocumentDataset:
    redactor = Modify(
        PiiModifier(
            supported_entities=["EMAIL_ADDRESS"],
            anonymize_action="replace",
            device="cpu",
        ),
    )
    return redactor(dataset)

def main(args):
    # cluster = LocalCluster(n_workers=5, processes=True, memory_limit='16GB')
    # client = Client(cluster)
    print(args)
    client = get_client(**ArgumentHelper.parse_client_args(args))
    backend = "cudf" if args.device == "gpu" else "pandas"

    if args.device == "gpu":
        client.run(pre_imports)
    data_dir = '/home/ykarnati/Downloads/tempdaata/legal-mc4/train/'
    output_dir = '/home/ykarnati/Downloads/tempdaata/legal-mc4/train/new_process'
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    all_files = glob.glob(f"{data_dir}/*.jsonl")
    print(f"all_+files {all_files}")
    input_dataset = DocumentDataset.read_json(input_files=all_files,add_filename=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filter_config_file = os.path.join(current_dir, 'config.yaml')
    filter_pipeline = build_filter_pipeline(filter_config_file)     
    curation_steps = Sequential(
        [
            clean_and_unify,
            redact_pii,
            filter_pipeline
        ]
    )
    input_dataset = curation_steps(input_dataset)

    input_dataset = input_dataset.persist()
    print(f"output_dir{output_dir}")
    input_dataset.to_json(output_dir, write_to_filename=True)

def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    return ArgumentHelper(parser).add_distributed_args()

if __name__ == "__main__":
   main(attach_args().parse_args())
    
    
   