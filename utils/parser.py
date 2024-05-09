# Import required packages
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments

# Define data training arguments
@dataclass
class DataTrainingArguments:
    # Define data training arguments
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)
    max_eval_samples: Optional[int] = field(default=None)

# Define model arguments
@dataclass
class ModelArguments:
    # Define model arguments
    model_name_or_path: str = field()
    quantization: str = field(default=None)
    bits: int = field(default=None)
    quantile: float = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default='main')
    token: str = field(default=None)
    trust_remote_code: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)

# Parse data and model arguments
def parse_arguments():
    # Define argument parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # Parse arguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Return arguments
    return model_args, data_args, training_args
