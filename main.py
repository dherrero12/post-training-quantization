# Import required packages
import sys
sys.path.insert(1, 'utils')
from parser import parse_arguments
from dataset import load_data
from model import load_models
from quantizer import quantize_models
from processing import preprocess_data
from evaluation import evaluate_models

# Define GLUE tasks
task_to_keys = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2')}

# Fine-tune pretrained model 
def main():
    # Parse data and model arguments
    model_args, data_args, training_args = parse_arguments()

    # Load evaluation datasets
    raw_datasets, label_list, num_labels = load_data(
            model_args, data_args, task_to_keys)

    # Load classification models
    models = load_models(model_args, data_args, num_labels, task_to_keys)

    # Quantize classification models
    quantized_models, quantized_sizes, model_sizes = quantize_models(models, model_args)

    # Preprocess evaluation datasets
    eval_dataset, raw_datasets, tokenizer = preprocess_data(
            quantized_models, raw_datasets,
            label_list, num_labels, model_args, data_args, training_args,
            task_to_keys)

    # Evaluate quantized models performance
    evaluate_models(
            quantized_models, quantized_sizes, model_sizes,
            eval_dataset, raw_datasets, tokenizer, 
            model_args, data_args, training_args)

# Evaluate models on GLUE tasks
if __name__ == '__main__':
    main()
