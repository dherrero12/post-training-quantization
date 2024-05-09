# Import required packages
from transformers import (
        AutoTokenizer,
        PretrainedConfig)

# Preprocess evaluation datasets
def preprocess_data(
        models, raw_datasets,
        label_list, num_labels, model_args, data_args, training_args,
        task_to_keys):

    # Generate evaluation datasets 
    eval_dataset = {}
    for task_name, (sentence1_key, sentence2_key) in task_to_keys.items():
        # Define tokenizer
        model_task = f'{model_args.model_name_or_path}-{task_name}'
        tokenizer = AutoTokenizer.from_pretrained(
                model_task,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code) 
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        # Define map from labels to IDs
        model = models[task_name]
        is_regression = task_name == 'stsb'
        num_labels_task = num_labels[task_name]
        label_to_id = None
        if (model.config.label2id != PretrainedConfig(
                num_labels=num_labels_task).label2id
            and not is_regression):
            label_name_to_id = {
                    k.lower(): v for k, v in model.config.label2id.items()}
            label_list_task = label_list[task_name]
            if sorted(label_name_to_id.keys()) == sorted(label_list_task):
                label_to_id = {i: int(label_name_to_id[label_list_task[i]])
                        for i in range(num_labels_task)}
            model.config.label2id = label_to_id 

            # Assign map to model
            if label_to_id is not None:
                model.config.label2id = label_to_id
                model.config.id2label = {id: label 
                        for label, id in label_to_id.items()}

        # Tokenize dataset
        with training_args.main_process_first(desc='Dataset map preprocessing'):
            raw_datasets[task_name] = raw_datasets[task_name].map(
                    lambda examples: preprocess_function(
                        tokenizer,
                        examples, sentence1_key, sentence2_key, max_seq_length),
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc='Running tokenizer on dataset')

        # Define evaluation datasets 
        eval_dataset[task_name] = raw_datasets[task_name][
            'validation_matched' if task_name == 'mnli' else 'validation']
        if data_args.max_eval_samples is not None:
            eval_dataset_task = eval_dataset[task_name]
            max_eval_samples = min(
                    len(eval_dataset_task), data_args.max_eval_samples)
            eval_dataset[task_name] = eval_dataset_task.select(
                    range(max_eval_samples))

    # Return preprocessed evaluation datasets
    return eval_dataset, raw_datasets, tokenizer

# Define preprocess function
def preprocess_function(
        tokenizer,
        examples, sentence1_key, sentence2_key, max_seq_length):
    # Tokenize texts
    args = (
            (examples[sentence1_key],) if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key]))
    result = tokenizer(*args, max_length=max_seq_length, 
        padding='max_length', truncation=True)

    # Return tokenized texts
    return result
