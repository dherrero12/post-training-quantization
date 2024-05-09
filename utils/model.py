# Import required packages
from transformers import ( 
        Trainer,
        AutoModelForSequenceClassification,
        AutoConfig)

# Load classification models 
def load_models(model_args, data_args, num_labels, task_to_keys):
    # Define classification models
    models = {}
    for task_name in task_to_keys.keys():
        # Define model configuration
        model_task = f'{model_args.model_name_or_path}-{task_name}'
        config = AutoConfig.from_pretrained(
                model_task,
                num_labels=num_labels[task_name],
                finetuning_task=task_name, 
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code)

        # Define pretrained model 
        model = AutoModelForSequenceClassification.from_pretrained(
                model_task,
                from_tf=bool('.ckpt' in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes)
        models[task_name] = model
        
    # Retrun classification models 
    return models
