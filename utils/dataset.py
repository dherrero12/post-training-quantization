# Import required packages
from datasets import load_dataset

# Load evaluation datasets
def load_data(model_args, data_args, task_to_keys):
    # Get task names
    tasks = task_to_keys.keys()

    # Download and load datasets form HuggingFace 
    raw_datasets = {task_name: load_dataset(
            'nyu-mll/glue', 
            task_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token)
            for task_name in tasks}

    # Get dataset labels
    label_list = {} 
    num_labels = {} 
    for task_name in tasks:
        # Get number of labels
        is_regression = task_name == 'stsb'
        if not is_regression:
            label_list_task = raw_datasets[task_name]['train'].features['label'].names
            label_list[task_name] = label_list_task
            num_labels[task_name] = len(label_list_task)
        else:
            num_labels[task_name] = 1

    # Return evaluation datasets
    return raw_datasets, label_list, num_labels
