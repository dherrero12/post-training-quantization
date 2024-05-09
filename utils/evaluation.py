from transformers import (
    default_data_collator,
    Trainer,
    EvalPrediction)
import numpy as np
import evaluate
import os
import torch
import time

# Evaluate models performance 
def evaluate_models(
        models, quantized_sizes, model_sizes,
        eval_dataset, raw_datasets, tokenizer, 
        model_args, data_args, training_args):
    # Evaluate models performance 
    for task_name in models.keys(): 
        # Define model trainer / evaluator
        model = models[task_name]
        eval_dataset_task = eval_dataset[task_name]
        is_regression = task_name == 'stsb'
        metric = evaluate.load('glue', task_name, 
                cache_dir=model_args.cache_dir)
        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=None,
                eval_dataset=eval_dataset_task,
                compute_metrics=lambda p: compute_metrics(
                    p, is_regression, metric),
                tokenizer=tokenizer,
                data_collator=default_data_collator)

        # Handle MNLI double evaluation
        tasks = [task_name]
        eval_datasets = [eval_dataset_task]
        if task_name == 'mnli':
            tasks.append('mnli-mm')
            valid_mm_dataset = raw_datasets[task_name]['validation_mismatched']
            if data_args.max_eval_samples is not None: 
                max_eval_samples = min(
                        len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(
                        range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        # Evaluate model on dataset
        for eval_dataset_task, task in zip(eval_datasets, tasks):
            started = time.time()
            metrics = trainer.evaluate(eval_dataset=eval_dataset_task)
            eval_time = time.time() - started
            metrics['eval_time'] = eval_time 
            metrics['quant_size'] = quantized_sizes[task_name]
            metrics['model_size'] = model_sizes[task_name]
            max_eval_samples = (
                    data_args.max_eval_samples 
                    if data_args.max_eval_samples is not None
                    else len(eval_dataset))
            metrics['eval_samples'] = min(
                    max_eval_samples, len(eval_dataset_task))
            if task == 'mnli-mm':
                metrics = {k + '_mm': v for k, v in metrics.items()}
            if 'mnli' in task:
                combined.update(metrics)
            split = f'eval_{task_name}' 
            if model_args.quantization is not None:
                split = f'{split}_{model_args.quantization}'
                if model_args.quantization in {'absmax', 'zeropoint'}:
                    split = f'{split}_{model_args.bits}'
                elif model_args.quantization == 'norm':
                    split = f'{split}_{model_args.quantile}'
                elif model_args.quantization == 'partial':
                    split = f'{split}_{model_args.quantile}-{model_args.bits}'
            else:
                split = f'{split}_none_none'
            trainer.log_metrics(split, metrics)
            trainer.save_metrics(
                    split, combined if 'mnli' in task else metrics, 
                    combined=False)

# Define compute metrics function    
def compute_metrics(p: EvalPrediction, is_regression, metric):
    preds = (p.predictions[0] if isinstance(p.predictions, tuple) 
        else p.predictions)
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result['combined_core'] = np.mean(list(result.values())).item()
    return result
