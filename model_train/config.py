"""
Configuration settings for the OCR model training pipeline.
"""

# Model configuration
MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "load_in_4bit": True,
    "use_gradient_checkpointing": False,
    "saved_model_path": "all_tuned_model"
}

# LoRA configuration
LORA_CONFIG = {
    "finetune_vision_layers": False,
    "finetune_language_layers": True,
    "finetune_attention_modules": True,
    "finetune_mlp_modules": True,
    "r": 64,
    "lora_alpha": 64 * 1.5,
    "lora_dropout": 0,
    "bias": "none",
    "random_state": 3407,
    "use_rslora": False,
    "loftq_config": None,
}

# Dataset configuration
DATASET_CONFIG = {
    "json_path": "/raid/niloycs/workspace_chait/workspace/ocr_dataset_spanish_2/dataset_new.json",
    "image_root_dir": "/raid/niloycs/workspace_chait/workspace/",
    "split": "train",
    "test_size": 0.05,
    "random_state": 42
}

# Training configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 5,
    "eval_steps": 20,
    "num_train_epochs": 1,
    "learning_rate": 1e-5,
    "logging_steps": 1,
    "optim": "paged_adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "outputs",
    "report_to": "wandb", # Change it to none if necessary
    "remove_unused_columns": False,
    "dataset_text_field": "",
    "dataset_kwargs": {"skip_prepare_dataset": True},
    "dataset_num_proc": 2,
    "max_seq_length": 2048,
    "save_strategy": "steps",
    "save_total_limit": 3,
    "save_steps": 20,
}

# Inference configuration
INFERENCE_CONFIG = {
    "max_new_tokens": 128,
    "use_cache": True,
    "temperature": 0.7,
    "min_p": 0.9
}