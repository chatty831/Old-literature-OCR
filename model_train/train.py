#!/usr/bin/env python
"""
OCR Model Training Pipeline

This script implements a fine-tuning pipeline for vision-language models
for OCR tasks, particularly for Spanish text recognition.
"""

import os
import sys
from typing import Dict, List, Any, Tuple, Optional, Union

import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

# Local imports
from data_load import load_conversation_dataset, convert_paths_to_images
from config import (
    MODEL_CONFIG,
    LORA_CONFIG,
    DATASET_CONFIG,
    TRAINING_CONFIG,
    INFERENCE_CONFIG
)


def load_model(
    model_name: str,
    load_in_4bit: bool = True,
    use_gradient_checkpointing: bool = False
) -> Tuple[Any, Any]:
    """
    Load the vision-language model and tokenizer.
    
    Args:
        model_name: Pretrained model name or path
        load_in_4bit: Whether to load the model in 4-bit precision
        use_gradient_checkpointing: Whether to use gradient checkpointing
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    return model, tokenizer


def setup_peft_model(model: Any, lora_config: Dict[str, Any]) -> Any:
    """
    Set up the PEFT (Parameter-Efficient Fine-Tuning) model with LoRA.
    
    Args:
        model: The base model
        lora_config: Configuration for LoRA
        
    Returns:
        PEFT model
    """
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_config["finetune_vision_layers"],
        finetune_language_layers=lora_config["finetune_language_layers"],
        finetune_attention_modules=lora_config["finetune_attention_modules"],
        finetune_mlp_modules=lora_config["finetune_mlp_modules"],
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        random_state=lora_config["random_state"],
        use_rslora=lora_config["use_rslora"],
        loftq_config=lora_config["loftq_config"],
    )
    return model


def prepare_datasets(dataset_config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare training and evaluation datasets.
    
    Args:
        dataset_config: Configuration for dataset preparation
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Load the conversation dataset
    dataset_dict = load_conversation_dataset(
        json_path=dataset_config["json_path"],
        image_root_dir=dataset_config["image_root_dir"],
        split=dataset_config["split"]
    )
    
    # Convert paths to actual PIL Images
    dataset = convert_paths_to_images(
        dataset_dict, 
        image_root_dir=dataset_config["image_root_dir"]
    )
    
    # Split into train and eval datasets
    train_dataset, eval_dataset = train_test_split(
        dataset, 
        test_size=dataset_config["test_size"], 
        random_state=dataset_config["random_state"]
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def run_inference_sample(
    model: Any, 
    tokenizer: Any, 
    dataset: List[Dict], 
    inference_config: Dict[str, Any]
) -> None:
    """
    Run inference on a sample from the dataset.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        dataset: The dataset containing samples
        inference_config: Configuration for inference
    """
    # Set model to inference mode
    FastVisionModel.for_inference(model)
    
    # Get first sample
    image = dataset[0]['messages'][0]['content'][1]['image']
    instruction = "Perform OCR on this, and give the Spanish text written on this with modern alphabets."
    
    # Prepare input
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    # Format input for model
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    # Run inference with streaming
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs, 
        streamer=text_streamer, 
        max_new_tokens=inference_config["max_new_tokens"],
        use_cache=inference_config["use_cache"], 
        temperature=inference_config["temperature"], 
        min_p=inference_config["min_p"]
    )


def setup_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: List[Dict],
    eval_dataset: List[Dict],
    training_config: Dict[str, Any]
) -> SFTTrainer:
    """
    Set up the SFT trainer.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_config: Training configuration
        
    Returns:
        SFTTrainer instance
    """
    # Set model to training mode
    FastVisionModel.for_training(model)
    
    # Create SFT configuration
    sft_config = SFTConfig(
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        warmup_steps=training_config["warmup_steps"],
        eval_steps=training_config["eval_steps"],
        eval_strategy="steps",
        num_train_epochs=training_config["num_train_epochs"],
        learning_rate=training_config["learning_rate"],
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=training_config["logging_steps"],
        optim=training_config["optim"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        seed=training_config["seed"],
        output_dir=training_config["output_dir"],
        report_to=training_config["report_to"],
        # Vision-specific settings
        remove_unused_columns=training_config["remove_unused_columns"],
        dataset_text_field=training_config["dataset_text_field"],
        dataset_kwargs=training_config["dataset_kwargs"],
        dataset_num_proc=training_config["dataset_num_proc"],
        max_seq_length=training_config["max_seq_length"],
        save_strategy=training_config["save_strategy"],
        save_total_limit=training_config["save_total_limit"],
        save_steps=training_config["save_steps"],
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )
    
    return trainer


def train_model(
    trainer: SFTTrainer, 
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train the model.
    
    Args:
        trainer: SFTTrainer instance
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Training statistics
    """
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer_stats


def save_model(
    model: Any, 
    tokenizer: Any, 
    save_path: str
) -> None:
    """
    Save the model and tokenizer.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        save_path: Path to save to
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


def main():
    """Main entry point for the training pipeline."""
    # Load model and tokenizer
    model, tokenizer = load_model(
        MODEL_CONFIG["model_name"],
        MODEL_CONFIG["load_in_4bit"],
        MODEL_CONFIG["use_gradient_checkpointing"]
    )
    
    # Set up PEFT model
    model = setup_peft_model(model, LORA_CONFIG)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(DATASET_CONFIG)
    
    # Optional: Run inference on a sample
    run_inference_sample(model, tokenizer, train_dataset, INFERENCE_CONFIG)
    
    # Set up trainer
    trainer = setup_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        TRAINING_CONFIG
    )
    
    # Train model
    trainer_stats = train_model(
        trainer, 
        resume_from_checkpoint=TRAINING_CONFIG.get("resume_from_checkpoint", None)
    )
    
    # Save model
    save_model(model, tokenizer, MODEL_CONFIG["saved_model_path"])
    
    return trainer_stats


if __name__ == "__main__":
    main()