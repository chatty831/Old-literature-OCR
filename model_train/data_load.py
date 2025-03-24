import json
import os
from typing import Optional, Dict, List
from PIL import Image
from tqdm import tqdm

def load_conversation_dataset(
    json_path: str,
    image_root_dir: Optional[str] = None,
    split: Optional[str] = None,
    instruction: str = "Perform OCR on this, and give the Spanish text written on this with modern alphabets."
) -> Dict[str, List[Dict]]:
    """
    Load a dataset from JSON and convert to conversation format for VLM training.
    
    Args:
        json_path: Path to the dataset JSON file.
        image_root_dir: Root directory for image paths (if paths in JSON are relative). 
                        Defaults to the directory containing the JSON file.
        split: Dataset split to filter by (e.g., 'train', 'test') or None for all data.
        instruction: The instruction text to include with each image.
        
    Returns:
        Dict[str, List[Dict]]: A dictionary with dataset entries.
    """
    # If image_root_dir is not provided, use the JSON file's directory
    if image_root_dir is None:
        image_root_dir = os.path.dirname(json_path)
    
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Extract samples based on structure and filter by split if needed
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    elif isinstance(data, list):
        samples = data
    else:
        samples = [data]  # Single item
    
    # Filter by split if specified
    if split:
        samples = [s for s in samples if s.get('split', 'train') == split]
    
    # Prepare dataset entries
    dataset_entries = []
    
    for sample in tqdm(samples, desc="Processing samples"):
        # Get image path and resolve it if it's relative
        image_path = sample.get('image_path', '')
        if image_root_dir and not os.path.isabs(image_path):
            image_path = os.path.join(image_root_dir, image_path)
            image_path = os.path.abspath(image_path)
        
        # Get transformed_text from the sample for OCR result
        transformed_text = sample.get("transformed_text", "")
        
        # Skip if image or transformed_text doesn't exist
        if not image_path or not os.path.exists(image_path) or not transformed_text:
            print(f"Warning: Missing image or transformed text for sample with image path {image_path}")
            continue
        
        # Create conversation structure
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image_path}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": transformed_text}
                ]
            }
        ]
        
        dataset_entries.append({"messages": conversation})
    
    return {"data": dataset_entries}

def process_image(content_item, image_root_dir: Optional[str] = None):
    """
    Helper function to process an image content item.
    """
    if content_item["type"] == "image":
        image_path = content_item["image"]
        image_path = os.path.join(image_root_dir, image_path)
        # image_path = os.path.abspath(image_path)
        if isinstance(image_path, str) and os.path.exists(image_path):
            # Open the image file
            img = Image.open(image_path)
            content_item["image"] = img
    
    # Remove keys with None values
    keys_to_pop = [k for k, v in content_item.items() if v is None]
    for key in keys_to_pop:
        content_item.pop(key)
    
    return content_item

def convert_paths_to_images(dataset_dict: Dict[str, List[Dict]], image_root_dir: Optional[str] = None) -> List[Dict]:
    """
    Optional function to load images into memory instead of keeping paths.
    Use this if you want the dataset to contain actual PIL Images instead of paths.
    Warning: This will increase memory usage significantly for large datasets.
    
    Args:
        dataset_dict: Dictionary with dataset entries containing image paths.
        image_root_dir: Root directory for image paths (if paths in JSON are relative).
        
    Returns:
        List[Dict]: List of dictionaries with loaded PIL Images.
    """
    # If image_root_dir is not provided, try to derive it from the first image path
    if image_root_dir is None:
        for example in dataset_dict.get("data", []):
            for message in example.get("messages", []):
                for content_item in message.get("content", []):
                    if content_item.get("type") == "image":
                        potential_path = content_item.get("image", "")
                        image_root_dir = os.path.dirname(potential_path)
                        break
                if image_root_dir:
                    break
            if image_root_dir:
                break
        if image_root_dir is None:
            image_root_dir = ""
    
    updated_entries = []
    
    for example in tqdm(dataset_dict["data"], desc="Converting paths to images"):
        messages = example["messages"]
        for message in messages:
            if message["role"] == "user":
                # Process each content item
                for i, content_item in enumerate(message["content"]):
                    message["content"][i] = process_image(content_item, image_root_dir)
        updated_entries.append(example)
    
    return updated_entries