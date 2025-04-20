import os
import argparse
from pathlib import Path
import pyrallis
import torch
from PIL import Image
import time
import random

from config import RunConfig
from run import load_model, run_on_prompt
from utils.ptp_utils import AttentionStore
from dataset_loader import load_prompts_from_file, analyze_prompt_for_predicates

def create_config_from_args(args):
    """Create a RunConfig from command line arguments"""
    config = RunConfig(
        prompt="placeholder",  # Will be replaced for each prompt
        use_dalle=True,
        dalle_version=args.dalle_version,
        openai_api_key=args.api_key,
        seeds=[args.seed],
        output_path=Path(args.output_dir),
        run_attention_sd=True,
        dataset_path=args.dataset_path
    )
    return config

def run_dalle_with_predicates(model, config, prompts, num_samples, start_idx=0):
    """Run predicated diffusion with DALL-E on multiple prompts
    
    Args:
        model: The DALL-E model
        config: The RunConfig
        prompts: List of prompts to process
        num_samples: Number of prompts to process (0 means all)
        start_idx: Index to start from
    """
    # Determine how many prompts to process
    if num_samples <= 0 or num_samples > len(prompts) - start_idx:
        num_samples = len(prompts) - start_idx
    
    # Create output directory
    output_path = config.output_path / 'dalle' / f'dalle-{config.dalle_version}'
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Process each prompt
    for i in range(start_idx, start_idx + num_samples):
        prompt = prompts[i]
        print(f"\n[{i+1}/{start_idx + num_samples}] Processing prompt: {prompt}")
        
        # Analyze the prompt for predicate constraints
        corr_pairs, exist_indices, possession_pairs = analyze_prompt_for_predicates(prompt)
        
        # Update config with the current prompt and predicate constraints
        config.prompt = prompt
        config.attention_corr_indices = corr_pairs
        config.attention_exist_indices = exist_indices
        config.attention_possession_indices = possession_pairs
        
        # Generate the image
        try:
            g = torch.Generator('cuda').manual_seed(config.seeds[0]) if torch.cuda.is_available() else None
            controller = AttentionStore()
            
            # Get token indices for attention
            token_indices = []
            if exist_indices:
                token_indices.extend(exist_indices)
            for pair in corr_pairs:
                token_indices.extend(pair)
            for pair in possession_pairs:
                token_indices.extend(pair)
            
            # Ensure we have unique indices
            token_indices = list(set(token_indices))
            if not token_indices:
                token_indices = [1, 2, 3, 4]  # Default if no constraints found
            
            config.token_indices = token_indices
            
            # Run the model
            image, loss_values, attention_maps = run_on_prompt(
                prompt=prompt,
                model=model,
                controller=controller,
                token_indices=token_indices,
                seed=g,
                config=config
            )
            
            # Save the generated image
            prompt_slug = prompt.replace(" ", "_")[:50]  # Create a slug for the filename
            image_path = output_path / f'sample_{i:04d}_{prompt_slug}.png'
            image.save(image_path)
            print(f"Saved image to {image_path}")
            
            # Sleep to avoid rate limiting if using DALL-E API
            time.sleep(2)
            
        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            # Sleep longer on error to mitigate potential rate limiting issues
            time.sleep(10)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run predicated diffusion with DALL-E on ABC-6K dataset")
    parser.add_argument("--dataset_path", type=str, default="datasets/ABC-6K.txt", 
                      help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                      help="Directory to save the generated images")
    parser.add_argument("--dalle_version", type=str, default="dall-e-3", choices=["dall-e-2", "dall-e-3"],
                      help="Version of DALL-E to use")
    parser.add_argument("--api_key", type=str, default=None,
                      help="OpenAI API key. If not provided, will try to use environment variable OPENAI_API_KEY")
    parser.add_argument("--num_samples", type=int, default=10,
                      help="Number of samples to generate (0 means all)")
    parser.add_argument("--start_idx", type=int, default=0,
                      help="Index to start processing from")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for generation")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Dataset file {args.dataset_path} not found!")
        print("You can download it using dataset_loader.py first")
        return
    
    # Load prompts from dataset
    prompts = load_prompts_from_file(args.dataset_path)
    if not prompts:
        print(f"No prompts found in dataset file {args.dataset_path}")
        return
    
    print(f"Loaded {len(prompts)} prompts from {args.dataset_path}")
    
    # Create config and load model
    config = create_config_from_args(args)
    model = load_model(config)
    
    # Run the model on the dataset
    run_dalle_with_predicates(model, config, prompts, args.num_samples, args.start_idx)

if __name__ == "__main__":
    main() 