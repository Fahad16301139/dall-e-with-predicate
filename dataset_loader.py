import os
import requests
import random
from typing import List, Optional
from pathlib import Path
import argparse

# URLs for datasets
ABC_6K_URL = "https://raw.githubusercontent.com/weixi-feng/Structured-Diffusion-Guidance/master/ABC-6K.txt"
CC_500_URL = "https://raw.githubusercontent.com/weixi-feng/Structured-Diffusion-Guidance/master/CC-500.txt"

def download_dataset(url: str, save_path: str) -> bool:
    """Download a dataset file from the specified URL
    
    Args:
        url: URL to download from
        save_path: Path to save the dataset file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Dataset successfully downloaded and saved to {save_path}")
        return True
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a text file
    
    Args:
        file_path: Path to the file containing prompts (one per line)
        
    Returns:
        List of prompts
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    
    except Exception as e:
        print(f"Error loading prompts from file: {e}")
        return []

def get_random_prompts(prompts: List[str], n: int = 1) -> List[str]:
    """Get n random prompts from the list
    
    Args:
        prompts: List of prompts to sample from
        n: Number of prompts to return
        
    Returns:
        List of n randomly selected prompts
    """
    if not prompts:
        return []
    
    # Ensure n is not larger than the number of available prompts
    n = min(n, len(prompts))
    
    return random.sample(prompts, n)

def analyze_prompt_for_predicates(prompt: str) -> tuple:
    """Analyze a prompt to identify potential predicated diffusion constraints
    
    This is a simplified heuristic to extract potential predicate logic constraints
    from the prompt text. A more sophisticated approach would use NLP tools.
    
    Args:
        prompt: The text prompt to analyze
        
    Returns:
        Tuple of (corr_indices, exist_indices, possession_indices)
    """
    words = prompt.lower().split()
    
    # Simple heuristic to find color-object pairs
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'brown', 'pink', 'gray', 'grey']
    
    corr_pairs = []
    exist_indices = []
    possession_indices = []
    
    # Find color-object correlations
    for i, word in enumerate(words):
        if word in colors and i < len(words) - 1:
            # This is a simple heuristic assuming color followed by object
            corr_pairs.append([i, i+1])
            exist_indices.append(i+1)  # The object should exist
    
    # Find possession relationships (very simplified)
    for i, word in enumerate(words):
        if word in ['with', 'wearing', 'holding'] and i > 0 and i < len(words) - 1:
            possession_indices.append([i-1, i+1])
    
    return corr_pairs, exist_indices, possession_indices

def main():
    parser = argparse.ArgumentParser(description="Download and process datasets for predicated diffusion")
    parser.add_argument("--dataset", type=str, default="ABC-6K", choices=["ABC-6K", "CC-500"], 
                      help="Dataset to download and process")
    parser.add_argument("--output_dir", type=str, default="datasets", 
                      help="Directory to save the dataset")
    parser.add_argument("--sample", type=int, default=0, 
                      help="Number of random prompts to sample (0 means all)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Select URL based on dataset choice
    url = ABC_6K_URL if args.dataset == "ABC-6K" else CC_500_URL
    save_path = output_dir / f"{args.dataset}.txt"
    
    # Download the dataset
    if not save_path.exists():
        download_dataset(url, str(save_path))
    else:
        print(f"Dataset already exists at {save_path}")
    
    # Load prompts
    prompts = load_prompts_from_file(str(save_path))
    print(f"Loaded {len(prompts)} prompts from {save_path}")
    
    # Sample prompts if requested
    if args.sample > 0:
        sampled_prompts = get_random_prompts(prompts, args.sample)
        print(f"Sampled {len(sampled_prompts)} prompts randomly")
        
        # Print sample prompts with predicate analysis
        for i, prompt in enumerate(sampled_prompts):
            corr_pairs, exist_indices, possession_indices = analyze_prompt_for_predicates(prompt)
            print(f"\nPrompt {i+1}: {prompt}")
            if corr_pairs:
                print(f"  Correlation pairs: {corr_pairs}")
            if exist_indices:
                print(f"  Existence indices: {exist_indices}")
            if possession_indices:
                print(f"  Possession indices: {possession_indices}")
    
if __name__ == "__main__":
    main() 