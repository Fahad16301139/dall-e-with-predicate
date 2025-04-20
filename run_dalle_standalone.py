"""
Standalone script for running the DALL-E integration without requiring the Stable Diffusion dependencies.
This script provides a simplified interface for using DALL-E with predicate logic guidance.
"""

import os
import argparse
from pathlib import Path
import torch
from PIL import Image
import time
import json
import base64
from io import BytesIO
import requests

# Initialize OpenAI client
def init_openai_client(api_key=None):
    """Initialize the OpenAI client with the provided API key or from environment variable"""
    try:
        from openai import OpenAI
        
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("No API key provided and OPENAI_API_KEY environment variable not set")
        
        return OpenAI(api_key=api_key)
    
    except ImportError:
        print("OpenAI package not found. Please install it with 'pip install openai'")
        raise

def enhance_prompt_with_predicates(prompt, corr_indices=None, leak_indices=None, exist_indices=None, possession_indices=None):
    """Enhance the prompt with predicate logic to improve DALL-E generation
    
    This translates the predicated diffusion model's logic constraints into natural language
    that DALL-E can understand better.
    """
    enhanced_parts = [prompt]
    
    # Add correlation constraints
    if corr_indices:
        for pair in corr_indices:
            if len(pair) == 2:
                # Extract the terms from the prompt tokens
                # This is a simplified version - in practice you'd need to map indices to actual tokens
                enhanced_parts.append(f"Make sure the attributes correctly correspond to their objects")
    
    # Add existence constraints
    if exist_indices:
        enhanced_parts.append(f"Make sure all objects are clearly visible in the image")
    
    # Add possession constraints
    if possession_indices:
        for pair in possession_indices:
            if len(pair) == 2:
                enhanced_parts.append(f"The objects should have a clear ownership relationship")
    
    # Create final enhanced prompt
    enhanced_prompt = prompt + ". " + " ".join(enhanced_parts[1:])
    return enhanced_prompt

def generate_image_with_dalle(client, prompt, dalle_version="dall-e-3", 
                             corr_indices=None, leak_indices=None, 
                             exist_indices=None, possession_indices=None):
    """Generate an image using DALL-E with the given prompt and predicate constraints"""
    # Configure DALL-E parameters
    params = {
        "model": dalle_version,
        "prompt": prompt,
        "n": 1,  # Number of images to generate
        "size": "1024x1024",  # Default size
    }
    
    # Apply quality parameter for DALL-E 3
    if dalle_version == "dall-e-3":
        params["quality"] = "standard"  # Could be "standard" or "hd"
        params["style"] = "vivid"  # Could be "vivid" or "natural"
    
    # Include predicate logic from predicated diffusion in the prompt
    enhanced_prompt = enhance_prompt_with_predicates(
        prompt, 
        corr_indices,
        leak_indices, 
        exist_indices,
        possession_indices
    )
    
    params["prompt"] = enhanced_prompt
    print(f"Enhanced prompt: {enhanced_prompt}")
    
    try:
        # Generate image using DALL-E API
        response = client.images.generate(**params)
        
        # Handle the response
        url = response.data[0].url
        print(f"Image generated at URL: {url}")
        
        # Download the image
        image_data = requests.get(url).content
        image = Image.open(BytesIO(image_data))
        
        return image, url
        
    except Exception as e:
        print(f"Error generating image with DALL-E: {e}")
        # Return a blank image in case of error
        return Image.new('RGB', (1024, 1024), color='white'), None

def analyze_prompt_for_predicates(prompt):
    """Analyze a prompt to identify potential predicated diffusion constraints
    
    This is a simplified heuristic to extract potential predicate logic constraints
    from the prompt text. A more sophisticated approach would use NLP tools.
    """
    words = prompt.lower().split()
    
    # Simple heuristic to find color-object pairs
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 
              'orange', 'brown', 'pink', 'gray', 'grey']
    
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
    parser = argparse.ArgumentParser(description="Generate images with DALL-E using predicate logic guidance")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="Text prompt for image generation")
    parser.add_argument("--dalle_version", type=str, default="dall-e-3", choices=["dall-e-2", "dall-e-3"],
                       help="Version of DALL-E to use")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key. If not provided, will try to use environment variable OPENAI_API_KEY")
    parser.add_argument("--output_dir", type=str, default="outputs/dalle",
                       help="Directory to save the generated image")
    parser.add_argument("--auto_analyze", action="store_true",
                       help="Automatically analyze prompt for predicate constraints")
    parser.add_argument("--corr_indices", type=str, default=None,
                       help="Correlation indices as string representation of list of pairs, e.g. '[[1,2],[4,5]]'")
    parser.add_argument("--exist_indices", type=str, default=None,
                       help="Existence indices as string representation of list, e.g. '[2,5]'")
    parser.add_argument("--possession_indices", type=str, default=None,
                       help="Possession indices as string representation of list of pairs, e.g. '[[2,5]]'")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize OpenAI client
    client = init_openai_client(args.api_key)
    
    # Parse constraint indices if provided
    corr_indices = eval(args.corr_indices) if args.corr_indices else []
    exist_indices = eval(args.exist_indices) if args.exist_indices else []
    possession_indices = eval(args.possession_indices) if args.possession_indices else []
    
    # Auto analyze prompt if requested
    if args.auto_analyze:
        print("Automatically analyzing prompt for predicate constraints...")
        corr_pairs, exist_idx, poss_pairs = analyze_prompt_for_predicates(args.prompt)
        
        if corr_pairs:
            print(f"Found correlation pairs: {corr_pairs}")
            corr_indices = corr_pairs
            
        if exist_idx:
            print(f"Found existence indices: {exist_idx}")
            exist_indices = exist_idx
            
        if poss_pairs:
            print(f"Found possession pairs: {poss_pairs}")
            possession_indices = poss_pairs
    
    # Generate image
    print(f"Generating image for prompt: {args.prompt}")
    print(f"Using predicate constraints:")
    print(f"  - Correlation indices: {corr_indices}")
    print(f"  - Existence indices: {exist_indices}")
    print(f"  - Possession indices: {possession_indices}")
    
    image, url = generate_image_with_dalle(
        client=client,
        prompt=args.prompt,
        dalle_version=args.dalle_version,
        corr_indices=corr_indices,
        exist_indices=exist_indices,
        possession_indices=possession_indices
    )
    
    # Save image
    timestamp = int(time.time())
    filename = f"{timestamp}_{args.prompt.replace(' ', '_')[:50]}.png"
    image_path = output_dir / filename
    image.save(image_path)
    print(f"Image saved to {image_path}")
    
    # Save metadata
    metadata = {
        "prompt": args.prompt,
        "timestamp": timestamp,
        "dalle_version": args.dalle_version,
        "corr_indices": corr_indices,
        "exist_indices": exist_indices,
        "possession_indices": possession_indices,
        "image_url": url,
        "image_path": str(image_path)
    }
    
    metadata_path = output_dir / f"{timestamp}_{args.prompt.replace(' ', '_')[:50]}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main() 