#!/bin/bash

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set your OpenAI API key with: export OPENAI_API_KEY=your_api_key"
    exit 1
fi

# Create necessary directories
mkdir -p datasets
mkdir -p outputs/dalle

# Make sure dependencies are installed
echo "Installing dependencies..."
pip install -r requirements_dalle.txt

# Function to run a single prompt with DALL-E
run_single_prompt() {
    local prompt="$1"
    if [ -z "$prompt" ]; then
        echo "No prompt provided."
        return 1
    fi
    
    echo "Running DALL-E with prompt: $prompt"
    python run.py --text_input "$prompt" --output_dir "outputs/dalle" --guidance_scale 7.5 --seed 42 --model_key "dalle"
    return 0
}

# Function to run a single prompt with standalone DALL-E script (no Stable Diffusion dependencies)
run_standalone() {
    local prompt="$1"
    if [ -z "$prompt" ]; then
        echo "No prompt provided."
        return 1
    fi
    
    echo "Running standalone DALL-E with prompt: $prompt"
    python run_dalle_standalone.py --prompt "$prompt" --output_dir "outputs/dalle" --auto_analyze
    return 0
}

# Parse command-line arguments
if [ "$1" = "--download-dataset" ]; then
    echo "Downloading ABC-6K dataset..."
    python dataset_loader.py
    exit 0
fi

if [ "$1" = "--run-dataset" ]; then
    num_samples=5
    if [ ! -z "$2" ]; then
        num_samples="$2"
    fi
    echo "Running DALL-E on $num_samples samples from ABC-6K dataset..."
    python run_dalle_abc6k.py --num_samples "$num_samples"
    exit 0
fi

if [ "$1" = "--standalone" ]; then
    if [ -z "$2" ]; then
        echo "Please provide a prompt after --standalone"
        exit 1
    fi
    run_standalone "$2"
    exit 0
fi

# If no specific args provided, run examples
if [ -z "$1" ]; then
    echo "Running example prompts with DALL-E..."
    
    echo "Example 1: Simple color-object binding"
    run_single_prompt "A red car and a blue boat"
    
    echo "Example 2: Object possession relationship"
    run_single_prompt "A frog wearing a hat"
    
    echo "Example 3: Complex scene with multiple constraints"
    run_single_prompt "A brown dog and a yellow bowl"
    
    echo "All examples completed. Check results in outputs/dalle directory."
else
    # If a prompt is provided, run it
    run_single_prompt "$1"
fi 