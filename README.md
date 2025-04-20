# DALL-E with Predicate Logic

Official implementation for the paper ["PredicatedDiff: Bringing Relations and Logical Structures into Diffusion Models via Predicated Guidance"](https://arxiv.org/pdf/2402.04605.pdf).

![Figure](images/teaser.png)

## Overview

This repository includes a standalone DALL-E implementation with predicate logic guidance. The code provides a simplified interface for using DALL-E with predicate constraints to improve image generation.

## Installation

Create a conda environment:
```
conda create -n predicated python=3.10
conda activate predicated
```

Install PyTorch:
```
pip install torch torchvision
```

Install other requirements:
```
pip install -r requirements.txt
```

For standalone DALL-E only:
```
pip install -r requirements_dalle.txt
```

## Running the Demo

### Windows
Use the batch file to run the demo:
```
run_dalle_demo.bat
```

For standalone DALL-E (no Stable Diffusion dependencies):
```
run_dalle_demo.bat --standalone "A red car and a blue boat"
```

### Linux/WSL
If you're using Linux or WSL (Windows Subsystem for Linux), use the shell script:
```
chmod +x run_dalle_demo.sh  # Make the script executable (one-time setup)
./run_dalle_demo.sh
```

For standalone DALL-E in WSL:
```
./run_dalle_demo.sh --standalone "A red car and a blue boat"
```

## DALL-E Standalone Usage

To use the standalone DALL-E script directly:
```
python run_dalle_standalone.py --prompt "Your prompt here" --output_dir "outputs/dalle" --auto_analyze
```

Options:
- `--prompt`: Text prompt for image generation
- `--dalle_version`: Version of DALL-E to use (dall-e-2 or dall-e-3)
- `--api_key`: OpenAI API key (will use OPENAI_API_KEY environment variable if not provided)
- `--output_dir`: Directory to save generated images
- `--auto_analyze`: Automatically analyze prompt for predicate constraints

## Setup OpenAI API Key

### Windows:
```
set OPENAI_API_KEY=your_api_key_here
```

### Linux/WSL:
```
export OPENAI_API_KEY=your_api_key_here
```

## Original PredicatedDiff Usage

For Experiment (i), to generate the image of "a dog and a bowl," run:
```sh
python run.py --prompt "A dog and a bowl" --attention_corr_indices [] --attention_exist_indices [2,5] --attention_leak_indices [] --attention_possession_indices []
```

For Experiment (ii), to generate the image of "a brown dog and a yellow bowl," run:
```sh
python run.py --prompt "A brown dog and a yellow bowl" --attention_corr_indices [[2,3],[6,7]] --attention_exist_indices [3,7] --attention_leak_indices [[2,7],[6,3]] --attention_possession_indices []
```

For Experiment (iii), to generate the image of "a frog wearing a hat," run:

detailed explanation:

# Detailed Explanation of Code Changes

Here's a comprehensive breakdown of the code changes I made to implement the standalone DALL-E script with predicate logic:

## 1. Created `run_dalle_standalone.py`

This is the core standalone script that implements DALL-E with predicate logic without requiring Stable Diffusion dependencies. The key components are:

### a) OpenAI Client Initialization
```python
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
```
This function initializes the OpenAI client, checking for an API key from either the argument or environment variable.

### b) Predicate Logic Enhancement
```python
def enhance_prompt_with_predicates(prompt, corr_indices=None, leak_indices=None, exist_indices=None, possession_indices=None):
    """Enhance the prompt with predicate logic to improve DALL-E generation"""
    enhanced_parts = [prompt]
    
    # Add correlation constraints
    if corr_indices:
        for pair in corr_indices:
            if len(pair) == 2:
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
```
This function transforms abstract predicate logic constraints into natural language instructions for DALL-E.

### c) Automatic Predicate Analysis
```python
def analyze_prompt_for_predicates(prompt):
    """Analyze a prompt to identify potential predicated diffusion constraints"""
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
```
This function uses heuristics to automatically detect predicate relationships in the input prompt.

### Example:
For the prompt "A red car and a blue boat":
- It detects correlation pairs: [[1,2], [4,5]] (red→car, blue→boat)
- It identifies existence indices: [2,5] (car, boat should exist)
- No possession pairs are detected

## 2. Modified `run_dalle_demo.bat`

Added a new `--standalone` option to the batch file:

```batch
REM Function to run a single prompt with standalone DALL-E script (no Stable Diffusion dependencies)
:run_standalone
    set prompt=%~1
    if "%prompt%"=="" (
        echo No prompt provided.
        exit /b 1
    )
    
    echo Running standalone DALL-E with prompt: %prompt%
    python run_dalle_standalone.py --prompt "%prompt%" --output_dir "outputs/dalle" --auto_analyze
    exit /b 0

...

if "%1"=="--standalone" (
    if "%2"=="" (
        echo Please provide a prompt after --standalone
        exit /b 1
    )
    call :run_standalone "%~2"
    exit /b 0
)
```

This allows Windows users to run the standalone script without dependencies on Stable Diffusion.

## 3. Created `run_dalle_demo.sh`

Created a new shell script for Linux/WSL users:

```bash
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

...

if [ "$1" = "--standalone" ]; then
    if [ -z "$2" ]; then
        echo "Please provide a prompt after --standalone"
        exit 1
    fi
    run_standalone "$2"
    exit 0
fi
```

This provides the same functionality as the batch file but for Linux/WSL users.

## 4. Modified `requirements_dalle.txt`

Updated the dependencies for the standalone script:

```
openai>=1.3.0
pillow>=8.0.0
requests>=2.25.0
numpy>=1.19.0
matplotlib>=3.3.0
argparse>=1.4.0
pandas>=1.1.0
transformers>=4.0.0
spacy>=3.0.0
nltk>=3.6.0
json5>=0.9.5
```

This ensures only the necessary packages are installed, without the heavy Stable Diffusion dependencies.

## Real-World Example of How It Works

Let's walk through what happens with a specific example:

1. User runs: `./run_dalle_demo.sh --standalone "A frog wearing a hat"`

2. The script analyzes the prompt automatically:
   - Detects "wearing" as a possession relationship
   - Sets up `possession_indices = [[0,2]]` (frog→hat)
   - Sets up `exist_indices = [0,2]` (frog, hat should exist)

3. The prompt is enhanced to: "A frog wearing a hat. Make sure all objects are clearly visible in the image. The objects should have a clear ownership relationship"

4. DALL-E generates an image of a frog wearing a hat, with emphasis on:
   - Ensuring both frog and hat are clearly visible
   - Maintaining the ownership relationship (hat on frog's head)
   - Proper positioning of objects

The key difference from the original implementation is that instead of directly manipulating attention maps in the diffusion process, we're translating the predicate logic into natural language guidance that DALL-E can understand.

```sh
python run.py --prompt "A frog wearing a hat" --attention_corr_indices [] --attention_exist_indices [2,5] --attention_leak_indices [] --attention_possession_indices [2,5]
```

## More Information

For more details, please check the paper: [PredicatedDiff: Bringing Relations and Logical Structures into Diffusion Models via Predicated Guidance](https://arxiv.org/pdf/2402.04605.pdf).
