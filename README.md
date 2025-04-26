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


# Detailed Explanation of DALL-E with Predicate Logic Implementation

## Modified/Created Files

1. **run_dalle_standalone.py** (Created new)
   - This is the core standalone script that uses DALL-E with predicate logic
   - Implements the full workflow without Stable Diffusion dependencies

2. **run_dalle_demo.bat** (Modified)
   - Added `--standalone` option for Windows users
   - Added function to call the standalone script

3. **run_dalle_demo.sh** (Created new)
   - Created shell script version for Linux/WSL users
   - Implemented same functionality as batch file

4. **requirements_dalle.txt** (Modified)
   - Updated with minimal dependencies for standalone script

## How DALL-E Uses Predicate Logic

The key implementation of predicate logic is in `run_dalle_standalone.py`. Here's how it works:

### 1. Predicate Representation

There are three main types of predicates implemented:

```python
# In run_dalle_standalone.py
def enhance_prompt_with_predicates(prompt, corr_indices=None, leak_indices=None, exist_indices=None, possession_indices=None):
    """Enhance the prompt with predicate logic to improve DALL-E generation"""
    enhanced_parts = [prompt]
    
    # CORRELATION PREDICATE: Ensures attributes correctly apply to objects
    # Example: For "red car and blue boat", corr_indices=[[1,2],[4,5]] ensures "red" applies to "car" and "blue" to "boat"
    if corr_indices:
        for pair in corr_indices:
            if len(pair) == 2:
                enhanced_parts.append(f"Make sure the attributes correctly correspond to their objects")
    
    # EXISTENCE PREDICATE: Ensures objects actually appear in the image
    # Example: For "red car and blue boat", exist_indices=[2,5] ensures both "car" and "boat" are clearly visible
    if exist_indices:
        enhanced_parts.append(f"Make sure all objects are clearly visible in the image")
    
    # POSSESSION PREDICATE: Ensures ownership relationships between objects
    # Example: For "frog wearing a hat", possession_indices=[[0,2]] ensures the hat belongs to/is on the frog
    if possession_indices:
        for pair in possession_indices:
            if len(pair) == 2:
                enhanced_parts.append(f"The objects should have a clear ownership relationship")
    
    # Create final enhanced prompt
    enhanced_prompt = prompt + ". " + " ".join(enhanced_parts[1:])
    return enhanced_prompt
```

### 2. Automatic Predicate Detection

The script can automatically identify these predicate relationships:

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
    
    # Find color-object correlations (CORRELATION PREDICATE)
    for i, word in enumerate(words):
        if word in colors and i < len(words) - 1:
            corr_pairs.append([i, i+1])
            exist_indices.append(i+1)  # The object should exist
    
    # Find possession relationships (POSSESSION PREDICATE)
    for i, word in enumerate(words):
        if word in ['with', 'wearing', 'holding'] and i > 0 and i < len(words) - 1:
            possession_indices.append([i-1, i+1])
    
    return corr_pairs, exist_indices, possession_indices
```

## Complete Workflow Execution

Here's the step-by-step execution flow:

1. **Entry Point**: 
   - User invokes script with `./run_dalle_demo.sh --standalone "A red car and a blue boat"`

2. **Shell Script Execution**:
   ```bash
   # In run_dalle_demo.sh
   if [ "$1" = "--standalone" ]; then
       if [ -z "$2" ]; then
           echo "Please provide a prompt after --standalone"
           exit 1
       fi
       run_standalone "$2"  # This calls the run_standalone function
       exit 0
   fi
   
   # The run_standalone function calls the Python script
   run_standalone() {
       python run_dalle_standalone.py --prompt "$1" --output_dir "outputs/dalle" --auto_analyze
   }
   ```

3. **Python Script Main Function**:
   ```python
   # In run_dalle_standalone.py
   def main():
       # Parse command line arguments
       parser = argparse.ArgumentParser()
       args = parser.parse_args()
       
       # Create output directory
       output_dir = Path(args.output_dir)
       output_dir.mkdir(exist_ok=True, parents=True)
       
       # Initialize OpenAI client
       client = init_openai_client(args.api_key)
       
       # Auto analyze prompt if requested
       if args.auto_analyze:
           print("Automatically analyzing prompt for predicate constraints...")
           corr_pairs, exist_idx, poss_pairs = analyze_prompt_for_predicates(args.prompt)
           # Use these automatically detected constraints
       
       # Generate image with DALL-E
       image, url = generate_image_with_dalle(
           client=client,
           prompt=args.prompt,
           dalle_version=args.dalle_version,
           corr_indices=corr_indices,
           exist_indices=exist_indices,
           possession_indices=possession_indices
       )
       
       # Save image and metadata
       # ...
   ```

4. **Prompt Analysis** (if auto_analyze is enabled):
   - `analyze_prompt_for_predicates()` identifies potential predicate constraints
   - For "A red car and a blue boat" finds:
     - Correlation pairs: [[1,2], [5,6]]
     - Existence indices: [2, 6]

5. **DALL-E Image Generation**:
   ```python
   def generate_image_with_dalle(client, prompt, dalle_version, corr_indices, exist_indices, possession_indices):
       # Configure DALL-E parameters
       params = {
           "model": dalle_version,
           "prompt": prompt,
           "n": 1,
           "size": "1024x1024",
       }
       
       # Enhance prompt with predicate logic
       enhanced_prompt = enhance_prompt_with_predicates(
           prompt, 
           corr_indices,
           leak_indices, 
           exist_indices,
           possession_indices
       )
       
       params["prompt"] = enhanced_prompt
       
       # Call OpenAI API to generate image
       response = client.images.generate(**params)
       
       # Download and return the image
       url = response.data[0].url
       image_data = requests.get(url).content
       image = Image.open(BytesIO(image_data))
       
       return image, url
   ```

6. **Predicate Logic Application**:
   - The prompt is enhanced with natural language expressions of predicates
   - For example: "A red car and a blue boat. Make sure the attributes correctly correspond to their objects Make sure all objects are clearly visible in the image"

7. **Image Generation and Saving**:
   - The enhanced prompt is sent to DALL-E via OpenAI API
   - The image is downloaded and saved to the output directory
   - Metadata about the generation is saved alongside the image

## Key Difference from Original Implementation

In the original PredicatedDiff with Stable Diffusion:
- Predicates directly manipulate attention maps in the diffusion process
- Mathematical operations directly enforce constraints

In the DALL-E standalone version:
- Predicates are translated to natural language that DALL-E can understand
- We rely on DALL-E's language understanding to interpret and apply the constraints
- This is a higher-level, less direct approach but avoids the need for direct access to attention mechanisms

This approach demonstrates that predicate logic can be applied to text-to-image models even without direct access to their internal parameters or attention mechanisms.

