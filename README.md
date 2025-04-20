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
```sh
python run.py --prompt "A frog wearing a hat" --attention_corr_indices [] --attention_exist_indices [2,5] --attention_leak_indices [] --attention_possession_indices [2,5]
```

## More Information

For more details, please check the paper: [PredicatedDiff: Bringing Relations and Logical Structures into Diffusion Models via Predicated Guidance](https://arxiv.org/pdf/2402.04605.pdf).
