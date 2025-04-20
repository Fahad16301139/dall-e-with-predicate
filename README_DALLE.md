# Predicated Diffusion with DALL-E Integration

This extension adds support for using OpenAI's DALL-E models (DALL-E 2 or DALL-E 3) as a backbone for the Predicated Diffusion model. This allows for generating high-quality images guided by predicate logic while leveraging the capabilities of DALL-E.

## Features

- Use DALL-E 2 or DALL-E 3 as the backbone for image generation
- Apply predicate logic constraints to guide the DALL-E generation:
  - Correlation: Ensure attributes correspond to the correct objects
  - Existence: Ensure objects are present in the image
  - Possession: Handle ownership/containment relationships
- Experiment with the ABC-6K dataset for attribute binding

## Setup

1. Clone this repository and set up the required dependencies:
   ```bash
   pip install openai requests pillow torch
   ```

2. Set up your OpenAI API key:
   ```bash
   # For Linux/Mac:
   export OPENAI_API_KEY=your_api_key
   
   # For Windows:
   set OPENAI_API_KEY=your_api_key
   
   # Or provide it directly when running the scripts
   ```

3. Download the ABC-6K dataset:
   ```bash
   python dataset_loader.py --dataset ABC-6K --output_dir datasets
   ```

## Usage

### Using the Convenience Scripts

For Windows users:
```
run_dalle_demo.bat                       # Run example prompts
run_dalle_demo.bat --download-dataset    # Download the ABC-6K dataset
run_dalle_demo.bat --run-dataset 10      # Run on 10 samples from the dataset
```

For Linux/Mac users:
```bash
chmod +x run_dalle_demo.sh               # Make the script executable
./run_dalle_demo.sh                      # Run example prompts
./run_dalle_demo.sh --download-dataset   # Download the ABC-6K dataset
./run_dalle_demo.sh --run-dataset 10     # Run on 10 samples from the dataset
```

### Generate a single image with DALL-E

```bash
python run.py --prompt "A red car and a blue boat" --use_dalle True --dalle_version "dall-e-3" --attention_corr_indices [[1,2],[4,5]] --attention_exist_indices [2,5]
```

### Run on the ABC-6K dataset

```bash
python run_dalle_abc6k.py --dataset_path datasets/ABC-6K.txt --num_samples 10 --dalle_version "dall-e-3"
```

## Command Line Arguments

### For `run.py`:
- `--prompt`: The text prompt to generate an image from
- `--use_dalle`: Set to True to use DALL-E instead of Stable Diffusion
- `--dalle_version`: DALL-E version to use ("dall-e-2" or "dall-e-3")
- `--openai_api_key`: Your OpenAI API key (will use env var if not provided)
- `--attention_corr_indices`: Indices for correlation constraints
- `--attention_exist_indices`: Indices for existence constraints
- `--attention_possession_indices`: Indices for possession relationships

### For `run_dalle_abc6k.py`:
- `--dataset_path`: Path to the dataset file
- `--output_dir`: Directory to save the generated images
- `--dalle_version`: Version of DALL-E to use
- `--api_key`: OpenAI API key (will use env var if not provided)
- `--num_samples`: Number of samples to generate
- `--start_idx`: Index to start processing from
- `--seed`: Random seed for generation

## How It Works

1. When using DALL-E, the `DALLEIntegration` class is loaded instead of Stable Diffusion
2. Predicate logic constraints are converted to natural language directives that DALL-E can understand
3. The enhanced prompt is sent to the DALL-E API for image generation
4. The generated image is saved with information about the predicate constraints used

## Example Results

The ABC-6K dataset contains attribute binding prompts. For example:

Prompt: "A man in a blue shirt riding a black horse"
- Correlation pairs: [[4,5], [8,9]]
- Existence indices: [5, 9]

This ensures that the attributes "blue" and "black" are correctly bound to "shirt" and "horse" respectively.

## Notes

- DALL-E API calls cost money, so be mindful of the number of images you generate
- DALL-E API has rate limits, so the script includes delays between requests
- The predicate logic constraints are translated to natural language hints for DALL-E, which may not be as precise as the original Stable Diffusion implementation

## Citation

If you use this code for your research, please cite the original Predicated Diffusion paper:

```
@inproceedings{feng2023trainingfree,
title={Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis},
author={Weixi Feng and Xuehai He and Tsu-Jui Fu and Varun Jampani and Arjun Reddy Akula and Pradyumna Narayana and Sugato Basu and Xin Eric Wang and William Yang Wang},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023}
}
``` 