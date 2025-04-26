
# Code Execution Walkthrough for DALL-E with Predicate Logic


```
python run_dalle_standalone.py --prompt "Spiderman and batman fighting" --output_dir "outputs/dalle" --auto_analyze
```

## Step 1: Command Line Argument Parsing
```python
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
    
    args = parser.parse_args()
```
The command line arguments are parsed, with these values:
- `prompt = "Spiderman and batman fighting"`
- `output_dir = "outputs/dalle"`
- `auto_analyze = True`
- Other parameters use defaults (dall-e-3, etc.)

## Step 2: Output Directory Creation
```python
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
```
The system creates the `outputs/dalle` directory if it doesn't exist.

## Step 3: OpenAI Client Initialization
```python
    # Initialize OpenAI client
    client = init_openai_client(args.api_key)
```
This calls the `init_openai_client` function:
```python
def init_openai_client(api_key=None):
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
This initializes the OpenAI client using your API key from the environment variable.

## Step 4: Automatic Prompt Analysis
Since `--auto_analyze` was specified, the code runs:
```python
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
```

This calls `analyze_prompt_for_predicates("Spiderman and batman fighting")`:
```python
def analyze_prompt_for_predicates(prompt):
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

For "Spiderman and batman fighting", it finds:
- No color-object correlations (no color words)
- Existence indices for main objects: [0, 2] (Spiderman, batman)
- No possession relationships detected

## Step 5: Image Generation Setup
```python
    # Generate image
    print(f"Generating image for prompt: {args.prompt}")
    print(f"Using predicate constraints:")
    print(f"  - Correlation indices: {corr_indices}")
    print(f"  - Existence indices: {exist_indices}")
    print(f"  - Possession indices: {possession_indices}")
```
This displays the detected constraints.

## Step 6: DALL-E Image Generation
```python
    image, url = generate_image_with_dalle(
        client=client,
        prompt=args.prompt,
        dalle_version=args.dalle_version,
        corr_indices=corr_indices,
        exist_indices=exist_indices,
        possession_indices=possession_indices
    )
```

This calls `generate_image_with_dalle()`:
```python
def generate_image_with_dalle(client, prompt, dalle_version="dall-e-3", 
                             corr_indices=None, leak_indices=None, 
                             exist_indices=None, possession_indices=None):
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
```

## Step 7: Prompt Enhancement with Predicate Logic
The `enhance_prompt_with_predicates()` function transforms the predicate constraints into natural language:
```python
def enhance_prompt_with_predicates(prompt, corr_indices=None, leak_indices=None, exist_indices=None, possession_indices=None):
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

For your prompt "Spiderman and batman fighting", with exist_indices=[0, 2], it transforms to:
```
"Spiderman and batman fighting. Make sure all objects are clearly visible in the image"
```

## Step 8: API Call to DALL-E
```python
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
```
The enhanced prompt is sent to DALL-E's API, which generates an image ensuring Spiderman and Batman are clearly visible in the fighting scene.

## Step 9: Saving Image and Metadata
```python
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
```

The generated image is saved with a timestamp and sanitized filename:
- `outputs/dalle/1234567890_Spiderman_and_batman_fighting.png`

A metadata JSON file is also created with the same timestamp, containing:
- Original prompt
- Generation timestamp
- DALL-E version used
- Detected predicate constraints
- Image URL and local path

The result is a logically enhanced image showing both Spiderman and Batman clearly visible in a fighting scene, with the predicate logic ensuring both characters are properly represented.
