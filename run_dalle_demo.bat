@echo off
setlocal enabledelayedexpansion

REM Check for OpenAI API key
if "%OPENAI_API_KEY%"=="" (
    echo Error: OPENAI_API_KEY environment variable is not set.
    echo Please set your OpenAI API key with: set OPENAI_API_KEY=your_api_key
    exit /b 1
)

REM Create necessary directories
if not exist "datasets" mkdir datasets
if not exist "outputs\dalle" mkdir outputs\dalle

REM Make sure dependencies are installed
echo Installing dependencies...
pip install -r requirements_dalle.txt

REM Function to run a single prompt with DALL-E
:run_single_prompt
    set prompt=%~1
    if "%prompt%"=="" (
        echo No prompt provided.
        exit /b 1
    )
    
    echo Running DALL-E with prompt: %prompt%
    python run.py --text_input "%prompt%" --output_dir "outputs/dalle" --guidance_scale 7.5 --seed 42 --model_key "dalle"
    exit /b 0

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

REM Parse command-line arguments
if "%1"=="--download-dataset" (
    echo Downloading ABC-6K dataset...
    python dataset_loader.py
    exit /b 0
)

if "%1"=="--run-dataset" (
    if "%2"=="" (
        set num_samples=5
    ) else (
        set num_samples=%2
    )
    echo Running DALL-E on %num_samples% samples from ABC-6K dataset...
    python run_dalle_abc6k.py --num_samples %num_samples%
    exit /b 0
)

if "%1"=="--standalone" (
    if "%2"=="" (
        echo Please provide a prompt after --standalone
        exit /b 1
    )
    call :run_standalone "%~2"
    exit /b 0
)

REM If no specific args provided, run examples
if "%1"=="" (
    echo Running example prompts with DALL-E...
    
    echo Example 1: Simple color-object binding
    call :run_single_prompt "A red car and a blue boat"
    
    echo Example 2: Object possession relationship
    call :run_single_prompt "A frog wearing a hat"
    
    echo Example 3: Complex scene with multiple constraints
    call :run_single_prompt "A brown dog and a yellow bowl"
    
    echo All examples completed. Check results in outputs/dalle directory.
) else (
    REM If a prompt is provided, run it
    call :run_single_prompt "%~1"
)

endlocal 