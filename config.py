from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig:
    # Guiding text prompt
    prompt: str
    # Negative prompt
    neg_prompt: str = None
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    # Whether to use DALL-E instead of Stable Diffusion
    use_dalle: bool = False
    # DALL-E version to use (dall-e-2 or dall-e-3)
    dalle_version: str = "dall-e-3"
    # OpenAI API key for DALL-E (if None, will try to use environment variable)
    openai_api_key: str = None
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = field(default_factory=lambda: [1,2,3,4])
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [0,1,2,3,4,5,6,7,8,9])#0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    # Path to save all outputs to
    output_path: Path = Path('./outputs')
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 25
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Whether to run Attention SD or Vanilla SD
    run_attention_sd: bool = True
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(default_factory=lambda: {0:0})#0:0,3:0,5:0,10:0,15:0,20:0 3:0,5:0,10:0
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    # Standard deviation for the Gaussian smoothing
    sigma: float = 5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False
    # index of semantic inclusion pair
    attention_corr_indices: List[List[int]] = field(default_factory=lambda: [[2,3],[6,7]])
    # index of semantic inclusion pair
    attention_leak_indices: List[List[int]] = field(default_factory=lambda: [[2,7],[6,3]])#[2,7],[6,3]
    # index of semantic inclusion pair
    attention_exist_indices: List[int] = field(default_factory=lambda: [3,7])
    # index of semantic inclusion pair
    attention_possession_indices: List[List[int]] = field(default_factory=lambda: [])
    # index of attention maps for save
    attention_save_t: List[int] = field(default_factory=lambda: [0,1,2,3,4,5,6,7,8,9,10,20,30,40,50])
    # loss_fuction
    loss_function: str = "attention_product_prob"
    # mode
    mode: str = "practice"
    # dataset path for ABC-6K prompts
    dataset_path: str = None

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
