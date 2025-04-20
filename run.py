import pprint
from typing import Union, Tuple,List
import numpy as np
#import matplotlib.pyplot as plt

import pyrallis
import torch
from PIL import Image
import os
import base64
from io import BytesIO
import requests
import time
import json

from config import RunConfig
# Only import PredicatedDiffPipeline when needed, not at top level
# from pipeline_predicated_diffusion import PredicatedDiffPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
from utils import ptp_utils,vis_utils
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    if config.use_dalle:
        try:
            from openai import OpenAI
            print("Using DALL-E as backbone")
            return DALLEIntegration(api_key=config.openai_api_key, dalle_version=config.dalle_version)
        except ImportError:
            print("OpenAI package not found. Please install it with 'pip install openai'")
            raise
    else:
        # Only import PredicatedDiffPipeline when using Stable Diffusion
        from pipeline_predicated_diffusion import PredicatedDiffPipeline
        
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        if config.sd_2_1:
            stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
        else:
            stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
        stable = PredicatedDiffPipeline.from_pretrained(stable_diffusion_version).to(device)
        stable.safety_checker = lambda images, **kwargs: (images, False)
        return stable


class DALLEIntegration:
    """Integration class for DALL-E API to work with the predicated diffusion model"""
    
    def __init__(self, api_key=None, dalle_version="dall-e-3"):
        """Initialize the DALL-E integration
        
        Args:
            api_key: OpenAI API key. If None, will look for environment variable OPENAI_API_KEY
            dalle_version: Version of DALL-E to use ('dall-e-2' or 'dall-e-3')
        """
        from openai import OpenAI
        
        if api_key is None:
            # Try to get API key from environment variable
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("No API key provided and OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.dalle_version = dalle_version
        print(f"DALL-E Integration initialized with version: {dalle_version}")
    
    def __call__(self, prompt, 
                 neg_prompt=None, 
                 attention_store=None,
                 indices_to_alter=None,
                 attention_res=16,
                 guidance_scale=7.5,
                 generator=None,
                 num_inference_steps=50,
                 max_iter_to_alter=25,
                 run_standard_sd=False,
                 run_attention_sd=True,
                 thresholds=None,
                 scale_factor=20,
                 scale_range=(1.0, 0.5),
                 smooth_attentions=True,
                 sigma=0.5,
                 kernel_size=3,
                 sd_2_1=False,
                 attention_corr_indices=None,
                 attention_leak_indices=None,
                 attention_exist_indices=None,
                 attention_possession_indices=None,
                 attention_save_t=None,
                 loss_function="attention",
                 **kwargs):
        """Generate image using DALL-E API
        
        Returns dummy loss values and attention maps to maintain compatibility with
        the PredicatedDiffPipeline interface
        """
        # Create dummy loss and attention values for compatibility
        loss_value_per_step = []
        attention_for_obj_t = torch.zeros(51, 16, 16, 20)
        
        # Configure DALL-E parameters
        params = {
            "model": self.dalle_version,
            "prompt": prompt,
            "n": 1,  # Number of images to generate
            "size": "1024x1024",  # Default size
        }
        
        # Apply quality parameter for DALL-E 3
        if self.dalle_version == "dall-e-3":
            params["quality"] = "standard"  # Could be "standard" or "hd"
            params["style"] = "vivid"  # Could be "vivid" or "natural"
        
        # Include predicate logic from predicated diffusion in the prompt
        enhanced_prompt = self._enhance_prompt_with_predicates(
            prompt, 
            attention_corr_indices,
            attention_leak_indices, 
            attention_exist_indices,
            attention_possession_indices
        )
        
        params["prompt"] = enhanced_prompt
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        try:
            # Generate image using DALL-E API
            response = self.client.images.generate(**params)
            
            # Handle the response
            url = response.data[0].url
            print(f"Image generated at URL: {url}")
            
            # Download the image
            image_data = requests.get(url).content
            image = Image.open(BytesIO(image_data))
            
            # Create a simple wrapper class to match the expected return type
            class ImageWrapper:
                def __init__(self, images):
                    self.images = images
            
            return ImageWrapper([image]), loss_value_per_step, attention_for_obj_t
            
        except Exception as e:
            print(f"Error generating image with DALL-E: {e}")
            # Return a blank image in case of error
            blank_image = Image.new('RGB', (1024, 1024), color='white')
            return ImageWrapper([blank_image]), loss_value_per_step, attention_for_obj_t
    
    def _enhance_prompt_with_predicates(self, prompt, corr_indices, leak_indices, exist_indices, possession_indices):
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


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    # For DALLEIntegration, we'll use a simplified approach since we don't have access to the tokenizer
    if isinstance(stable, DALLEIntegration):
        print("Using DALL-E integration, analyzing prompt to determine token indices...")
        words = prompt.split()
        print(f"Words in prompt: {words}")
        
        # Ask the user which words to focus on
        word_indices_input = input("Please enter comma-separated word indices to focus on (e.g., 1,3 for the 2nd and 4th words): ")
        try:
            word_indices = [int(i) for i in word_indices_input.split(",")]
            print(f"Will focus on words: {[words[i] for i in word_indices if i < len(words)]}")
            return word_indices
        except ValueError:
            print("Invalid input, defaulting to indices [1, 2, 3, 4]")
            return [1, 2, 3, 4]
    
    # For the standard SD pipeline, use the tokenizer approach
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: PredicatedDiffPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig,
                  ) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs,loss_value_per_step,attention_for_obj_t = model(prompt=prompt,
                    neg_prompt = config.neg_prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    run_attention_sd = config.run_attention_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    attention_corr_indices=config.attention_corr_indices,
                    attention_leak_indices=config.attention_leak_indices,
                    attention_exist_indices=config.attention_exist_indices,
                    attention_possession_indices=config.attention_possession_indices,
                    attention_save_t=config.attention_save_t,
                    loss_function=config.loss_function,
                    )
    print("IMAGE",type(outputs.images))
    #print("att",attention_maps.shape)

    image = outputs.images[0]
    return image,loss_value_per_step,attention_for_obj_t

def show_cross_attention(prompt: str,
                         attention_maps: np.ndarray,
                         indices_to_alter: List[int],
                         res: int,
                         tokenizer,
                         orig_image=None):
    images = []
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            image = text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    view_images(np.stack(images, axis=0))

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)

    

    token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices
    tokenizer = stable.tokenizer
    tokens = tokenizer.encode(config.prompt)

    if config.run_attention_sd:
        att_output_path = config.output_path /'proposed'/config.prompt
        att_output_path.mkdir(exist_ok=True, parents=True)
    else:
        att_output_path = config.output_path /'vanilla'/config.prompt
        att_output_path.mkdir(exist_ok=True, parents=True)

    images = []
    attend_maps_for_tokens = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        #seed += 25
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image,loss_value_per_step,attention_maps = run_on_prompt(prompt=config.prompt,
                                                                                    model=stable,
                                                                                    controller=controller,
                                                                                    token_indices=token_indices,
                                                                                    seed=g,
                                                                                    config=config,
                                                                                    )
        
        image.save(att_output_path/f'{seed}_{config.prompt}.jpeg')
        images.append(image)

        #attention_maps_for_tokens = []
        #attention_for_obj = []

        """
        attention_path = config.output_path /'proposed'/'ABC-6K'/config.prompt/f"seed{seed}_attention"
        attention_path.mkdir(exist_ok=True, parents=True)
    
        for i in range (51):
            for j in range(len(tokens)):
                attention = attention_maps[i,:,:,j]
                attention = (attention-attention.min()) / (attention.max()-attention.min())
                attention = 255 * attention
                attention = attention.unsqueeze(-1).expand(*attention.shape, 3)
                attention = attention.detach().to('cpu').numpy().astype(np.uint8)
                attention = Image.fromarray(attention).resize((16,16))

                if j == 0:
                    token_name = "sot"
                elif j == len(tokens)-1:
                    token_name = "eot"
                else:
                    token_name = j
                attention.save(attention_path/f'token_{token_name}_timestep_{i}_seed_{seed}.jpeg')
        #for i,t in enumerate(config.attention_save_t):
        
            for j in range(len(tokens)):#config.token_indices:#range(len(tokens))
                attention = attention_maps[i][:,:,j]
                #print("check",attention.shape)
                attention = (attention-attention.min()) / (attention.max()-attention.min())
                attention = 255 * attention
                attention = attention.unsqueeze(-1).expand(*attention.shape, 3)
                attention = attention.detach().to('cpu').numpy().astype(np.uint8)
                attention = Image.fromarray(attention).resize((16,16))
                attention_maps_for_tokens.append(attention)

                ## ###
            
            attention_maps_for_obj = attention_for_obj_t[i]
            print(len(attention_maps_for_obj))
            
            for k in range(len(attention_maps_for_obj)):
                #print(attention_maps_for_obj[k].shape)
                attention = 255 * attention_maps_for_obj[k]
                attention = attention.view(16,16)
                attention = attention.unsqueeze(-1).expand(*attention.shape, 3)
                attention = attention.detach().to('cpu').numpy().astype(np.uint8)
                attention = Image.fromarray(attention).resize((16,16))
                attention_for_obj.append(attention)
        """    
                
        #joined_attention = vis_utils.get_attention_grid(attention_maps_for_tokens,len(tokens))#len(tokens)len(config.token_indices)
        #joined_attention.save(att_output_path/f'attention_{seed}.png')

        """
        if config.run_attention_sd:
            joined_attention_for_obj = vis_utils.get_attention_grid(attention_for_obj,len(attention_maps_for_obj))
            joined_attention_for_obj.save(att_output_path/f'attention_for_obj_{seed}.png')
        """



    

    # save a grid of results across all seeds
    #joined_image = vis_utils.get_image_grid(images)
    #joined_image.save(att_output_path/f'{config.prompt}_0_{seed}.png')





if __name__ == '__main__':
    main()
