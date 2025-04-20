
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention,aggregate_attention_origin
import torch.distributions as dist

logger = logging.get_logger(__name__)

class PredicatedDiffPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds
        

    def _aggregate_attention_per_token(self, attention_store: AttentionStore, attention_res: int = 16):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),#, "down", "mid"
            is_cross=True,
            select=0)
        return attention_maps

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents


    @staticmethod
    def _compute_loss_attention(attention_maps: torch.Tensor,attention_corr_indices:List[List[int]],size:int) -> torch.Tensor:
        loss = 0
        loss_attend = 0
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        for ind in attention_corr_indices:
            attenB = attention_maps[:,:,ind[0]]
            attenA = attention_maps[:,:,ind[1]]
            # B : adj
            # A : noun
            # loss
            # B → A : attenB * (1 - attenA)
            # A → B : attenA * (1 - attenB)

            loss_value = (1 - attenA * (1 - attenB))

            loss_value = torch.sum(loss_value) / size
            loss += loss_value
            print("loss_value",loss_value)
        return - loss

    @staticmethod
    def _compute_loss_attention_product_prob(attention_maps: torch.Tensor,attention_corr_indices:List[List[int]],attention_leak_indices:List[List[int]],attention_exist_indices:List[int],attention_possession_indices:List[List[int]],size:int) -> torch.Tensor:
        loss = 0
        loss_value_list = []
        loss_attend = 0
        eps = 1e-20 
        attention_for_obj = []
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        print("max_att",attention_maps[:,:,:].max())
        # prepare attention for exist loss
        
        last_idx = -1
        attention_normalize = attention_maps[:, :, 1:last_idx].clone()
        attention_normalize *= 100
        attention_normalize = torch.nn.functional.softmax(attention_normalize, dim=-1)
        
        print("max_att",attention_maps[:,:,:].max())

        ################################
        # 1.compute exist loss
        ################################
        for exist_ind in attention_exist_indices:
            print("exist",exist_ind)
            atten_obj_exist = attention_normalize[:,:,exist_ind-1]
        
            """
            smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=1, dim=2).cuda()
            input = F.pad(atten_obj_exist.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            atten_obj_exist = smoothing(input).squeeze(0).squeeze(0)
            """
            
            loss_attend_value = torch.sum(torch.log(1 - atten_obj_exist + 1e-4))/ size
            loss_attend_value =  - torch.log(1 - torch.exp(loss_attend_value))
        
            loss = loss + 0.1 * loss_attend_value /len(attention_exist_indices)
            print("attend",loss_attend_value)
        #print("count",len(attention_exist_indices))
        
        ################################
        # 2.compute correspondece loss
        ################################
        for corr_ind in attention_corr_indices:
            #print("corr")
            atten_adj = attention_maps[:,:,corr_ind[0]]
            atten_obj = attention_maps[:,:,corr_ind[1]]

            # 0-1 normalized attention
            atten_adj = (atten_adj - atten_adj.min())/(atten_adj.max() - atten_adj.min())
            atten_obj = (atten_obj - atten_obj.min())/(atten_obj.max() - atten_obj.min())
            attention_for_obj.append(atten_adj)
            attention_for_obj.append(atten_obj)

            
            

            # obj → adj loss + adj → obj loss
            loss_corr_value = - torch.log(1 - (atten_obj * (1 - atten_adj)) + eps) - torch.log(1 - (atten_adj * (1 - atten_obj)) + eps)
            #loss_corr_value = - torch.log(1 - (atten_adj * (1 - atten_obj)) + eps)
            loss_corr_value = torch.sum(loss_corr_value)/size

            loss = loss + loss_corr_value / len(attention_corr_indices)
            print("corr",loss_corr_value)
        

        ################################
        # 3.compute leak loss
        ################################
        for leak_ind in attention_leak_indices:
            
            # leak loss
            # P → ¬Q
            # Q → ¬P
            # log(1 - attenP * attenQ)
            atten_adj = attention_maps[:,:,leak_ind[0]]
            atten_obj = attention_maps[:,:,leak_ind[1]]
            
            
            #print("leak")
            # 0-1 normalized attention
            atten_adj = (atten_adj - atten_adj.min())/(atten_adj.max() - atten_adj.min())
            atten_obj = (atten_obj - atten_obj.min())/(atten_obj.max() - atten_obj.min())

            loss_leak_value = - torch.log(1 - (atten_obj * atten_adj) + eps)
            loss_leak_value = torch.sum(loss_leak_value)/size

            loss = loss + 0.3 * loss_leak_value / len(attention_leak_indices)
            print("leak",loss_leak_value)

        ################################
        # 4.compute possession loss
        ################################
        for possession_ind in attention_possession_indices:
            
            
            atten_holder = attention_maps[:,:,possession_ind[0]]
            atten_possession = attention_maps[:,:,possession_ind[1]]

            # 0-1 normalized attention
            atten_possession = (atten_possession - atten_possession.min())/(atten_possession.max() - atten_possession.min())
            atten_holder = (atten_holder - atten_holder.min())/(atten_holder.max() - atten_holder.min())
            attention_for_obj.append(atten_possession)
            attention_for_obj.append(atten_holder)

            # obj(ex. glasses) → adj(ex. man) loss
            loss_possession_value = - torch.log(1 - (atten_possession * (1 - atten_holder)) + eps)
            loss_possession_value = torch.sum(loss_possession_value)/size

            loss = loss + loss_possession_value / len(attention_possession_indices)
            print("possession",loss_possession_value)

            
            
            
            
        
        print("loss",loss)

        
        return loss,attention_for_obj


    
    @staticmethod
    def _compute_loss_attention_exist(attention_maps: torch.Tensor,attention_corr_indices:List[List[int]],size:int) -> torch.Tensor:
        loss = 0
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        for ind in attention_corr_indices:
            """
            smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).cuda()
            input = F.pad(attention_maps[:,:,ind[1]].unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            attention_maps[:,:,ind[1]] = smoothing(input).squeeze(0).squeeze(0)
            input = F.pad(attention_maps[:,:,ind[3]].unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            attention_maps[:,:,ind[3]] = smoothing(input).squeeze(0).squeeze(0)
            """
            
            last_idx = -1
            #last_idx = 9 - 1
            attention_normalize = attention_maps[:, :, 1:last_idx]
            attention_normalize *= 100
            attention_maps[:,:,1:last_idx] = torch.nn.functional.softmax(attention_normalize, dim=-1)
            
            
            attenB = attention_maps[:,:,ind[1]]
            attenD = attention_maps[:,:,ind[3]]
            
            
            
            #print("attenB",attention_maps[:,:,ind[1]])
            print("max1",attenB.max())
            print("min1",attenB.min())
            print("max2",attenD.max())
            print("min2",attenD.min())
            
            #attenB = torch.where(attenB<=0.1,0,attenB)
            #attenD = torch.where(attenD<=0.1,0,attenD)
        
            
            
            loss_value_noun = - torch.log(1 - torch.exp(torch.sum(torch.log(1 - attenB + 1e-8))/ size)) - torch.log(1 - torch.exp(torch.sum(torch.log(1 - attenD + 1e-8))/ size))
            #loss_value_noun = - torch.log(1 - torch.exp(torch.sum(torch.log(1 - attenB + 1e-8)))) - torch.log(1 - torch.exp(torch.sum(torch.log(1 - attenD + 1e-8))))
            #loss_value_noun = - torch.log(1 - (1-attenB).prod()) - torch.log(1 - (1-attenD).prod()) 
            
            loss += loss_value_noun
            print("loss_value_noun",loss_value_noun)
        
        print("loss",loss)
        attention_for_obj = [attenB,attenD]
        return loss,attention_for_obj
        
    
    @staticmethod    
    def _compute_loss_attention_negation(attention_maps: torch.Tensor,attention_corr_indices:List[List[int]],size:int) -> torch.Tensor:
        loss = 0
        loss_attend = 0
        loss_neg = 0
        for ind in attention_corr_indices:
            # A : positive prompt
            # B : negative prompt
            # A → not B 
            #attenA = attention_maps[:,:,ind[0]]
            #attenB = attention_maps[:,:,ind[1]] 
            last_idx = -1
            #last_idx = 9 - 1
            attention_normalize = attention_maps[:, :, 1:last_idx]
            attention_normalize *= 100
            attention_maps[:,:,1:last_idx] = torch.nn.functional.softmax(attention_normalize, dim=-1)
            
            attenA = attention_maps[:,:,1]
            

            """
            smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=1, dim=2).cuda()
            input = F.pad(attenA.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            attenA = smoothing(input).squeeze(0).squeeze(0)
            """
            

            #print("attenB",attention_maps[:,:,ind[1]])
            print("max",attenA.max())
            print("min",attenA.min())

            ###########

            ## Method 1 ##
            #loss_neg += torch.log(1 - (1-attenA).prod() + 1e-20)
            #loss_neg += -torch.log((1-attenA).prod())
            #attenA = (attenA-attenA.min())/ (attenA.max()-attenA.min())
            loss_neg += - torch.sum(torch.log(1-attenA+1e-8))
            loss_neg = 0.1 * loss_neg/size

            ###############

            ## Method 2 ##
            """
            attenA = (attenA-attenA.min())/ (attenA.max()-attenA.min()) 
            attenB = (attenB-attenB.min())/ (attenB.max()-attenB.min())
            

            loss_neg = - torch.log(1-(attenA * attenB) + 1e-8)
            loss_neg = torch.sum(loss_neg) / size
            """
            #attenA = (attenA-attenA.min())/ (attenA.max()-attenA.min()) 
            ###############
        
        loss = loss_neg #+ loss_attend
        print("loss",loss)
        attention_for_obj = [attenA]
        return loss,attention_for_obj
    

    def _perform_iterative_refinement_fixed_step_att(self,
                                                    latents: torch.Tensor,
                                                    loss: torch.Tensor,
                                                    threshold: float,
                                                    text_embeddings: torch.Tensor,
                                                    text_input,
                                                    attention_corr_indices: List[List[int]],
                                                    attention_leak_indices: List[List[int]],
                                                    attention_exist_indices: List[int],
                                                    attention_possession_indices: List[List[int]],
                                                    attention_store: AttentionStore,
                                                    step_size: float,
                                                    t: int,
                                                    attention_res: int = 16,
                                                    max_refinement_steps: int = 20,
                                                    attention_size: int = 256,
                                                    attention_loss_function: str = "attention_exist",
                                                    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        for iteration in range(5):
            
            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            attention_maps = self._aggregate_attention_per_token(
                        attention_store=attention_store,
                        attention_res=attention_res,
                    )

            if attention_loss_function == "attention_product_prob":
                loss,attention_for_obj = self._compute_loss_attention_product_prob(attention_maps=attention_maps,size=attention_size,
                                                                                    attention_corr_indices=attention_corr_indices,
                                                                                    attention_leak_indices=attention_leak_indices,
                                                                                    attention_exist_indices=attention_exist_indices,
                                                                                    attention_possession_indices= attention_possession_indices)
            elif attention_loss_function == "attention_exist":
                loss,attention_for_obj = self._compute_loss_attention_exist(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
            elif attention_loss_function == "attention_negation":
                loss,attention_for_obj = self._compute_loss_attention_negation(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
                

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            print(f"iter{iteration}_loss_{loss}")

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        attention_maps = self._aggregate_attention_per_token(
                        attention_store=attention_store,
                        attention_res=attention_res,
                    )

        if attention_loss_function == "attention_product_prob":
            loss,attention_for_obj = self._compute_loss_attention_product_prob(attention_maps=attention_maps,size=attention_size,
                                                                                attention_corr_indices=attention_corr_indices,
                                                                                attention_leak_indices=attention_leak_indices,
                                                                                attention_exist_indices=attention_exist_indices,
                                                                                attention_possession_indices= attention_possession_indices)
        elif attention_loss_function == "attention_exist":
            loss,attention_for_obj = self._compute_loss_attention_exist(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
        elif attention_loss_function == "attention_negation":
            loss,attention_for_obj = self._compute_loss_attention_negation(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
        
        print(f"\t Finished with loss of: {loss}")
        return loss, latents

    #def _caluate_loss()

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            neg_prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            run_attention_sd: bool = True, 
            thresholds: Optional[dict] = {0: 0.99, 5: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.8),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
            attention_corr_indices: List[List[int]] = None,
            attention_leak_indices: List[List[int]] = None,
            attention_exist_indices: List[int] = None,
            attention_save_t: List[int] = None,
            attention_possession_indices: List[List[int]] = None,
            loss_function: str = "attention",
            
            
    ):
        """
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_positive_inputs, prompt_positive_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        #print("num_channels_latents",num_channels_latents)
        
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_positive_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))
        

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        attention_maps_t = []
        attention_size = attention_res ** 2
        loss_value_per_step = []

        #attention_for_obj_t = []
        attention_for_obj_t = torch.zeros(51,16,16,20)


        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                print("I",i)
                with torch.enable_grad():

                    latents = latents.clone().detach().requires_grad_(True)

                    # Forward pass of denoising with text conditioning
                    noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_positive_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                    self.unet.zero_grad()
                    print("prompt_positive_embeds[1].unsqueeze(0)",prompt_positive_embeds[1].unsqueeze(0).shape)
                    print("noise_pred_text",noise_pred_text.shape)
                
                    # Get attention maps per token 
                    # attention maps : [16,16,77]
                    """
                    attention_maps = self._aggregate_attention_per_token(
                        attention_store=attention_store,
                        attention_res=attention_res,
                    )
                    """
                    attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)
                    
                
                    if run_attention_sd:
                        #attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)
                        
                        
                        print(loss_function)
                        

                        if loss_function == "attention":
                            loss = self._compute_loss_attention(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
                            if i < max_iter_to_alter:
                                if loss != 0:
                                    latents = self._update_latent(latents=latents, loss=loss,
                                                            step_size=scale_factor * np.sqrt(scale_range[i]))
                                    print(f'Iteration {i} | Loss: {loss:0.4f}')  
                        elif loss_function == "attention_product_prob":
                            
                            loss,attention_for_obj = self._compute_loss_attention_product_prob(attention_maps=attention_maps,size=attention_size,
                                                                                                attention_corr_indices=attention_corr_indices,
                                                                                                attention_leak_indices=attention_leak_indices,
                                                                                                attention_exist_indices=attention_exist_indices,
                                                                                                attention_possession_indices= attention_possession_indices)
                            run_negation = False
                            if run_negation == True:
                                if neg_prompt is None:
                                    1/0
                                else:
                                    #prompt_add_neg = prompt + ' ' + neg_prompt
                                    prompt_neg = neg_prompt
                                    #print("prompt_add_neg",neg_prompt)
                            

                                text_add_neg_inputs,prompt_add_neg_embeds = self._encode_prompt(prompt_neg,device,num_images_per_prompt,do_classifier_free_guidance,negative_prompt,prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_prompt_embeds)
                                noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_add_neg_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                                self.unet.zero_grad()

                                attention_maps_for_negation = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)
                                loss_neg,attention_for_obj_neg = self._compute_loss_attention_negation(attention_maps=attention_maps_for_negation,attention_corr_indices=attention_corr_indices,size=attention_size)

                                loss = loss + loss_neg
                            
                            if i < max_iter_to_alter:
                                if loss != 0:
                                    latents = self._update_latent(latents=latents, loss=loss,
                                                            step_size=scale_factor * np.sqrt(scale_range[i]))
                                    print(f'Iteration {i} | Loss: {loss:0.4f}')

                                    
                        elif loss_function == "attention_exist":
                            print("Exist")
                            loss,attention_for_obj = self._compute_loss_attention_exist(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
                            if i < max_iter_to_alter:
                                if loss != 0:
                                    latents = self._update_latent(latents=latents, loss=loss,
                                                            step_size=scale_factor * np.sqrt(scale_range[i]))
                                    print(f'Iteration {i} | Loss: {loss:0.4f}')
                        elif loss_function == "attention_negation":
                            print("Negation")
                            if neg_prompt is None:
                                1/0
                            else:
                                #prompt_add_neg = prompt + ' ' + neg_prompt
                                prompt_add_neg = neg_prompt
                            #print("prompt_add_neg",neg_prompt)
                            

                            text_add_neg_inputs,prompt_add_neg_embeds = self._encode_prompt(prompt_add_neg,device,num_images_per_prompt,do_classifier_free_guidance,negative_prompt,prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_prompt_embeds)
    

                            noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_add_neg_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                            self.unet.zero_grad()

                            attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)
                            loss,attention_for_obj = self._compute_loss_attention_negation(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
                            
                            if i < max_iter_to_alter:
                                if loss != 0:
                                    latents = self._update_latent(latents=latents, loss=loss,
                                                            step_size=scale_factor * np.sqrt(scale_range[i]))
                                    print(f'Iteration {i} | Loss: {loss:0.4f}')
                            
                            
                    
                        
                        
                        if i in thresholds.keys(): #and loss > 1. - thresholds[i]:
                            if loss_function != "attention_negation":
                                loss,latents = self._perform_iterative_refinement_fixed_step_att(
                                    latents=latents,
                                    loss=loss,
                                    threshold=thresholds[i],
                                    text_embeddings=prompt_positive_embeds,
                                    text_input=text_positive_inputs,
                                    attention_store=attention_store,
                                    step_size=scale_factor * np.sqrt(scale_range[i]),
                                    t=t,
                                    attention_res=attention_res,
                                    max_refinement_steps=20,
                                    attention_corr_indices=attention_corr_indices,
                                    attention_leak_indices=attention_leak_indices,
                                    attention_exist_indices=attention_exist_indices,
                                    attention_possession_indices= attention_possession_indices,
                                    attention_loss_function=loss_function)
                            else:
                                loss,latents = self._perform_iterative_refinement_fixed_step_att(
                                    latents=latents,
                                    loss=loss,
                                    threshold=thresholds[i],
                                    text_embeddings=prompt_add_neg_embeds,
                                    text_input=text_add_neg_inputs,
                                    attention_store=attention_store,
                                    step_size=scale_factor * np.sqrt(scale_range[i]),
                                    t=t,
                                    attention_res=attention_res,
                                    max_refinement_steps=20,
                                    attention_corr_indices=attention_corr_indices,
                                    attention_leak_indices=attention_leak_indices,
                                    attention_exist_indices=attention_exist_indices,
                                    attention_possession_indices= attention_possession_indices,
                                    attention_loss_function=loss_function)

                        loss_value_per_step.append(loss.detach().cpu().numpy())#loss.detach().cpu().numpy()
                        #loss_value_per_step = torch.cat(loss_value_per_step,dim=0)
                        #print(loss_value_per_step)
                        

                attention_for_obj_t[i,:,:,:]=attention_maps[:,:,0:20]
                    

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                #print("latent_model_input",latent_model_input.shape)
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_positive_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                self.unet.zero_grad()
                if i == 50:

                    attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                


                
                # save attention maps for t
                if i in attention_save_t:
                    """
                    text_add_neg_inputs,prompt_add_neg_embeds = self._encode_prompt(neg_prompt,device,num_images_per_prompt,do_classifier_free_guidance,negative_prompt,prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_prompt_embeds)
                    noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_add_neg_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                    self.unet.zero_grad()
                    """
                    """
                    attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res) 
                    print("max",attention_maps[:,:,1].max())
                    print("ave",torch.mean(attention_maps[:,:,1]))
                    """

                    if run_attention_sd == False:
                        attention_for_obj = []

                    """
                    print("len_token",num_tokens)
                    last_idx = num_tokens - 1
                    #last_idx = -1
                    attention_normalize = attention_maps[:, :, 1:last_idx]
                    attention_normalize *= 100
                    attention_maps[:,:,1:last_idx] = torch.nn.functional.softmax(attention_normalize, dim=-1)
                    print("AttShape",attention_maps[:,:,3].max())
                    """
                    
                
                    #attention_maps_t.append(attention_maps)
                

                
                #attention_for_obj_t.append(attention_maps[:,:,0:20])
                    
        #attention_for_obj_t[50,:,:,:]=attention_maps[:,:,0:20]       
                
            
        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_positive_embeds.dtype)

        print("image_numpy",image.shape)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            
            #attention_maps = attention_maps.to('cpu').numpy()
            

        if not return_dict:
            return (image, has_nsfw_concept)

        
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept),loss_value_per_step,attention_for_obj_t
