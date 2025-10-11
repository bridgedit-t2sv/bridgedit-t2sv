import pytorch_lightning as pl
import torch
from typing import Optional, Union, List, Dict, Any 
from models.stable_audio.stable_audio_transformer import StableAudioDiTModel
from models.stable_audio.stable_audio_modeling import StableAudioProjectionModel
from models.wan_video.wan_video_transformer import WanTransformer3DModel
from models.dit.dit_dual_stream_clean import DualDiTBlockAdaLN, CrossDiTBlockAdaLN, AddFusionDiTBlockAdaLN
from diffusers.models import AutoencoderOobleck, AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler, CosineDPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, T5EncoderModel, UMT5EncoderModel
from diffusers.models.embeddings import get_1d_rotary_pos_embed
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm
from diffusers.utils import export_to_video
import torchvision.io as tvio
import torchvision.transforms as transforms
import torchaudio
import torchvision
import soundfile as sf
import regex as re
import ftfy
import html
import inspect
import yaml
from moviepy import VideoFileClip, AudioFileClip, VideoClip, AudioArrayClip
import numpy as np
import time
import torch.nn as nn
from functools import partial
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
import logging
from datetime import datetime
import math
import os
import deepspeed 
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

class BasePipeline(pl.LightningModule):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.load_path_config = self.config['load_path_config']
        self.train_config = self.config['train_config']
        self.model_archi_config = self.config['model_archi_config']
        self.sample_config = self.config['sample_config']
        self.weight_dtype = torch.bfloat16
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        get_huggingface_model_from_local_dir = lambda ckpt_dir, model: \
            model.from_pretrained(
                ckpt_dir, local_files_only=True,    # From pretrained
                ignore_mismatched_sizes=False,      # Strict for pretrained model structure
                use_safetensors=True,               # Safe tensor
                torch_dtype=self.weight_dtype,      # bfloat16
            )
        self.model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ## Load Model Part
        ckpt_dir_audio_diffusion = self.load_path_config['audio_model_path']['ckpt_dir_audio_diffusion']
        ckpt_dir_audio_vae = self.load_path_config['audio_model_path']['ckpt_dir_audio_vae']
        ckpt_dir_projection = self.load_path_config['audio_model_path']['ckpt_dir_projection']
        audio_pipeline_scheduler_config = self.load_path_config['audio_model_path']['audio_pipeline_scheduler_config']
        audio_text_encoder = self.load_path_config['audio_model_path']['audio_text_encoder']
        audio_tokenizer = self.load_path_config['audio_model_path']['audio_tokenizer']
        ckpt_dir_video_vae = self.load_path_config['video_model_path']['ckpt_dir_video_vae']
        ckpt_dir_video_diffusion = self.load_path_config['video_model_path']['ckpt_dir_video_diffusion']
        video_pipeline_scheduler_config = self.load_path_config['video_model_path']['video_pipeline_scheduler_config']
        video_text_encoder = self.load_path_config['video_model_path']['video_text_encoder']
        video_tokenizer = self.load_path_config['video_model_path']['video_tokenizer']
        self.audio_diffusion = get_huggingface_model_from_local_dir(ckpt_dir_audio_diffusion, StableAudioDiTModel)
        self.audio_vae = get_huggingface_model_from_local_dir(ckpt_dir_audio_vae, AutoencoderOobleck)
        self.audio_projection_model = get_huggingface_model_from_local_dir(ckpt_dir_projection, StableAudioProjectionModel)
        # self.audio_text_encoder = get_huggingface_model_from_local_dir(audio_text_encoder, T5EncoderModel)
        self.audio_tokenizer = AutoTokenizer.from_pretrained(audio_tokenizer)
        self.audio_pipeline_scheduler = get_huggingface_model_from_local_dir(audio_pipeline_scheduler_config, CosineDPMSolverMultistepScheduler)
        self.sample_audio_length_s = self.config.get('sample_audio_length_s', 5.40)
        self.sample_audio_sampling_rate = self.config.get('sample_audio_sampling_rate', 44100)
        self.video_diffusion = get_huggingface_model_from_local_dir(ckpt_dir_video_diffusion, WanTransformer3DModel)
        self.video_vae = get_huggingface_model_from_local_dir(ckpt_dir_video_vae, AutoencoderKLWan)
        self.video_tokenizer = AutoTokenizer.from_pretrained(video_tokenizer)
        self.video_pipeline_scheduler = get_huggingface_model_from_local_dir(video_pipeline_scheduler_config, UniPCMultistepScheduler)
        ## Sample Config Part
        self.sample_video_fps = self.sample_config.get('sample_video_fps', 15)
        self.sample_video_num_frames = self.sample_config.get('sample_video_num_frames', 81)
        self.sample_video_height = self.sample_config.get('sample_video_height', 480)
        self.sample_video_width = self.sample_config.get('sample_video_width', 832)
        self.num_inference_steps = self.sample_config.get('num_inference_steps', 50)
        self.sample_video_guidance_scale = self.sample_config.get('sample_video_guidance_scale', 6)
        self.num_samples_per_prompt = self.sample_config.get('num_samples_per_prompt', 1)
        self.inference_seed = self.sample_config.get('inference_seed', 0)
        self.audio_guidance_scale = self.sample_config.get('audio_guidance_scale', 7)
        self.video_save_path = self.sample_config.get('video_save_path', "save_videos")
        self.video_vae_scale_factor_temporal = 2 ** sum(self.video_vae.temperal_downsample) if getattr(self, "video_vae", None) else 4
        self.video_vae_scale_factor_spatial = 2 ** len(self.video_vae.temperal_downsample) if getattr(self, "video_vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.video_vae_scale_factor_spatial)
        self.negative_prompt = self.sample_config.get('negative_prompt', "Low quality")
        self.use_condition_embedding_cache = self.train_config.get('cache_config', {}).get('use_condition_embedding_cache', True)
        if not self.use_condition_embedding_cache:
            print(audio_text_encoder, video_text_encoder)
            self.audio_text_encoder = get_huggingface_model_from_local_dir(audio_text_encoder, T5EncoderModel)
            self.video_text_encoder = get_huggingface_model_from_local_dir(video_text_encoder, UMT5EncoderModel)
        print("Successfully loaded model")
        
    def forward(self):
        pass

    def prepare_audio_latents(self, shape, dtype, device, generator, init_noise_sigma, latents=None):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * init_noise_sigma
        return latents

    def prepare_video_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.video_vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.video_vae_scale_factor_spatial,
            int(width) // self.video_vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_samples_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,  
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = dtype or self.video_text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.video_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long() 
        self.video_text_encoder.to(device)
        prompt_embeds = self.video_text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_samples_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_samples_per_prompt, seq_len, -1)

        return prompt_embeds
    
    def encode_video_prompt(
        self, 
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_samples_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_samples_per_prompt=num_samples_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
        negative_prompt_embeds = self._get_t5_prompt_embeds(
            prompt=negative_prompt,
            num_samples_per_prompt=num_samples_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        # prompts_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds, negative_prompt_embeds
                
    def encode_audio_prompt(
        self, 
        prompt: Union[str, List[str]], 
        device: Optional[torch.device] = None, 
        do_classifier_free_guidance: bool=True, 
        negative_prompt: Union[str, List[str]]=None, 
        prompt_embeds: Optional[torch.Tensor] = None, 
        negative_prompt_embeds: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        negative_attention_mask: Optional[torch.Tensor] = None,  
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        self.audio_text_encoder.to(device)
        if prompt_embeds is None:
            # 1. Tokenize text
            text_inputs = self.audio_tokenizer(
                prompt,
                padding="max_length",
                max_length=self.audio_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = self.audio_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.audio_tokenizer.batch_decode(
                    untruncated_ids[:, self.audio_tokenizer.model_max_length - 1 : -1]
                )
                print(
                    f"The following part of your input was truncated because {self.audio_text_encoder.config.model_type} can "
                    f"only handle sequences up to {self.audio_tokenizer.model_max_length} tokens: {removed_text}"
                )

            text_input_ids = text_input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # 2. Text encoder forward
            self.audio_text_encoder.eval()
            prompt_embeds = self.audio_text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if do_classifier_free_guidance and negative_prompt is not None:
            uncond_tokens: List[str]
            if type(prompt) is not type(negative_prompt):
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

            # 1. Tokenize text
            uncond_input = self.audio_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.audio_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            negative_attention_mask = uncond_input.attention_mask.to(device)

            # 2. Text encoder forward
            self.audio_text_encoder.eval()
            negative_prompt_embeds = self.audio_text_encoder(
                uncond_input_ids,
                attention_mask=negative_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

            if negative_attention_mask is not None:
                # set the masked tokens to the null embed
                negative_prompt_embeds = torch.where(
                    negative_attention_mask.to(torch.bool).unsqueeze(2), negative_prompt_embeds, 0.0
                )

        # 3. Project prompt_embeds and negative_prompt_embeds
        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the negative and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if attention_mask is not None and negative_attention_mask is None:
                negative_attention_mask = torch.ones_like(attention_mask)
            elif attention_mask is None and negative_attention_mask is not None:
                attention_mask = torch.ones_like(negative_attention_mask)

            if attention_mask is not None:
                attention_mask = torch.cat([negative_attention_mask, attention_mask])

        prompt_embeds = self.audio_projection_model(
            text_hidden_states=prompt_embeds,
        ).text_hidden_states
        if attention_mask is not None:
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)
        return prompt_embeds
    
    def denoising(self):
        pass

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.video_pipeline_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.video_pipeline_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        accepts_eta = "eta" in set(inspect.signature(self.audio_pipeline_scheduler.step).parameters.keys())
        extra_step_kwargs_a = {}
        if accepts_eta:
            extra_step_kwargs_a["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.audio_pipeline_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs_a["generator"] = generator

        return extra_step_kwargs, extra_step_kwargs_a
    
    @torch.no_grad()
    def encode_duration(
        self,
        audio_start_in_s,
        audio_end_in_s,
        device,
        do_classifier_free_guidance,
        batch_size,
    ):
        audio_start_in_s = audio_start_in_s if isinstance(audio_start_in_s, list) else [audio_start_in_s]
        audio_end_in_s = audio_end_in_s if isinstance(audio_end_in_s, list) else [audio_end_in_s]

        if len(audio_start_in_s) == 1:
            audio_start_in_s = audio_start_in_s * batch_size
        if len(audio_end_in_s) == 1:
            audio_end_in_s = audio_end_in_s * batch_size

        # Cast the inputs to floats
        audio_start_in_s = [float(x) for x in audio_start_in_s]
        audio_start_in_s = torch.tensor(audio_start_in_s).to(device)

        audio_end_in_s = [float(x) for x in audio_end_in_s]
        audio_end_in_s = torch.tensor(audio_end_in_s).to(device)
        self.audio_projection_model.eval()
        projection_output = self.audio_projection_model(
            start_seconds=audio_start_in_s,
            end_seconds=audio_end_in_s,
        )
        seconds_start_hidden_states = projection_output.seconds_start_hidden_states
        seconds_end_hidden_states = projection_output.seconds_end_hidden_states

        # For classifier free guidance, we need to do two forward passes.
        # Here we repeat the audio hidden states to avoid doing two forward passes
        if do_classifier_free_guidance:
            seconds_start_hidden_states = torch.cat([seconds_start_hidden_states, seconds_start_hidden_states], dim=0)
            seconds_end_hidden_states = torch.cat([seconds_end_hidden_states, seconds_end_hidden_states], dim=0)
        
        return seconds_start_hidden_states, seconds_end_hidden_states

class JointDiT_T2AV(BasePipeline):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.strict_loading = False
        # video hidden state torch.Size([B, 32760, 1536]),  audio hidden state (B, 1025, 1536)
        self.dual_block_nums = self.model_archi_config["dual_dit_configs"].get('num_dual_dit_blocks', 3)
        self.num_channels_1 = self.model_archi_config["dual_dit_configs"].get('num_input_channels_1', 1536)
        self.num_channels_2 = self.model_archi_config["dual_dit_configs"].get('num_input_channels_2', 1536)
        self.num_qk_channels = self.model_archi_config["dual_dit_configs"].get('num_qk_channels', 1536)
        self.num_v_channels = self.model_archi_config["dual_dit_configs"].get('num_v_channels', 1536)
        self.num_heads = self.model_archi_config["dual_dit_configs"].get('num_heads', 12)
        self.t_emb_dim_1 = self.model_archi_config["dual_dit_configs"].get('t_emb_dim_1', 9216)
        self.t_emb_dim_2 = self.model_archi_config["dual_dit_configs"].get('t_emb_dim_2', 1536)
        self.use_lora = self.train_config['lora_config'].get('use_lora', True)
        self.fusion_type = self.train_config['fusion_type']
        if self.use_lora:
            self.lora_rank = self.train_config['lora_config'].get('lora_rank', 128)
            self.lora_alpha = self.train_config['lora_config'].get('lora_alpha', 64)
            self.lora_target_modules = self.train_config['lora_config'].get('lora_target_modules', ["to_q", "to_k", "to_v", "to_out.0"])
        self.dual_dit_blocks = nn.ModuleList([])
        if self.fusion_type == "full_attn":
            for i in range(self.dual_block_nums):
                self.dual_dit_blocks.append(
                    DualDiTBlockAdaLN(
                        self.num_channels_1, 
                        self.num_channels_2,
                        self.num_qk_channels,
                        self.num_v_channels,
                        self.num_heads,
                        self.t_emb_dim_1,
                        self.t_emb_dim_2,
                    )
                )
        elif self.fusion_type == "bicross" or self.fusion_type == "v2a" or self.fusion_type == "a2v":
            for i in range(self.dual_block_nums):
                self.dual_dit_blocks.append(
                    CrossDiTBlockAdaLN(
                        self.num_channels_1, 
                        self.num_channels_2,
                        self.num_qk_channels,
                        self.num_v_channels,
                        self.num_heads,
                        self.t_emb_dim_1,
                        self.t_emb_dim_2,
                        self.fusion_type,
                    )
                )
        else:
            for i in range(self.dual_block_nums):
                self.dual_dit_blocks.append(
                    AddFusionDiTBlockAdaLN(
                        self.num_channels_1,
                        self.num_channels_2,
                        self.t_emb_dim_1,
                        self.t_emb_dim_2,
                    )
                )
        self.init_weight()
        # Video Tensor (B, 32760, 1536). Audio Tensor (B, 1025, 1536).
        self.rope_pos_embeds_1d_video = get_1d_rotary_pos_embed(self.num_qk_channels//self.num_heads, 32760, use_real=True,)
        # self.rope_pos_embeds_1d_audio = get_1d_rotary_pos_embed(self.num_qk_channels//self.num_heads, 1025, use_real=True,)
        self.rope_pos_embeds_1d_audio = get_1d_rotary_pos_embed(self.num_qk_channels//self.num_heads, 117, use_real=True,)

        self.rope_pos_embeds = get_1d_rotary_pos_embed(self.num_qk_channels//self.num_heads, 32877, use_real=True,)
        # Freeze.
        self.video_frame_transform = transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x/255 * 2 - 1),
            torchvision.transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            torchvision.transforms.Resize((480, 832), antialias=True),  
        ])
        self.duration = 5.40
        self.audio_process_config = {
            "sampling_rate": 44100,
            "min_waveform_length": 100,
        }
        self.video_process_config = {
            "target_frame_nums": 81,
            "target_fps": 15,
            "vae_scale_factor": 4,
            "video_h": 480,
            "video_w": 832,
        }
        for _module in [self.video_vae, self.video_diffusion, self.audio_vae, self.audio_diffusion, self.audio_projection_model]:
            _module.requires_grad_(False)
        if not self.use_condition_embedding_cache:
            self.video_text_encoder.requires_grad_(False)
            self.audio_text_encoder.requires_grad_(False)
        if self.use_lora:
            video_diffusion_lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                init_lora_weights=True,
                target_modules=self.lora_target_modules,
            )
            self.video_diffusion.add_adapter(video_diffusion_lora_config)
            cast_training_params(self.video_diffusion, dtype=torch.bfloat16)
            audio_diffusion_lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                init_lora_weights=True,
                target_modules=self.lora_target_modules,
            )
            self.audio_diffusion.add_adapter(audio_diffusion_lora_config)
            cast_training_params(self.audio_diffusion, dtype=torch.bfloat16)
            print("Set lora Successfully!")

        if not self.use_lora:
            train_layer_config = self.train_config.get('train_layer_config', {})
            video_train_num = train_layer_config.get('video_train_num', 3)
            audio_train_num = train_layer_config.get('audio_train_num', 3)
            train_diff_layers = train_layer_config.get('train_diff_layers', True)
            for idx, blk in enumerate(self.video_diffusion.blocks):
                blk.requires_grad_(idx < video_train_num)
            for idx, blk in enumerate(self.audio_diffusion.transformer_blocks):
                blk.requires_grad_(idx < audio_train_num)
        # PL training setting
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.unconditional_prob = self.train_config.get('unconditional_prob', 0.1)
        assert 0.0 <= self.unconditional_prob <= 1.0, "Unconditional_prob should be in [0.0, 1.0]"
        
        # Initialize a logger for the pipeline
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        
        # Create a file handler to log to a file
        file_handler = logging.FileHandler('logs/train.log')
        file_handler.setLevel(logging.INFO)
        
        # Create a console handler to log to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)
        self.save_hyperparameters()
        self.video_diffusion.to(self.weight_dtype)
        self.audio_diffusion.to(self.weight_dtype)
        self.audio_projection_model.to(self.weight_dtype)
        self.dual_dit_blocks.to(self.weight_dtype)
        self.video_vae.to(self.weight_dtype)
        self.audio_vae.to(self.weight_dtype)
        if not self.use_condition_embedding_cache:
            self.video_text_encoder.to(self.weight_dtype)
            self.audio_text_encoder.to(self.weight_dtype)
        self.pre_process = self.train_config['cache_config'].get('pre_process', False)
        weight_dtype_str = self.train_config.get('weight_dtype', 'torch.bfloat16')

        # 将字符串转换为真正的 torch.dtype 对象
        self.weight_dtype = getattr(torch, weight_dtype_str.split('.')[-1])
        self.v_segments, self.a_segments = [], []
        self.video_bridge_points = self.model_archi_config['video_diffusion_configs'].get('video_bridge_points', [7, 11, 15])
        self.audio_bridge_points = self.model_archi_config['video_diffusion_configs'].get('audio_bridge_points', [5, 8, 11])
        
        last_split_point = -1
        for point in self.video_bridge_points:
            segment = (last_split_point + 1, point + 1)
            self.v_segments.append(segment)
            last_split_point = point
        final_segment = (last_split_point + 1, len(self.video_diffusion.blocks))
        self.v_segments.append(final_segment)

        last_split_point = -1
        for point in self.audio_bridge_points:
            segment = (last_split_point + 1, point + 1)
            self.a_segments.append(segment)
            last_split_point = point
           
        final_segment = (last_split_point + 1, len(self.audio_diffusion.transformer_blocks))
        self.a_segments.append(final_segment)
        print(f"Video segments: {self.v_segments}")
        print(f"Audio segments: {self.a_segments}")

        use_deepspeed = self.train_config.get('deepspeed_config', {}).get('use_deepspeed', False)
        if use_deepspeed:
            deepspeed.utils.set_z3_leaf_modules(self, [AutoencoderKLWan])
            print("Successfully set VAE as a leaf module for DeepSpeed ZeRO-3.")
        # self.set_trainable_layers()

    def set_trainable_layers(self): 
        for idx, blk in enumerate(self.video_diffusion.blocks):
            blk.requires_grad_(idx in self.video_bridge_points)
        for idx, blk in enumerate(self.audio_diffusion.transformer_blocks):
            blk.requires_grad_(idx in self.audio_bridge_points)
        print(f"Video trainable layers: {self.video_bridge_points}")
        print(f"Audio trainable layers: {self.audio_bridge_points}")

    def init_from_ckpt(self, ckpt_path, ignore_keys=list()):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("=> Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=True)
        print(f"=> Restored T2AV model from {ckpt_path}")

    def init_weight(self):
        for attn_block in self.dual_dit_blocks:
            if not isinstance(attn_block, AddFusionDiTBlockAdaLN):
                torch.nn.init.xavier_uniform_(attn_block.attn.q_proj_1.weight)
                torch.nn.init.xavier_uniform_(attn_block.attn.k_proj_1.weight)
                torch.nn.init.xavier_uniform_(attn_block.attn.v_proj_1.weight)
                torch.nn.init.constant_(attn_block.attn.o_proj_1.weight , 0)
                torch.nn.init.xavier_uniform_(attn_block.attn.q_proj_2.weight)
                torch.nn.init.xavier_uniform_(attn_block.attn.k_proj_2.weight)
                torch.nn.init.xavier_uniform_(attn_block.attn.v_proj_2.weight)
                torch.nn.init.constant_(attn_block.attn.o_proj_2.weight , 0)
                if attn_block.attn.q_proj_1.bias is not None:
                    nn.init.constant_(attn_block.attn.q_proj_1.bias , 0)
                if attn_block.attn.k_proj_1.bias is not None:
                    nn.init.constant_(attn_block.attn.k_proj_1.bias , 0)
                if attn_block.attn.v_proj_1.bias is not None:
                    nn.init.constant_(attn_block.attn.v_proj_1.bias , 0)
                if attn_block.attn.o_proj_1.bias is not None:
                    nn.init.constant_(attn_block.attn.o_proj_1.bias , 0)
                if attn_block.attn.q_proj_2.bias is not None:
                    nn.init.constant_(attn_block.attn.q_proj_2.bias , 0)
                if attn_block.attn.k_proj_2.bias is not None:
                    nn.init.constant_(attn_block.attn.k_proj_2.bias , 0)
                if attn_block.attn.v_proj_2.bias is not None:
                    nn.init.constant_(attn_block.attn.v_proj_2.bias , 0)
                if attn_block.attn.o_proj_2.bias is not None:
                    nn.init.constant_(attn_block.attn.o_proj_2.bias , 0)
            if attn_block.mlp_1 is not None and attn_block.mlp_2 is not None:
                nn.init.constant_(attn_block.mlp_1.fc2.weight , 0)
                nn.init.constant_(attn_block.mlp_1.fc2.bias , 0)
                nn.init.constant_(attn_block.mlp_2.fc2.weight , 0)
                nn.init.constant_(attn_block.mlp_2.fc2.bias , 0)
    
    def configure_optimizers(self):
        lr = self.train_config.get('lr', 1e-4)
        weight_decay = self.train_config.get('weight_decay', 0.0)
        use_deepspeed = self.train_config.get('deepspeed_config', {}).get('use_deepspeed', False)
        if use_deepspeed:
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=lr, 
                betas=self.train_config.get('betas', [0.9, 0.95]),
                weight_decay=weight_decay
            )
        
        # Scheduler. 
        def fn(warmup_steps, step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return 1.0
        def linear_warmup_decay(warmup_steps):
            return partial(fn, warmup_steps)
        lr_warmup_steps = self.train_config.get('lr_warmup_steps', 1000)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(lr_warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }
        return ([optimizer], [scheduler])

    def on_save_checkpoint(self, checkpoint):
        # Save the Checkpoint and the adapters
        adapter_state = {}
        # 获取当前模型中所有 requires_grad=True 的参数和参数值
        video_trainable_param_values = {
            name: param.data for name, param in self.video_diffusion.named_parameters()
            if param.requires_grad
        }
        audio_trainable_param_values = {
            name: param.data for name, param in self.audio_diffusion.named_parameters()
            if param.requires_grad
        }
        
        checkpoint["state_dict"] = {**video_trainable_param_values, **audio_trainable_param_values, **self.dual_dit_blocks.state_dict()}

    def on_load_checkpoint(self, checkpoint):
        # load the checkpoint and the adapter
        missing, unexpected = self.video_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Missing keys video_adapter_state: {missing}")
        print(f"Unexpected keys video_adapter_state: {unexpected}")
        missing, unexpected = self.audio_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Missing keys audio_adapter_state: {missing}")
        print(f"Unexpected keys audio_adapter_state: {unexpected}")
        missing, unexpected = self.dual_dit_blocks.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Missing keys dual_dit_state: {missing}")
        print(f"Unexpected keys dual_dit_state: {unexpected}")

    def get_input(self, batch):
        if self.pre_process:
            if self.use_condition_embedding_cache:
                waveform, video_frame, video_prompt_embeds, audio_prompt_embeds, video_id = batch["waveform"], batch["video_frame"], batch["video_emb"], batch["audio_emb"], batch["video_id"]
                return video_frame, waveform, video_prompt_embeds, audio_prompt_embeds, video_id
            else:
                waveform, video_frame, text_prompt, video_id = batch["waveform"], batch["video_frame"], batch["text_prompt"], batch["video_id"]
                # Type convert
                waveform = waveform.to(dtype=self.weight_dtype)
                video_frame = video_frame.to(dtype=self.weight_dtype)
                return video_frame, waveform, text_prompt, video_id
        else:
            video_path, video_prompt, audio_prompt = batch["video_path"], batch["video_caption"], batch["audio_caption"]
            return video_path, video_prompt, audio_prompt

    def _gradient_checkpointing_func(self, module, *inputs):
        def custom_forward(*inputs):
            return module(*inputs)
        return checkpoint.checkpoint(custom_forward, *inputs)

    def forward_2(self, v_in_kwargs, a_in_kwargs):
        v_out = self.video_diffusion.forward(**v_in_kwargs)

        a_out = self.audio_diffusion.forward(**a_in_kwargs)
        return v_out, a_out

    # def forward(self, v_in_kwargs, a_in_kwargs, full_rope, full_rope2):
    #     self.video_diffusion.to(self.weight_dtype)
    #     self.audio_diffusion.to(self.weight_dtype)
    #     v_h, v_enc_h, v_time_proj, v_temb, v_rotary_emb, write_shape = self.video_diffusion.forward_1(**v_in_kwargs)
    #     a_h, a_attn_mask, a_crs_h, a_time_emb, a_rotary_emb = self.audio_diffusion.forward_1(**a_in_kwargs)
    #     self.dual_dit_blocks.to(self.weight_dtype)     
    #     # v_time_emb = v_time_proj.view(v_time_proj.shape[0],-1)
    #     v_time_emb = v_temb
    #     for i, v_block in enumerate(self.video_diffusion.blocks):
    #         if 0 <= i < 8:
    #             v_h = self._gradient_checkpointing_func(v_block, v_h, v_enc_h, v_time_proj, v_rotary_emb)
    #     for i, a_block in enumerate(self.audio_diffusion.transformer_blocks):
    #         if 0 <= i < 6:
    #             a_h = self._gradient_checkpointing_func(a_block, a_h, a_attn_mask, a_crs_h, None, a_rotary_emb)
        
    #     v_h, a_h = self.dual_dit_blocks[0](v_h, a_h, v_time_emb, a_time_emb, full_rope)
    #     for i, v_block in enumerate(self.video_diffusion.blocks):
    #         if 8 <= i < 12:
    #             v_h = self._gradient_checkpointing_func(v_block, v_h, v_enc_h, v_time_proj, v_rotary_emb)
    #     for i, a_block in enumerate(self.audio_diffusion.transformer_blocks):
    #         if 6 <= i < 9:
    #             a_h = self._gradient_checkpointing_func(a_block, a_h, a_attn_mask, a_crs_h, None, a_rotary_emb)
    #     v_h, a_h = self.dual_dit_blocks[1](v_h, a_h, v_time_emb, a_time_emb, full_rope)

    #     for i, v_block in enumerate(self.video_diffusion.blocks):
    #         if 12 <= i < 16:
    #             v_h = self._gradient_checkpointing_func(v_block, v_h, v_enc_h, v_time_proj, v_rotary_emb)
    #     for i, a_block in enumerate(self.audio_diffusion.transformer_blocks):
    #         if 9 <= i < 12:
    #             a_h = self._gradient_checkpointing_func(a_block, a_h, a_attn_mask, a_crs_h, None, a_rotary_emb)
    #     v_h, a_h = self.dual_dit_blocks[2](v_h, a_h, v_time_emb, a_time_emb, full_rope)
    #     for i, v_block in enumerate(self.video_diffusion.blocks):
    #         if 16 <= i:
    #             v_h = self._gradient_checkpointing_func(v_block, v_h, v_enc_h, v_time_proj, v_rotary_emb)
    #     for i, a_block in enumerate(self.audio_diffusion.transformer_blocks):
    #         if 12 <= i:
    #             a_h = self._gradient_checkpointing_func(a_block, a_h, a_attn_mask, a_crs_h, None, a_rotary_emb)
    #     v_out = self.video_diffusion.forward_2(v_h, v_temb, write_shape, False)
    #     a_out = self.audio_diffusion.forward_2(a_h, False)

    #     return v_out, a_out

    def forward(self, v_in_kwargs, a_in_kwargs, rope_pos_emb_1, rope_pos_emb_2):
        # 1. 设置数据类型并从各个塔获取初始隐藏状态
        self.video_diffusion.to(self.weight_dtype)
        self.audio_diffusion.to(self.weight_dtype)
        self.dual_dit_blocks.to(self.weight_dtype)

        v_h, v_enc_h, v_time_proj, v_temb, v_rotary_emb, write_shape = self.video_diffusion.forward_1(**v_in_kwargs)
        a_h, a_attn_mask, a_crs_h, a_time_emb, a_rotary_emb = self.audio_diffusion.forward_1(**a_in_kwargs)
        
        #v_time_emb = v_time_proj.view(v_time_proj.shape[0],-1)
        v_time_emb = v_temb
        # 2. 迭代处理每个塔的分段并应用桥接层
        # 这个循环的次数由桥接模型的层数决定
        for i, bridge_block in enumerate(self.dual_dit_blocks):
            # --- 处理视频分段 ---
            v_start, v_end = self.v_segments[i]
            for block_idx in range(v_start, v_end):
                v_block = self.video_diffusion.blocks[block_idx]
                v_h = self._gradient_checkpointing_func(v_block, v_h, v_enc_h, v_time_proj, v_rotary_emb)

            # --- 处理音频分段 ---
            a_start, a_end = self.a_segments[i]
            for block_idx in range(a_start, a_end):
                a_block = self.audio_diffusion.transformer_blocks[block_idx]
                a_h = self._gradient_checkpointing_func(a_block, a_h, a_attn_mask, a_crs_h, None, a_rotary_emb)
            
            # --- 应用第 i 个桥接层 ---
            v_h, a_h = bridge_block(v_h, a_h, v_time_emb, a_time_emb, rope_pos_emb_1, rope_pos_emb_2)

        # 3. 处理最后一个桥接层之后剩余的最终分段
        # --- 处理视频最终分段 ---
        v_start, v_end = self.v_segments[-1]
        for block_idx in range(v_start, v_end):
            v_block = self.video_diffusion.blocks[block_idx]
            v_h = self._gradient_checkpointing_func(v_block, v_h, v_enc_h, v_time_proj, v_rotary_emb)

        # --- 处理音频最终分段 ---
        a_start, a_end = self.a_segments[-1]
        for block_idx in range(a_start, a_end):
            a_block = self.audio_diffusion.transformer_blocks[block_idx]
            a_h = self._gradient_checkpointing_func(a_block, a_h, a_attn_mask, a_crs_h, None, a_rotary_emb)

        # 4. 通过各自的最终层（forward_2）得到输出
        v_out = self.video_diffusion.forward_2(v_h, v_temb, write_shape, False)
        a_out = self.audio_diffusion.forward_2(a_h, False)

        return v_out, a_out

    def forward_video(self, hidden_states, encoder_hidden_states, timestep, return_dict=False):
       v_h, v_enc_h, v_emb, temb_v ,v_rotary_emb, write_shape  = self.video_diffusion.forward_1(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, return_dict=return_dict)
       for i, v_block in enumerate(self.video_diffusion.blocks):
            v_h = self._gradient_checkpointing_func(v_block, v_h, v_enc_h, v_emb, v_rotary_emb)
       v_out = self.video_diffusion.forward_2(v_h, temb_v, write_shape, False)[0]
       return v_out

    def process_video(self, video_paths):
        video_frames = []
        for video_path in video_paths:
            video_tuple = torchvision.io.read_video(video_path, pts_unit='sec')
            video_raw_frame_num = len(video_tuple[0])
            video_raw_fps = video_tuple[2]['video_fps']
            video_raw_duration = video_raw_frame_num / video_raw_fps
            video_tensor = video_tuple[0]
            video_tensor = self.video_frame_transform(video_tensor)         
            ''' Sample frames as the same as the audio part and Pad'''
            if video_raw_frame_num <= self.video_process_config["target_frame_nums"]:
                temp_video = video_tensor.repeat(10, 1, 1, 1)
                sample_videos = temp_video[:self.video_process_config["target_frame_nums"]]
            else:
                sample_videos = video_tensor[:self.video_process_config["target_frame_nums"]]
            video_frames.append(sample_videos.permute(1, 0, 2, 3))
        video_frames = torch.stack(video_frames, dim=0)
        return video_frames
    
    def process_audio(self, audio_paths):
        waveforms = []
        for audio_path in audio_paths:
            waveform, sr = torchaudio.load(audio_path)
            waveform = torchaudio.functional.resample(waveform, sr, self.audio_process_config["sampling_rate"])
            waveform = waveform[0, ...].float()
            # Validate the waveform length
            waveform_length = waveform.shape[-1]
            ''' Random segment and pad '''
            target_length = int(self.audio_process_config["sampling_rate"] * self.duration)
            if waveform_length <= target_length:
                temp_wav = waveform.repeat(10)
                sample_waveforms = temp_wav[:target_length]
            else: 
                cur_waveform_1 = waveform[:target_length]
                sample_waveforms = cur_waveform_1
            waveforms.append(sample_waveforms)
        waveforms = torch.stack(waveforms, dim=0)
        return waveforms
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
    def retrieve_latents(
        self, encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
    ):
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def training_step(self, batch, batch_idx, dataloader_idx=0, custom_device=None):
        # input process
        if self.pre_process:
            if self.use_condition_embedding_cache:
                video_frame, waveform, video_prompt_embeds, audio_prompt_embeds, video_id = self.get_input(batch)
            else:
                video_frame, waveform, text_prompt, video_id = self.get_input(batch)
        else:
            video_path, video_prompt, audio_prompt = self.get_input(batch)
            video_frame = self.process_video(video_path)
            waveform = self.process_audio(video_path)
        batch_size = waveform.shape[0]
        device = self.model_device
        
        self.audio_vae.to(device, self.weight_dtype)
        self.video_vae.to(device, self.weight_dtype)
        
        if self.inference_seed is not None:
            rand_seed = self.inference_seed
        else:
            rand_seed = torch.randint(0, 100000, (1,),).item()
        generator = torch.Generator(device=device).manual_seed(rand_seed)
        video_frame = video_frame.to(device, self.weight_dtype)
        with torch.no_grad():
            video_latents = self.retrieve_latents(self.video_vae.encode(video_frame), generator, 'sample').unbind(0)
        del video_frame
        video_latents_list = []
        for i in range(len(video_latents)):
            video_latents_list.append(video_latents[i].unsqueeze(0))
        video_latents = torch.cat(video_latents_list, dim=0)
        ## 2. Normalize video latents
        video_latents_mean = (
            torch.tensor(self.video_vae.config.latents_mean)
            .view(1, self.video_vae.config.z_dim, 1, 1, 1)
            .to(video_latents.device, video_latents.dtype)
        )
        video_latents_std = 1.0 / torch.tensor(self.video_vae.config.latents_std).view(1, self.video_vae.config.z_dim, 1, 1, 1).to(
            video_latents.device, video_latents.dtype
        )
        video_latents = (video_latents - video_latents_mean) * video_latents_std
        video_latents = video_latents.detach()
        ## 3. Encode Text prompt
        if not self.use_condition_embedding_cache:
            video_prompt_embeds, video_negative_prompt_embeds = self.encode_video_prompt(
                prompt=video_prompt,
                negative_prompt=[self.negative_prompt]*batch_size,
                do_classifier_free_guidance=True,
                num_samples_per_prompt=1,
            )
        # self.video_vae.to("cpu")
        # self.video_text_encoder.to("cpu")
        ## 4. Sample a random timestep for each sample
        t = self.rng.draw(batch_size)[:, 0].to(self.model_device, self.weight_dtype)
        video_timesteps = t * (self.video_pipeline_scheduler.config.num_train_timesteps - 1)
        video_timesteps = video_timesteps.long()
        ## 5. Add noise to video latents
        video_noise = torch.randn_like(video_latents, dtype=self.weight_dtype)
        # video_latents_noisy = self.video_pipeline_scheduler.add_noise(video_latents, video_noise, video_timesteps)
        video_sigma = (video_timesteps / 1000).to(video_latents.dtype).view(-1, 1, 1, 1, 1)
        video_latents_noisy = (1 - video_sigma) * video_latents + video_noise * video_sigma
        '''For Audio part'''
        ## 1. Convert wav to latent space 
        audio_vae_weight_dtype = self.audio_vae.encoder.conv1.weight.dtype
        # change waveform size from (B, T) to (B, 2, T) two channels
        waveform = waveform.unsqueeze(1)
        waveform = torch.cat([waveform, waveform], dim=1)
        with torch.no_grad():
            self.audio_vae.to(self.weight_dtype)
            audio_latents = self.audio_vae.encode(waveform.to(self.weight_dtype).to(device)).latent_dist.sample()
        del waveform
        audio_latents = audio_latents.detach()
        ##  2. Sample noise, sample a random timestep, add noise to the latents 
        def get_alphas_sigmas(t):
            """Returns the scaling factors for the clean image (alpha) and for the
            noise (sigma), given a timestep."""
            return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        audio_noise = torch.randn_like(audio_latents)
        # audio_latents_noisy = self.audio_pipeline_scheduler.add_noise(audio_latents, audio_noise, t)
        audio_latents_noisy = audio_latents * alphas + audio_noise * sigmas
        audio_target = audio_noise * alphas - audio_latents * sigmas
        ## 3. Get the embedding for conditioning        
        if not self.use_condition_embedding_cache:
            audio_prompt_embeds = self.encode_audio_prompt(
                prompt=audio_prompt,
                device=device,
                do_classifier_free_guidance=False,
                negative_prompt=[self.negative_prompt]*batch_size,
            )
        # self.audio_text_encoder.to("cpu")
        # self.audio_vae.to("cpu")
        ## 4. Encode duration
        seconds_start_hidden_states, seconds_end_hidden_states = self.encode_duration(
            0.0,
            self.sample_audio_length_s,
            device,
            False,
            batch_size,
        )

        audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2)

        # 4. RoPE for audio
        audio_rotary_embed_dim = self.audio_diffusion.config.attention_head_dim // 2
        audio_rotary_embedding = get_1d_rotary_pos_embed(
            audio_rotary_embed_dim,
            audio_latents_noisy.shape[2] + audio_duration_embeds.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        ''' Forward and Loss'''
        ## 1. Forward input
        video_input_kwargs = {
            "hidden_states": video_latents_noisy.to(self.weight_dtype),
            "encoder_hidden_states": video_prompt_embeds.to(self.weight_dtype),
            "timestep": video_timesteps,
            "return_dict": False,

        }

        audio_input_kwargs = {
            "hidden_states": audio_latents_noisy.to(self.weight_dtype),
            "timestep": t,
            "encoder_hidden_states": audio_prompt_embeds.to(self.weight_dtype),
            "global_hidden_states": audio_duration_embeds.to(self.weight_dtype),
            "rotary_embedding": audio_rotary_embedding,
        }
        # video_pre, audio_pre = self.forward(video_input_kwargs, audio_input_kwargs, self.rope_pos_embeds)
        video_pre, audio_pre = self.forward(
            video_input_kwargs, 
            audio_input_kwargs,
            self.rope_pos_embeds_1d_video,
            self.rope_pos_embeds_1d_audio
            # self.rope_pos_embeds, 
            # None
        )
        video_pre = video_pre[0].to(self.weight_dtype)
        audio_pre = audio_pre[0].to(self.weight_dtype)
        ## 2. Loss for audio    
        loss_audio = nn.functional.mse_loss(audio_pre, audio_target, reduction="mean")      
        ## 3. Loss for video
        video_target = video_noise - video_latents
        loss_video = nn.functional.mse_loss(video_pre, video_target, reduction="mean")
        loss = loss_video + loss_audio
        ## 4. check Loss Nan
        has_nan_loss = torch.isnan(loss).any().item()
        assert not has_nan_loss, "Loss has nan value!"
        ## 5. Log
        print(loss.item(), loss_video.item(), loss_audio.item())
        assert loss.dim() == 0
        del video_latents_noisy, video_noise, video_latents, video_prompt_embeds, audio_prompt_embeds, audio_latents_noisy, audio_noise, audio_latents, audio_duration_embeds, audio_rotary_embedding
        torch.cuda.empty_cache()
        # loss = loss.to(self.weight_dtype)
        # print(loss.dtype)
        #self._logger.info(f"Loss: {loss.item()}, Loss_video: {loss_video.item()}, Loss_audio: {loss_audio.item()}")
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_video", loss_video, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_audio", loss_audio, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss
    
    """Inference Process"""
    def on_predict_epoch_start(self):
        video_scheduler_config = self.config.get("video_scheduler_config", None)
        audio_pipeline_scheduler_config = self.config.get("audio_pipeline_scheduler_config", None)
        self.video_scheduler = get_huggingface_model_from_local_dir(video_scheduler_config, CogVideoXDPMScheduler)
        self.audio_pipeline_scheduler = get_huggingface_model_from_local_dir(audio_pipeline_scheduler_config, CosineDPMSolverMultistepScheduler)
        
    def predict_step(self, video_prompt, audio_prompt, video_negative_prompt, audio_negative_prompt, save_name='test'):
        batch_size = 1
        if self.inference_seed is not None:
            rand_seed = self.inference_seed
        else:
            rand_seed = torch.randint(0, 100000, (1,),).item()
        generator = torch.Generator(device=self.model_device).manual_seed(rand_seed)
        if video_negative_prompt is None:
            video_negative_prompt = ["Low quality."]
        if audio_negative_prompt is None:
            audio_negative_prompt = ["Low quality."]
        self.video_text_encoder.to(self.model_device)
        self.audio_text_encoder.to(self.model_device)
        self.audio_projection_model.to(self.model_device)
        self.video_diffusion.to(self.model_device)
        self.audio_diffusion.to(self.model_device)
        self.dual_dit_blocks.to(self.model_device)
        self.video_vae.to(self.model_device)
        self.audio_vae.to(self.model_device)
        video_tensor, audio_tensor = self.denoising(
            video_prompt_text=video_prompt,
            audio_prompt_text=audio_prompt,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt,
            video_height=self.sample_video_height,
            video_width=self.sample_video_width,
            video_num_frames=self.sample_video_num_frames,
            generator=generator,
        )
        # video_tensor shape (batch_size, frame_nums, height, width, channel)
        # Log video
        for sample_idx in range(self.num_samples_per_prompt):
            cur_video = video_tensor[sample_idx]
            cur_video_path = os.path.join(self.video_save_path, "{}.mp4".format(save_name))
            video_np = cur_video  # e.g. shape (T, H, W, C)
            video_np = (video_np * 255).astype(np.uint8)  # 转成uint8，范围0-255，moviepy需要
            fps = self.sample_video_fps
            duration = video_np.shape[0] / fps
            def make_frame(t):
                frame_index = min(int(t * fps), video_np.shape[0] - 1)
                return video_np[frame_index]
            video_clip = VideoClip(make_frame, duration=duration)
            # 2. 处理音频数据：audio_tensor[0] 维度为 (channels, samples)
            audio_np = audio_tensor[sample_idx].squeeze(0).T.float().cpu().numpy()  # 转成 numpy, shape (samples, channels)
            sampling_rate = self.audio_vae.sampling_rate
            audio_clip = AudioArrayClip(audio_np, fps=sampling_rate)
            # 3. 合成音视频
            video_clip = video_clip.with_audio(audio_clip)
            # 4. 直接导出带音频的视频文件
            video_clip.write_videofile(cur_video_path, fps=fps, codec="libx264", audio_codec="aac")

        return video_tensor, audio_tensor

    @torch.no_grad()
    def denoising(
        self,
        video_prompt_text: Union[str, List[str]] = None,
        audio_prompt_text: Union[str, List[str]] = None,
        video_negative_prompt: Union[str, List[str]] = None,
        audio_negative_prompt: Union[str, List[str]] = None,
        video_height: Optional[int] = None,
        video_width: Optional[int] = None,
        video_num_frames: int = 81,
        video_use_dynamic_cfg: bool = False,
        video_latents: Optional[torch.FloatTensor] = None,
        video_attention_kwargs: Optional[Dict[str, Any]] = None,
        num_inference_steps: int = 50,
        num_samples_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_text_max_sequence_length: int = 1024,
        video_guidance_scale: float = 6.0,
        audio_guidance_scale: float = 7,
        first_frame_for_clip: Optional[torch.FloatTensor] = None,
        audio_end_in_s: float = 5.40
    ):
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Convert audio input length from seconds to latent length
        downsample_ratio = self.audio_vae.hop_length
        max_audio_length_in_s = self.audio_diffusion.config.sample_size * downsample_ratio / self.audio_vae.config.sampling_rate
        if audio_end_in_s is None:
            audio_end_in_s = max_audio_length_in_s
        if audio_end_in_s > max_audio_length_in_s:
            raise ValueError(
                f"The total audio length requested ({audio_end_in_s}s) is longer than the model maximum possible length ({max_audio_length_in_s}). Make sure that 'audio_end_in_s-audio_start_in_s<={max_audio_length_in_s}'."
            )

        waveform_start = int(0.0 * self.audio_vae.config.sampling_rate)
        waveform_end = int(audio_end_in_s * self.audio_vae.config.sampling_rate)
        # waveform_length = int(self.audio_diffusion.config.sample_size)
        waveform_length = int((waveform_end - waveform_start)/ downsample_ratio)
        # 1. Check inputs. Raise error if not correct. TODO

        # 2. Default call parameters
        video_do_classifier_free_guidance = video_guidance_scale > 1.0
        audio_do_classifier_free_guidance = audio_guidance_scale > 1.0
        if video_prompt_text is not None and isinstance(video_prompt_text, str):
            batch_size = 1
        elif video_prompt_text is not None and isinstance(video_prompt_text, list):
            batch_size = len(video_prompt_text)
        # 3. Encode input prompt
        video_prompt_embeds, video_negative_prompt_embeds = self.encode_video_prompt(
            prompt=video_prompt_text,
            negative_prompt=video_negative_prompt,
            do_classifier_free_guidance=video_do_classifier_free_guidance,
            num_samples_per_prompt=num_samples_per_prompt,
            max_sequence_length=prompt_text_max_sequence_length,
        )
        # Optional: Concatenate the prompt embeds for classifier free guidance
        video_prompt_embeds = torch.cat([video_negative_prompt_embeds, video_prompt_embeds], dim=0)
        audio_prompt_embeds = self.encode_audio_prompt(
            prompt=audio_prompt_text,
            device=device,
            do_classifier_free_guidance=audio_do_classifier_free_guidance,
            negative_prompt=audio_negative_prompt
        )
        # Encode duration
        seconds_start_hidden_states, seconds_end_hidden_states = self.encode_duration(
            0.0,
            audio_end_in_s,
            device,
            audio_do_classifier_free_guidance and audio_negative_prompt is not None,
            batch_size,
        )
        # Create text_audio_duration_embeds and audio_duration_embeds
        text_audio_duration_embeds = torch.cat(
            [audio_prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
        )

        audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2)

        # In case of classifier free guidance without negative prompt, we need to create unconditional embeddings and
        # to concatenate it to the embeddings
        if audio_do_classifier_free_guidance and audio_negative_prompt is None:
            negative_text_audio_duration_embeds = torch.zeros_like(
                text_audio_duration_embeds, device=text_audio_duration_embeds.device
            )
            text_audio_duration_embeds = torch.cat(
                [negative_text_audio_duration_embeds, text_audio_duration_embeds], dim=0
            )
            audio_duration_embeds = torch.cat([audio_duration_embeds, audio_duration_embeds], dim=0)

        bs_embed, seq_len, hidden_size = text_audio_duration_embeds.shape
        num_waveforms_per_prompt = 1
        # duplicate audio_duration_embeds and text_audio_duration_embeds for each generation per prompt, using mps friendly method
        text_audio_duration_embeds = text_audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
        text_audio_duration_embeds = text_audio_duration_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )

        audio_duration_embeds = audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
        audio_duration_embeds = audio_duration_embeds.view(
            bs_embed * num_waveforms_per_prompt, -1, audio_duration_embeds.shape[-1]
        )

        # 4. Prepare timesteps
        self.video_pipeline_scheduler.set_timesteps(num_inference_steps, device=device)
        video_timesteps = self.video_pipeline_scheduler.timesteps
        self.audio_pipeline_scheduler.set_timesteps(num_inference_steps, device=device)
        audio_timesteps = self.audio_pipeline_scheduler.timesteps
        # 5. Prepare latents
        audio_latents_shape = (
            batch_size * num_samples_per_prompt,
            self.audio_diffusion.config.in_channels,
            waveform_length
        )
        audio_latents = self.prepare_audio_latents(
            audio_latents_shape, 
            audio_prompt_embeds.dtype, 
            device, 
            generator,           
            self.audio_pipeline_scheduler.init_noise_sigma, 
            None,
        )    
        # shape of audio_latents: (batch_size, 64, 1024)
        # shape of audio_prompt_embeds: (batch_size, 128, 768)
        # print(audio_latents.shape)
        # print(audio_prompt_embeds.shape)
        video_latents = self.prepare_video_latents(
            batch_size,
            self.video_diffusion.config.in_channels,
            video_height,
            video_width,
            video_num_frames,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        # 6. Prepare extra step kwargs
        extra_step_kwargs_v, extra_step_kwargs_a = self.prepare_extra_step_kwargs(generator, eta)
        # 7. Create rotary embeds if required
        self.rotary_embed_dim = self.audio_diffusion.config.attention_head_dim // 2
        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            audio_latents.shape[2] + audio_duration_embeds.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        # 8. Denoising loop
        old_pred_original_sample = None

        for i, vt in tqdm(enumerate(video_timesteps), total=len(video_timesteps)):
            # 8.1 Prepare audio latents for the step
            at = audio_timesteps[i]
            audio_latents_model_input = torch.cat([audio_latents] * 2) if audio_do_classifier_free_guidance else audio_latents
            audio_latents_model_input = self.audio_pipeline_scheduler.scale_model_input(audio_latents_model_input, at)
            # 8.2 Prepare video latents for the step
            video_latent_model_input = torch.cat([video_latents] * 2) if video_do_classifier_free_guidance else video_latents
            video_latent_model_input = self.video_pipeline_scheduler.scale_model_input(video_latent_model_input, vt)

            # video_latent_model_input = video_latents.to(self.video_vae.dtype)
            # 8.2 Prepare input kwargs for video and audio diffusion models
            video_timestep = vt.expand(video_latent_model_input.shape[0])
            video_diff_input_kwargs = { 
                "hidden_states": video_latent_model_input,
                "encoder_hidden_states": video_prompt_embeds,
                "timestep": video_timestep,
                "return_dict": False,
            }

            audio_diff_input_kwargs = {
                "hidden_states": audio_latents_model_input,
                "timestep": at.unsqueeze(0),
                "encoder_hidden_states": audio_prompt_embeds,
                "global_hidden_states": audio_duration_embeds,
                "rotary_embedding": rotary_embedding,
                "return_dict": False,
            }

            video_noise_pred, audio_noise_pred = self.forward(
                video_diff_input_kwargs, 
                audio_diff_input_kwargs,
                self.rope_pos_embeds_1d_video,
                self.rope_pos_embeds_1d_audio,
                # self.rope_pos_embeds,
                # None
            )
            # video_noise_pred, audio_noise_pred = self.forward_2(video_diff_input_kwargs, audio_diff_input_kwargs)
            # print(video_noise_pred.shape)
            # print(audio_noise_pred.shape)
            video_noise_pred = video_noise_pred[0]
            audio_noise_pred = audio_noise_pred[0]
            # perform video guidance
            if video_do_classifier_free_guidance:
                video_noise_pred_uncond, video_noise_pred_text = video_noise_pred.chunk(2)
                # video_noise_pred_uncond = self.forward_video(
                #     hidden_states=video_latent_model_input,
                #     encoder_hidden_states=video_negative_prompt_embeds,
                #     timestep=video_timestep,
                #     return_dict=False
                # )[0]
                video_noise_pred = video_noise_pred_uncond + video_guidance_scale * (video_noise_pred_text - video_noise_pred_uncond)       
        
            # perform audio guidance
            if audio_do_classifier_free_guidance:
                audio_noise_pred_uncond, audio_noise_pred_text = audio_noise_pred.chunk(2)
                audio_noise_pred = audio_noise_pred_uncond + audio_guidance_scale * (audio_noise_pred_text - audio_noise_pred_uncond)
            # Shape of Video Noist Pred: (batch_size, 16, 21 , 60, 104)
            # Shape of Audio Noist Pred: (batch_size, 64, 104)
            # compute the previous noisy sample x_t -> x_t-1
            audio_latents = self.audio_pipeline_scheduler.step(audio_noise_pred, at, audio_latents, **extra_step_kwargs_a).prev_sample
            video_latents = self.video_pipeline_scheduler.step(video_noise_pred, vt, video_latents, return_dict=False)[0]
            # Shape of Video Latents: (batch_size, 16, 21 , 60, 104)
            # Shape of Audio Latents: (batch_size, 64, 104)
        # 9. Post-processing
        video_latents = video_latents.to(self.video_vae.dtype)
        audio_latents = audio_latents.to(self.audio_vae.dtype)
        video_latents_mean = (
            torch.tensor(self.video_vae.config.latents_mean)
            .view(1, self.video_vae.config.z_dim, 1, 1, 1)
            .to(video_latents.device, video_latents.dtype)
        )
        video_latents_std = 1.0 / torch.tensor(self.video_vae.config.latents_std).view(1, self.video_vae.config.z_dim, 1, 1, 1).to(
            video_latents.device, video_latents.dtype
        )

        video_latents = video_latents / video_latents_std + video_latents_mean
        video = self.video_vae.decode(video_latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type="np")
        audio = self.audio_vae.decode(audio_latents).sample
        audio = audio[:, :, waveform_start:waveform_end]
        # audio = audio[:, 0, waveform_start:waveform_end]
        return video, audio

    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        state_dict = checkpoint["state_dict"]
        # print(state_dict.keys())
        missing, unexpected = self.video_diffusion.load_state_dict(state_dict, strict=False)
        # print(f"Missing keys video_adapter_state: {missing}")
        # print(f"Unexpected keys video_adapter_state: {unexpected}")
        missing, unexpected = self.audio_diffusion.load_state_dict(state_dict, strict=False)
        # print(f"Missing keys audio_adapter_state: {missing}")
        # print(f"Unexpected keys audio_adapter_state: {unexpected}")
        missing, unexpected = self.dual_dit_blocks.load_state_dict(state_dict, strict=False)
        # print(f"Missing keys dual_dit_state: {missing}")
        # print(f"Unexpected keys dual_dit_state: {unexpected}")
       
    def load_ckpt_deepspeed(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        # print(checkpoint['module'].keys())
        # state_dict = checkpoint["state_dict"]
        state_dict = checkpoint['module']
       
        video_adapter_state = {
            key.replace("video_diffusion.", ""): value
            for key, value in state_dict.items()
            if key.startswith("video_diffusion.")
        }
        video_trainable_param_names = {
            name for name, param in self.video_diffusion.named_parameters()
            if param.requires_grad
        }
        video_filtered_state = {
            key: value for key, value in video_adapter_state.items()
            if key in video_trainable_param_names
        }
        missing, unexpected = self.video_diffusion.load_state_dict(video_filtered_state, strict=False)
        print(f"Missing keys video_adapter_state: {missing}")
        print(f"Unexpected keys video_adapter_state: {unexpected}")
        
        
        
        audio_adapter_state = {
            key.replace("audio_diffusion.", ""): value
            for key, value in state_dict.items()
            if key.startswith("audio_diffusion.")
        }
        audio_trainable_param_names = {
            name for name, param in self.audio_diffusion.named_parameters()
            if param.requires_grad
        }
        audio_filtered_state = {
            key: value for key, value in audio_adapter_state.items()
            if key in audio_trainable_param_names
        }
        missing, unexpected = self.audio_diffusion.load_state_dict(audio_filtered_state, strict=False)
        # print(f"Missing keys audio_adapter_state: {missing}")
        # print(f"Unexpected keys audio_adapter_state: {unexpected}")

        # 修改参数名，去掉前缀（如去掉 "dual_dit_blocks.2."）
        modified_state_dict = {}
        for key, value in state_dict.items():
            # 提取模块的编号（比如 "dual_dit_blocks.2." 提取为 "2."）
            new_key = key.split('dual_dit_blocks.')[-1]  # 去掉 'dual_dit_blocks.' 前缀
            modified_state_dict[new_key] = value

        # 加载修改后的state_dict
        missing, unexpected = self.dual_dit_blocks.load_state_dict(modified_state_dict, strict=False)
        print(f"Missing keys dual_dit_state: {missing}")
        print(f"Unexpected keys dual_dit_state: {unexpected}")


if __name__ == "__main__":
    prompt = "A woman sits perched on a chair on a stage, playing a beige lute. She wears a green cheongsam, and there is a dark curtain behind her. In front of her stands a black microphone capturing the music she produces. Her fingers deftly move across the strings, creating a captivating melody."
    negative_prompts = "low quality"
    pipeline = OverallCaption_T2V_T2A_Pipeline("config/sample.yaml")
    #pipeline = JointDiT_T2AV("config/sample.yaml")
    pipeline.predict_step(prompt,negative_prompts)
    

        
        
