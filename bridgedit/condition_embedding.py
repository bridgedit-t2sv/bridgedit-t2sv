import pytorch_lightning as pl
import torch
from typing import Optional, Union, List, Dict, Any 
from transformers import AutoTokenizer, T5EncoderModel, UMT5EncoderModel
from tqdm import tqdm
import numpy as np
import time
from models.stable_audio.stable_audio_modeling import StableAudioProjectionModel
import torch.nn as nn
from functools import partial
from diffusers.training_utils import cast_training_params
import importlib
from datetime import datetime
import math
from dataset.dataset_va import VideoDataset
import ftfy
import html
import re
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf
import pickle



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

class EmbeddingCondition():
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        get_huggingface_model_from_local_dir = lambda ckpt_dir, model: \
            model.from_pretrained(
                ckpt_dir, local_files_only=True,   
                ignore_mismatched_sizes=False,      # Strict for pretrained model structure
                use_safetensors=True,               # Safe tensor
                torch_dtype=self.weight_dtype,      # bfloat16
            )
        self.weight_dtype = torch.bfloat16
        self.model_device = device or torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        video_text_encoder = "/mnt/task_runtime/models/Wan2.1-T2V-1.3B-Diffusers/text_encoder"
        audio_text_encoder = "/mnt/task_runtime/models/stable-audio-open-1.0/text_encoder"
        video_tokenizer = "/mnt/task_runtime/models/Wan2.1-T2V-1.3B-Diffusers/tokenizer"
        audio_tokenizer = "/mnt/task_runtime/models/stable-audio-open-1.0/tokenizer"
        audio_projection_model = "/mnt/task_runtime/models/stable-audio-open-1.0/projection_model"
        self.video_text_encoder = get_huggingface_model_from_local_dir(video_text_encoder, UMT5EncoderModel).to(self.model_device)
        self.audio_text_encoder = get_huggingface_model_from_local_dir(audio_text_encoder, T5EncoderModel).to(self.model_device)
        self.video_tokenizer = AutoTokenizer.from_pretrained(video_tokenizer)
        self.audio_tokenizer = AutoTokenizer.from_pretrained(audio_tokenizer)
        self.audio_projection_model = get_huggingface_model_from_local_dir(audio_projection_model, StableAudioProjectionModel)
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
        prompt_embeds = self.video_text_encoder(text_input_ids.to(self.model_device), mask.to(self.model_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.model_device)
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
        num_samples_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
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
        return prompt_embeds
                
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

            text_input_ids = text_input_ids.to(self.model_device)
            attention_mask = attention_mask.to(self.model_device)
            # 2. Text encoder forward
            self.audio_text_encoder.to(self.model_device)
            prompt_embeds = self.audio_text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = self.audio_projection_model(
            text_hidden_states=prompt_embeds,
        ).text_hidden_states
        if attention_mask is not None:
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)
        return prompt_embeds
# decord_vr = decord.VideoReader(va_path, num_threads=1, ctx=decord.cpu(0))
# video_raw_frame_num = len(decord_vr)

# # 变为 tensor
# video_tensor = torchvision.io.read_video(va_path, pts_unit='sec')


# video_frame_transform = transforms.Compose([
#     transforms.Lambda(lambda x: x/255 * 2 - 1),
#     transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
#     transforms.Resize((480, 832), antialias=True),  
# ])

# video_tensor = video_frame_transform(video_tensor[0])
# print(video_tensor.shape)
# # print(video_tensor[0])


def main():
    seperate_flag = True
    save_path = "/mnt/task_runtime/t2av/code_base/emb_cache/seperate" if seperate_flag else "/mnt/task_runtime/t2av/code_base/emb_cache/overall"
    config = OmegaConf.load("config/dataset.yaml")
    def get_obj_from_str(string, reload=False):
        module, obj_class = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), obj_class)

    def get_class_from_config(config):
        if not "target" in config:
            raise KeyError("Expected key `target` to instantiate.")
        return get_obj_from_str(config["target"])

    def instantiate_from_config(config):
        if not "target" in config:
            raise KeyError("Expected key `target` to instantiate.")
        return get_obj_from_str(config["target"])(**config.get("params", dict()))

    dist.init_process_group(backend="nccl")

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    train_dataset = instantiate_from_config(config.data.favd.train)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=8,
        prefetch_factor=1,
        persistent_workers=True,
    )

    embedding_model = EmbeddingCondition(device=torch.device(f"cuda:{local_rank}"))
    embedding_js = {}
    if seperate_flag:
       with torch.no_grad():
            for batch in tqdm(train_loader, desc=f"Rank {local_rank}"):
                audio_prompt, video_prompt = batch["audio_prompt"], batch["video_prompt"]
                video_embs = embedding_model.encode_video_prompt(video_prompt)
                audio_embs = embedding_model.encode_audio_prompt(audio_prompt)
                video_ids = batch["video_id"]
                for video_id, video_emb, audio_emb in zip(video_ids, video_embs, audio_embs):
                    embedding_js[video_id] = (video_emb.cpu(), audio_emb.cpu())
    else:
        with torch.no_grad():
            for batch in tqdm(train_loader, desc=f"Rank {local_rank}"):
                text_prompt = batch["text_prompt"]
                video_embs = embedding_model.encode_video_prompt(text_prompt)
                audio_embs = embedding_model.encode_audio_prompt(text_prompt)
                video_ids = batch["video_id"]
                for video_id, video_emb, audio_emb in zip(video_ids, video_embs, audio_embs):
                    embedding_js[video_id] = (video_emb.cpu(), audio_emb.cpu())

    # 保存每个rank的结果，比如保存到不同文件
    torch.save(embedding_js, f"{save_path}/condition_embedding_rank{local_rank}.pt")

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     # save the embedding condition to a file
#     def get_obj_from_str(string, reload=False):
#         module, obj_class = string.rsplit(".", 1)
#         if reload:
#             module_imp = importlib.import_module(module)
#             importlib.reload(module_imp)
#         return getattr(importlib.import_module(module, package=None), obj_class)

#     def get_class_from_config(config):
#         if not "target" in config:
#             raise KeyError("Expected key `target` to instantiate.")
#         return get_obj_from_str(config["target"])

#     def instantiate_from_config(config):
#         if not "target" in config:
#             raise KeyError("Expected key `target` to instantiate.")
#         return get_obj_from_str(config["target"])(**config.get("params", dict()))

#     from omegaconf import OmegaConf
#     config = OmegaConf.load("config/dataset.yaml")
    
#     train_dataset = instantiate_from_config(config.data.favd.train)
#     print(len(train_dataset))
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, prefetch_factor=1, persistent_workers=True)
#     # 将train_loader中的数据分为 8 部分，每个显卡处理 1 部分
    
#     embedding_model = EmbeddingCondition()
#     embedding_js = {}
#     add tqdm
#     with torch.no_grad():
#         for batch in tqdm(fuckyou_loader):
#             text_prompt = batch["text_prompt"]
#             video_embs = embedding_model.encode_video_prompt(text_prompt)
#             audio_embs = embedding_model.encode_audio_prompt(text_prompt)
#             video_ids = batch["video_id"]
#             for video_id, video_emb, audio_emb in zip(video_ids, video_embs, audio_embs):
#                 embedding_js[video_id] =  (video_emb.cpu(), audio_emb.cpu())
#     with open("condition_embedding.pkl", "wb") as f:
#         pickle.dump(embedding_js, f)
    