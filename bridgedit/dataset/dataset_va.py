import sys, os, random, traceback, json, decord, einops
import numpy as np
from tqdm import tqdm
from typing import Union, List
from librosa.filters import mel as librosa_mel
import imageio
from PIL import Image
import torch, torchaudio, torchvision
import torchvision.transforms as transforms

from transformers import CLIPTokenizer, RobertaTokenizer, RobertaTokenizerFast
from diffusers.video_processor import VideoProcessor

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 # Dataset mode params
                 meta_dir   : str,
                 seperate_meta_dir   : str,
                 use_cache  : bool = False,
                 filtered_id_path   : str = None,
                 load_mode_item     : str = "favd", 
                 load_mode_meta     : str = "favd",
                 load_from_cache    : bool = False,
                 load_seperate      : bool = False,
                 # Process params 
                 audio_process_config : dict = None,
                 video_process_config : dict = None,
                 embedding_cache_path_overall : str = None,
                 embedding_cache_path_seperate : str = None,
        ):
        # Dataset params
        self.load_mode_item  = load_mode_item
        self.load_mode_meta  = load_mode_meta
        if load_seperate:
            self.meta_dir   = seperate_meta_dir
        else:
            self.meta_dir   = meta_dir
        self.load_seperate = load_seperate
        self.metas = self._load_meta(meta_dir, load_from_cache, load_seperate)
        # Video audio process params
        self.audio_process_config = audio_process_config
        self.video_process_config = video_process_config
        assert self.audio_process_config.duration == self.video_process_config.duration, "Audio and video duration should be the same."
        self.duration = self.audio_process_config.duration
        self.audio_process_config.target_mel_length = int(self.duration * self.audio_process_config.sampling_rate / self.audio_process_config.hop_length)
        self.video_process_config.target_frame_length = int(self.duration * self.video_process_config.target_fps)
        self.target_fps = self.video_process_config.target_fps
        self.target_frame_nums = self.video_process_config.target_frame_nums
        self.video_processor = VideoProcessor(vae_scale_factor=self.video_process_config.vae_scale_factor)
        self.video_fram_h, self.video_frame_w = self.video_process_config.video_h, self.video_process_config.video_w
        self.video_frame_transform = transforms.Compose([
            transforms.Lambda(lambda x: x/255 * 2 - 1),
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            transforms.Resize((self.video_fram_h, self.video_frame_w), antialias=True),  
        ])
        self.load_from_cache = load_from_cache
        self.embedding_cache_path = embedding_cache_path_overall if not load_seperate else embedding_cache_path_seperate

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def _load_meta(self, meta_dir, load_from_cache, ext=".csv", meta_item_num=-1, filtered_id_path=None):
        cur_metas = []
        if self.load_mode_meta == "favd" and self.load_seperate == False:
            # NOTE Only support meta_dir as list in the format of [video_dir, audio_dir, desc_json_path, (cache_path)], and split as []
            with open(meta_dir[1], 'r') as file:
                data = json.load(file)
            id_to_desc = {item["id"]: item["descriptions"] for item in data}
            videos_names = os.listdir(meta_dir[0])
            
            desc = []
            # extract video and description
            for video_name in videos_names:
                vid = video_name.split('.')[0]
                desc = id_to_desc[vid]
                cur_metas.append([os.path.join(meta_dir[0], video_name), desc])
        if self.load_mode_meta == "favd" and self.load_seperate == True:
            with open(self.meta_dir[1], 'r') as file:
                data = json.load(file)
            id_to_desc = {item["id"]: (item['audio_caption'], item['video_caption']) for item in data}
            videos_names = os.listdir(meta_dir[0])
            
            desc = []
            # extract video and description
            for video_name in videos_names:
                vid = video_name.split('.')[0]
                desc = id_to_desc[vid]
                cur_metas.append([os.path.join(meta_dir[0], video_name), desc])
        assert len(cur_metas) > 0, f"Meta data is empty."
        
        return cur_metas

    def __len__(self):
        return len(self.metas)
    
    def load_from_cache_path(self):
        cache_files = {}
        # merge to one dict
        # print(f"Loading cache files from {self.embedding_cache_path}")
        for file in os.listdir(self.embedding_cache_path):
            if file.endswith(".pt"):
                file_path = os.path.join(self.embedding_cache_path, file)
                data = torch.load(file_path)
                cache_files.update(data)
        # print(f"Load {len(cache_files)} cache files from {self.embedding_cache_path}")
        
        return cache_files

    def __getitem__(self, idx):
        if self.load_from_cache:
            self.cache_files = self.load_from_cache_path()
            return self.getitem_favd_from_cache(idx, self.cache_files)
        else:
            return self.getitem_favd(idx)

    def getitem_favd(self, idx):
        cur_video_path, description = self.metas[idx]
        cur_video_id = os.path.basename(cur_video_path).split('.')[0]
        """ Audio part """
        waveform = self.prepare_audio_data_from_va_file(va_path=cur_video_path)
        """ Video part """
        video_frames = self.prepare_video_data_from_va_file(va_path=cur_video_path)
        if self.load_seperate:
            res = {
                "waveform": waveform.squeeze(0),
                "video_frame": video_frames.squeeze(0),
                "video_id": cur_video_id,
                "audio_prompt": description[0],
                "video_prompt": description[1],
            }
        else:
            res = {
                "waveform": waveform.squeeze(0),
                "video_frame": video_frames.squeeze(0),
                "video_id": cur_video_id,
                "text_prompt": description,
            }
        
        return res
        
    def getitem_favd_from_cache(self, idx, cache_files):
        cur_video_path, description = self.metas[idx]
        cur_video_id = os.path.basename(cur_video_path).split('.')[0]
        
        cur_video_path, description = self.metas[idx]
        cur_video_id = os.path.basename(cur_video_path).split('.')[0]
        """ Audio part """
        # log_mel_specs, stfts = self.prepare_audio_data_from_va_file(va_path=cur_video_path)
        # log_mel_specs = torch.stack(log_mel_specs)
        # stfts = torch.stack(stfts)
        waveform = self.prepare_audio_data_from_va_file(va_path=cur_video_path)
        """ Video part """
        video_frames = self.prepare_video_data_from_va_file(va_path=cur_video_path)
        res = {
            # "mel": log_mel_specs.squeeze(0),
            # "stft": stfts.squeeze(0),
            "waveform": waveform.squeeze(0),
            "video_frame": video_frames.squeeze(0),
            "video_id": cur_video_id,
            "video_emb": cache_files[cur_video_id][0],
            "audio_emb": cache_files[cur_video_id][1],
        }
        
        return res
        
    def prepare_audio_data_from_va_file(self, va_path, maximum_amplitude=0.5):
        ''' Load audio data and resample to the target sampling rate '''
        
        waveform, sr = torchaudio.load(va_path)
        waveform = torchaudio.functional.resample(waveform, sr, self.audio_process_config.sampling_rate)
        waveform = waveform[0, ...].float()
        # Validate the waveform length
        waveform_length = waveform.shape[-1]
        if waveform_length <= self.audio_process_config.min_waveform_length:
            raise RuntimeError(f"Waveform is too short, {waveform_length}.")

        ''' Random segment and pad '''
        target_length = int(self.audio_process_config.sampling_rate * self.duration)
        if waveform_length <= target_length:
            temp_wav = waveform.repeat(10)
            sample_waveforms = temp_wav[:target_length]
        else: 
            cur_waveform_1 = waveform[:target_length]
            sample_waveforms = cur_waveform_1
        return sample_waveforms
    def prepare_video_data_from_va_file(self, va_path, backend="decord"):
        ''' Load video data '''
        video_tuple = torchvision.io.read_video(va_path, pts_unit='sec')
        video_raw_frame_num = len(video_tuple[0])
        video_raw_fps = video_tuple[2]['video_fps']
        video_raw_duration = video_raw_frame_num / video_raw_fps
        video_tensor = video_tuple[0]
        video_tensor = self.video_frame_transform(video_tensor)            
        ''' Sample frames as the same as the audio part and Pad'''
        if video_raw_frame_num <= self.target_frame_nums:
            temp_video = video_tensor.repeat(10, 1, 1, 1)
            sample_videos = temp_video[:self.target_frame_nums]
        else:
            sample_videos = video_tensor[:self.target_frame_nums]
        return sample_videos.permute(1, 0 , 2, 3)    
        
# 在类外部定义全局函数
def normalize_video(x):
    return x.float() / 255.0 * 2 - 1

def permute_channels(x):
    return x.permute(0, 3, 1, 2)

class VGGSoundDataset(torch.utils.data.Dataset):
    def __init__(self, meta_dir='/mnt/task_runtime/t2av/caption_pipeline/recaption/vggsound_train_all.json', mode="train", separate_caption=True):
        self.video_dir = '/mnt/task_runtime/dataset/vgg_filter'
        self.meta_dir = meta_dir
        self.mode = mode
        self.video_path = os.path.join(self.video_dir, self.mode)
        self.metas_json = self._load_meta(meta_dir)     
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
        self.separate_caption = separate_caption
        # 使用全局函数和固定参数创建transform
        self.video_frame_transform = transforms.Compose([
            transforms.Lambda(normalize_video),
            transforms.Lambda(permute_channels),
            transforms.Resize(
                (self.video_process_config["video_h"], self.video_process_config["video_w"]), 
                antialias=True
            )
        ])
    
    def _load_meta(self, meta_dir):
        with open(meta_dir, 'r') as file:
            return json.load(file)
    
    def __len__(self):
        return len(self.metas_json)
    
    def __getitem__(self, idx):
        keys = list(self.metas_json.keys())
        video_id = keys[idx]
        video_caption = self.metas_json[video_id]['video_caption']
        audio_caption = self.metas_json[video_id]['audio_caption']
        video_path = os.path.join(self.video_path, video_id + ".mp4")
        if self.separate_caption:
            return {
                "video_path": video_path,
                "video_caption": video_caption,
                "audio_caption": audio_caption,
            }
        else:
            return {
                "video_path": video_path,
                "video_caption": video_caption,
                "audio_caption": video_caption,
            }

class AvsyncDataset(torch.utils.data.Dataset):
    def __init__(self, meta_dir='/mnt/task_runtime/t2av/caption_pipeline/recaption/avsync-train-72B-captions-omini.json', mode="train"):
        self.meta_dir = meta_dir
        self.mode = mode
        self.metas_json = self._load_meta(meta_dir)
        self.duration = 5.40

    def _load_meta(self, meta_dir):
        with open(meta_dir, 'r') as file:
            return json.load(file)
    
    def __len__(self):
        return len(self.metas_json)

    def __getitem__(self, idx):
        keys = list(self.metas_json.keys())
        video_id = keys[idx]
        video_caption = self.metas_json[video_id]['video_caption']
        audio_caption = self.metas_json[video_id]['audio_caption']
        video_path = video_id
        return {    
            "video_path": video_path,
            "video_caption": video_caption,
            "audio_caption": audio_caption,
        }
class VGGSoundSSDataset(torch.utils.data.Dataset):
    def __init__(self, meta_dir='/mnt/task_runtime/t2av/caption_pipeline/recaption/vgg-ss-72B-captions.json', mode="train"):
        self.meta_dir = meta_dir
        self.mode = mode
        self.metas_json = self._load_meta(meta_dir)
        self.duration = 5.40

    def _load_meta(self, meta_dir):
        with open(meta_dir, 'r') as file:
            return json.load(file)
    
    def __len__(self):
        return len(self.metas_json)

    def __getitem__(self, idx):
        keys = list(self.metas_json.keys())
        video_id = keys[idx]
        video_caption = self.metas_json[video_id]['video_caption']
        audio_caption = self.metas_json[video_id]['audio_caption']
        video_path = video_id
        return {
            "video_path": video_path,
            "video_caption": video_caption,
            "audio_caption": audio_caption,
        }

class LandscapeDataset(torch.utils.data.Dataset):
    def __init__(self, meta_dir='/mnt/task_runtime/t2av/caption_pipeline/recaption/landscape-captions-train.json', mode="train"):
        self.meta_dir = meta_dir
        self.mode = mode
        self.metas_json = self._load_meta(meta_dir)
        self.duration = 5.40

    def _load_meta(self, meta_dir):
        with open(meta_dir, 'r') as file:
            return json.load(file)
    
    def __len__(self):
        return len(self.metas_json)

    def __getitem__(self, idx):
        keys = list(self.metas_json.keys())
        video_id = keys[idx]
        video_caption = self.metas_json[video_id]['video_caption']
        audio_caption = self.metas_json[video_id]['audio_caption']
        video_path = video_id
        return {
            "video_path": video_path,
            "video_caption": video_caption,
            "audio_caption": audio_caption,
        }

if __name__ == "__main__":
    dataset = VGGSoundSSDataset(mode="train")
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset item: {dataset[0]}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    print(f"Dataloader length: {len(dataloader)}")
    for batch in dataloader:
        print(batch)
        break
    # import importlib
    # from tqdm import tqdm
    # from omegaconf import OmegaConf
    
    # def get_obj_from_str(string, reload=False):
    #     module, obj_class = string.rsplit(".", 1)
    #     if reload:
    #         module_imp = importlib.import_module(module)
    #         importlib.reload(module_imp)
    #     return getattr(importlib.import_module(module, package=None), obj_class)

    # def instantiate_from_config(config):
    #     if not "target" in config:
    #         raise KeyError("Expected key `target` to instantiate.")
    #     return get_obj_from_str(config["target"])(**config.get("params", dict()))

    # config = OmegaConf.load("../config/dataset.yaml")
    # # config = OmegaConf.load("/home/xihua/workspace/bridgediffusion/bridiff/config/bridge_dit_animatediff_audioldm/base.yaml")

    # train_dataset = instantiate_from_config(config.data.favd.train)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, prefetch_factor=1, persistent_workers=True)
    # for batch in train_loader:
    #     print(batch)
    #     print(batch["video_emb"].shape)
    #     print(batch["audio_emb"].shape)
    #     break
    
            
    