import os, sys, wandb
from datetime import datetime, timezone, timedelta
import torch
import pytorch_lightning as pl
import json
import torch.multiprocessing as mp
from tqdm import tqdm
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from base import get_parser, prepare_args, prepare_config, instantiate_from_config, get_func_from_str
# from infer_joint import JointDiT_T2AV
from train_joint import JointDiT_T2AV
from dataset.dataset_va import VGGSoundDataset, AvsyncDataset, VGGSoundSSDataset, LandscapeDataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import CSVLogger
from accelerate import load_checkpoint_in_model
import argparse
deepspeed_strategy = DeepSpeedStrategy(config="config/deepspeed.json")

def train_vgg_multi_node():
    import os
    os.environ["NCCL_DEBUG"] = "ERROR"
    os.environ["FI_LOG_LEVEL"] = "ERROR"
    import importlib
    from tqdm import tqdm
    from omegaconf import OmegaConf
    
    config = OmegaConf.load("config/dataset.yaml")
    train_dataset = VGGSoundDataset(mode="train")
    model = JointDiT_T2AV(config_path="config/train_large.yaml")
    checkpoint_callback = ModelCheckpoint(
        dirpath="/mnt/task_runtime/t2av/code_base/checkpoints/joint_t2av",
        filename="model-{step}",  # 文件名包含步骤数
        every_n_train_steps=1000,  # 每1000步保存一次
        save_top_k=-1,             # 保存所有检查点
        save_on_train_epoch_end=False,  # 不在epoch结束时额外保存
        save_weights_only=False    # 保存完整状态（包含优化器等）
    )
    
    # ==================== 修改训练器配置 ====================
    trainer = Trainer(
        log_every_n_steps=5,
        default_root_dir="/mnt/task_runtime/t2av/code_base/checkpoints/joint_t2av",
        accelerator="gpu",
        num_nodes=2,
        devices=8,
        strategy=deepspeed_strategy,
        precision='bf16',
        max_steps=50000,
        enable_checkpointing=True,  # 启用检查点功能
        callbacks=[checkpoint_callback]  # 添加回调
    )
    if trainer.global_rank == 0:
        wandb_logger = WandbLogger(project="T2AV-vgg-14B-nolora")
        trainer.logger = wandb_logger
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=4, 
        prefetch_factor=2, 
        persistent_workers=True
    )
    trainer.fit(model, train_loader)

def train_vgg_ss_b200():
    import os
    os.environ["NCCL_DEBUG"] = "ERROR"
    os.environ["FI_LOG_LEVEL"] = "ERROR"
    import importlib
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from pytorch_lightning.strategies import DDPStrategy
    wandb_logger = WandbLogger(project="vgg-ss-14B-bicross-fusion-no-train-layers")
    config = OmegaConf.load("config/dataset.yaml")
    train_dataset = VGGSoundSSDataset(mode="train")
    model = JointDiT_T2AV(config_path="config/train_large.yaml")
    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=5,
        default_root_dir="/mnt/task_runtime/t2av/code_base/checkpoints/vggss",
        accelerator="gpu",
        devices=8,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True, static_graph=True),
        precision='bf16',
        max_steps=15000,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=1, persistent_workers=False)
    trainer.fit(model, train_loader)

def train_avsync():
    import os
    os.environ["NCCL_DEBUG"] = "ERROR"
    os.environ["FI_LOG_LEVEL"] = "ERROR"
    import importlib
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from pytorch_lightning.strategies import DDPStrategy
    # ckpt_path = "/mnt/task_runtime/t2av/code_base/T2AV-v1.0-avsync/ngonifk1/checkpoints/epoch=59-step=5101.ckpt"
    wandb_logger = WandbLogger(project="avsync-1.3B-bridge-bicross")
    train_dataset = AvsyncDataset(mode="train")
    model = JointDiT_T2AV(config_path="config/train.yaml")
    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=5,  # 每10步刷新一次日志
        default_root_dir="/mnt/task_runtime/t2av/code_base/checkpoints/joint_t2av",
        accelerator="gpu",  
        devices=8,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True, static_graph=True),
        precision='bf16',
        max_steps=15000,
    )
    trainer.strategy.strict_loading = False
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=1, persistent_workers=False)
    trainer.fit(model, train_loader)

def train_avsync_multi_node():
    import os
    os.environ["NCCL_DEBUG"] = "ERROR"
    os.environ["FI_LOG_LEVEL"] = "ERROR"
    import importlib
    from tqdm import tqdm
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/dataset.yaml")
    train_dataset = AvsyncDataset(mode="train")
    model = JointDiT_T2AV(config_path="config/train_large.yaml")
    checkpoint_callback = ModelCheckpoint(
        dirpath="/mnt/task_runtime/t2av/code_base/checkpoints/avsync",
        filename="model-{step}",  # 文件名包含步骤数
        every_n_train_steps=1000,  # 每1000步保存一次
        save_top_k=-1,             # 保存所有检查点
        save_on_train_epoch_end=False,  # 不在epoch结束时额外保存
        save_weights_only=False    # 保存完整状态（包含优化器等）
    )
    
    # ==================== 修改训练器配置 ====================
    trainer = Trainer(
        log_every_n_steps=5,
        default_root_dir="/mnt/task_runtime/t2av/code_base/checkpoints/avsync",
        accelerator="gpu",
        num_nodes=2,
        devices=8,
        strategy=deepspeed_strategy,
        precision='bf16',
        max_steps=50000,
        enable_checkpointing=True,  # 启用检查点功能
        callbacks=[checkpoint_callback]  # 添加回调
    )
    if trainer.global_rank == 0:
        wandb_logger = WandbLogger(project="T2AV-avsync-14B")
        trainer.logger = wandb_logger
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=4, 
        prefetch_factor=2, 
        persistent_workers=True
    )
    trainer.fit(model, train_loader)

def train_vgg_ss():
    import os
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["FI_LOG_LEVEL"] = "ERROR"
    import importlib
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from pytorch_lightning.strategies import DDPStrategy
    wandb_logger = WandbLogger(project="vgg-ss-1.3B-fullattn-fusion")
    ckpt_path = "/mnt/task_runtime/t2av/code_base/vgg-ss-1.3B-fullattn-fusion/d6tjx0wj/checkpoints/epoch=27-step=8848.ckpt"
    config = OmegaConf.load("config/dataset.yaml")
    train_dataset = VGGSoundSSDataset(mode="train")
    model = JointDiT_T2AV(config_path="config/train.yaml")
    trainer = Trainer(
        # logger=wandb_logger,
        log_every_n_steps=5,
        default_root_dir="/mnt/task_runtime/t2av/code_base/checkpoints/vggss",
        accelerator="gpu",
        devices=8,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True, static_graph=True),
        precision='bf16',
        max_steps=10000,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=1, persistent_workers=False)
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)

def train_landscape():
    import os
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["FI_LOG_LEVEL"] = "ERROR"
    import importlib
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from pytorch_lightning.strategies import DDPStrategy
    wandb_logger = WandbLogger(project="landscape-1.3B-bicross")
    config = OmegaConf.load("config/dataset.yaml")
    train_dataset = LandscapeDataset(mode="train")
    model = JointDiT_T2AV(config_path="config/train.yaml")
    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=5,
        default_root_dir="/mnt/task_runtime/t2av/code_base/checkpoints/landscape",
        accelerator="gpu",
        devices=8,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True, static_graph=True),
        precision='bf16',
        max_steps=10000,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=1, persistent_workers=False)
    trainer.fit(model, train_loader)

def predict():
    video_prompt = "In a serene backyard setting, a German Shepherd stands alert on a stone-paved path beside a crystal-clear swimming pool, the water shimmering under the soft, diffused sunlight. The atmosphere is tranquil, with lush greenery flanking the path and a white lounge chair resting invitingly near the pool's edge. The dog, with its sleek black and tan coat, pricks its ears forward, its body tense with anticipation. Its eyes focus intently on something off-screen, and its tail wags slightly, creating a subtle breeze that rustles the nearby foliage. As the dog begins to move, its paws press firmly against the cool stone tiles, sending tiny pebbles skittering aside. It leaps forward with a burst of energy, its muscles contracting powerfully, propelling it through the air. The dog's landing is graceful yet forceful, its paws striking the ground with enough impact to send a faint tremor through the earth. The surrounding plants sway gently in response to the dog's swift movements, their leaves brushing softly against each other. The scene is captured in a medium shot, the camera remaining steady, allowing the viewer to fully absorb the dynamic interplay between the dog's actions and the peaceful garden environment. The visual style is naturalistic and highly detailed, with soft, natural lighting that enhances the textures of the dog's fur, the stone path, and the vibrant greenery, all rendered in crisp, 8K quality."
    video_negative_prompts = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    audio_prompt = "The gentle rustle of leaves and the soft swaying of plants accompany the alert barks of a German Shepherd, creating a tranquil yet energetic soundscape."
    audio_negative_prompts = "Low quality, unclear and noisy."
    model = JointDiT_T2AV(config_path="config/sample.yaml")
    model.load_ckpt("/mnt/task_runtime/t2av/code_base/test-trainable_layers_set_bridge_layer/yx08xpmk/checkpoints/epoch=1-step=170.ckpt")
    model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts,save_name="test111")

def process_subset(subset_keys, gpu_index):
    prompt_file = '/mnt/task_runtime/t2av/caption_pipeline/recaption/avsync-test-72B-captions.json'
    prompt_json = json.load(open(prompt_file, 'r'))
    video_negative_prompts = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    audio_negative_prompts = "Low quality, unclear and noisy."
    
    torch.cuda.set_device(gpu_index)
    model = JointDiT_T2AV(config_path="config/sample.yaml")
    # model.load_ckpt("/mnt/task_runtime/t2av/code_base/avsync-bicross-1.3B/epoch=59-step=5100.ckpt")
    
    for key in subset_keys:
        video_name = os.path.basename(key)
        if video_name in os.listdir('/mnt/task_runtime/t2av/code_base/save_videos/no_training_single_caption'):
            continue
        video_id = os.path.basename(key)[:-4]
        video_prompt = prompt_json[key]['video_caption']
        # audio_prompt = prompt_json[key]['audio_caption']
        audio_prompt = video_prompt
        model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts, save_name=video_id)

def predict_avsync_parallel():
    prompt_file = '/mnt/task_runtime/t2av/caption_pipeline/recaption/avsync-test-72B-captions.json'
    prompt_json = json.load(open(prompt_file, 'r'))
    keys = list(prompt_json.keys())
    num_keys = len(keys)
    
    # 将 keys 分成 8 份
    subset_sizes = [num_keys // 8 + (1 if i < num_keys % 8 else 0) for i in range(8)]
    subsets = []
    start = 0
    for size in subset_sizes:
        subsets.append(keys[start:start + size])
        start += size
    
    # 创建并启动 8 个进程
    processes = []
    for i in range(8):
        p = mp.Process(target=process_subset, args=(subsets[i], i))
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()

def process_subset_vggss(subset_keys, gpu_index):
    prompt_file = '/mnt/task_runtime/t2av/caption_pipeline/recaption/vgg-ss-test-72B-caption.json'
    prompt_json = json.load(open(prompt_file, 'r'))
    video_negative_prompts = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    audio_negative_prompts = "Low quality, unclear and noisy."
    torch.cuda.set_device(gpu_index)
    model = JointDiT_T2AV(config_path="config/sample.yaml")
    model.load_ckpt("/mnt/task_runtime/t2av/code_base/vgg-ss-1.3B-fullattn-fusion/d6tjx0wj/checkpoints/epoch=27-step=8848.ckpt")
    
    for key in subset_keys:
        video_id = os.path.basename(key)[:-4]
        video_prompt = prompt_json[key]['video_caption']
        audio_prompt = prompt_json[key]['audio_caption']
        model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts, save_name=video_id)

def predict_vggss_parallel():
    prompt_file = '/mnt/task_runtime/t2av/caption_pipeline/recaption/vgg-ss-test-72B-caption.json'
    prompt_json = json.load(open(prompt_file, 'r'))
    keys = list(prompt_json.keys())
    num_keys = len(keys)
    
    # 将 keys 分成 8 份
    subset_sizes = [num_keys // 8 + (1 if i < num_keys % 8 else 0) for i in range(8)]
    subsets = []
    start = 0
    for size in subset_sizes:
        subsets.append(keys[start:start + size])
        start += size
    
    # 创建并启动 8 个进程
    processes = []
    for i in range(8):
        p = mp.Process(target=process_subset_vggss, args=(subsets[i], i))
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()

def process_subset_landscape(subset_keys, gpu_index):
    prompt_file = '/mnt/task_runtime/t2av/caption_pipeline/recaption/landscape-captions-test.json'
    prompt_json = json.load(open(prompt_file, 'r'))
    video_negative_prompts = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    audio_negative_prompts = "Low quality, unclear and noisy."
    torch.cuda.set_device(gpu_index)
    model = JointDiT_T2AV(config_path="config/sample.yaml")
    # model.load_ckpt("/mnt/task_runtime/t2av/code_base/landscape-1.3B-bicross/fi5torc1/checkpoints/epoch=175-step=10000.ckpt")
    for key in subset_keys:
        video_id = os.path.basename(key)[:-4]
        video_prompt = prompt_json[key]['video_caption']
        audio_prompt = prompt_json[key]['audio_caption']
        model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts, save_name=video_id)

def predict_landscape_parallel():
    prompt_file = '/mnt/task_runtime/t2av/caption_pipeline/recaption/landscape-captions-test.json'
    prompt_json = json.load(open(prompt_file, 'r'))
    keys = list(prompt_json.keys())
    num_keys = len(keys)
    
    # 将 keys 分成 8 份
    subset_sizes = [num_keys // 8 + (1 if i < num_keys % 8 else 0) for i in range(8)]
    subsets = []
    start = 0
    for size in subset_sizes:
        subsets.append(keys[start:start + size])
        start += size
    
    # 创建并启动 8 个进程
    processes = []
    for i in range(8):
        p = mp.Process(target=process_subset_landscape, args=(subsets[i], i))
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
def infer_deepspeed():
    video_prompt = "In a serene outdoor setting, a hand firmly grips a meticulously crafted flintlock pistol, its polished brass barrel gleaming under natural daylight while the rich, dark wood handle exudes a sense of historical craftsmanship. The scene is calm, with soft, diffused sunlight filtering through unseen trees, casting gentle shadows across the grassy ground scattered with fallen leaves. The camera focuses closely on the pistol, capturing every intricate detail from the ornate silver trigger guard to the finely engraved lock mechanism. As the hand adjusts its grip, the fingers press against the wooden surface, subtly emphasizing the texture and weight of the weapon. The thumb moves to cock the hammer, lifting it with a deliberate motion that suggests the potential energy stored within. The surrounding environment remains still, with only the faintest hint of movement in the distant foliage, creating a stark contrast to the imminent action. The visual counterpart of the impending discharge is palpable as the hammer is drawn back, poised to strike the flint, generating a shower of sparks that will ignite the gunpowder. This moment is captured in high-definition, photorealistic detail, with a cinematic quality that highlights the tension and anticipation of the scene, all set against the backdrop of nature's quietude."
    video_negative_prompts = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    audio_prompt = "The sharp, resonant report of a cap gun firing echoes through the air, accompanied by the faint rustle of dry grass and bushes disturbed by the concussion."
    audio_negative_prompts = "Low quality, unclear and noisy."
    model = JointDiT_T2AV(config_path="config/sample.yaml")
    load_checkpoint_in_model(model, checkpoint="/mnt/task_runtime/t2av/code_base/checkpoints/joint_t2av/model-step=11000.ckpt")
    print("Loaded checkpoint successfully")
    model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts, save_name="gun")


if __name__ == "__main__":
    train_avsync()
    # train_vgg_ss()
    # train_landscape()
