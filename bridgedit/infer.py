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

def predict():
    video_prompt = "In a serene backyard setting, a German Shepherd stands alert on a stone-paved path beside a crystal-clear swimming pool, the water shimmering under the soft, diffused sunlight. The atmosphere is tranquil, with lush greenery flanking the path and a white lounge chair resting invitingly near the pool's edge. The dog, with its sleek black and tan coat, pricks its ears forward, its body tense with anticipation. Its eyes focus intently on something off-screen, and its tail wags slightly, creating a subtle breeze that rustles the nearby foliage. As the dog begins to move, its paws press firmly against the cool stone tiles, sending tiny pebbles skittering aside. It leaps forward with a burst of energy, its muscles contracting powerfully, propelling it through the air. The dog's landing is graceful yet forceful, its paws striking the ground with enough impact to send a faint tremor through the earth. The surrounding plants sway gently in response to the dog's swift movements, their leaves brushing softly against each other. The scene is captured in a medium shot, the camera remaining steady, allowing the viewer to fully absorb the dynamic interplay between the dog's actions and the peaceful garden environment. The visual style is naturalistic and highly detailed, with soft, natural lighting that enhances the textures of the dog's fur, the stone path, and the vibrant greenery, all rendered in crisp, 8K quality."
    video_negative_prompts = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    audio_prompt = "The gentle rustle of leaves and the soft swaying of plants accompany the alert barks of a German Shepherd, creating a tranquil yet energetic soundscape."
    audio_negative_prompts = "Low quality, unclear and noisy."
    model = JointDiT_T2AV(config_path="config/sample.yaml")
    model.load_ckpt("/mnt/task_runtime/t2av/code_base/test-trainable_layers_set_bridge_layer/yx08xpmk/checkpoints/epoch=1.ckpt")
    model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts,save_name="test111")


  
def infer_deepspeed():
    video_prompt = "In a serene outdoor setting, a hand firmly grips a meticulously crafted flintlock pistol, its polished brass barrel gleaming under natural daylight while the rich, dark wood handle exudes a sense of historical craftsmanship. The scene is calm, with soft, diffused sunlight filtering through unseen trees, casting gentle shadows across the grassy ground scattered with fallen leaves. The camera focuses closely on the pistol, capturing every intricate detail from the ornate silver trigger guard to the finely engraved lock mechanism. As the hand adjusts its grip, the fingers press against the wooden surface, subtly emphasizing the texture and weight of the weapon. The thumb moves to cock the hammer, lifting it with a deliberate motion that suggests the potential energy stored within. The surrounding environment remains still, with only the faintest hint of movement in the distant foliage, creating a stark contrast to the imminent action. The visual counterpart of the impending discharge is palpable as the hammer is drawn back, poised to strike the flint, generating a shower of sparks that will ignite the gunpowder. This moment is captured in high-definition, photorealistic detail, with a cinematic quality that highlights the tension and anticipation of the scene, all set against the backdrop of nature's quietude."
    video_negative_prompts = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    audio_prompt = "The sharp, resonant report of a cap gun firing echoes through the air, accompanied by the faint rustle of dry grass and bushes disturbed by the concussion."
    audio_negative_prompts = "Low quality, unclear and noisy."
    model = JointDiT_T2AV(config_path="config/sample.yaml")
    load_checkpoint_in_model(model, checkpoint="/mnt/task_runtime/t2av/code_base/checkpoints/joint_t2av/model-step=20000.ckpt")
    print("Loaded checkpoint successfully")
    model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts, save_name="gun")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_prompt", type=str, default="In a serene backyard setting, a German Shepherd stands alert on a stone-paved path beside a crystal-clear swimming pool, the water shimmering under the soft, diffused sunlight. The atmosphere is tranquil, with lush greenery flanking the path and a white lounge chair resting invitingly near the pool's edge. The dog, with its sleek black and tan coat, pricks its ears forward, its body tense with anticipation. Its eyes focus intently on something off-screen, and its tail wags slightly, creating a subtle breeze that rustles the nearby foliage. As the dog begins to move, its paws press firmly against the cool stone tiles, sending tiny pebbles skittering aside. It leaps forward with a burst of energy, its muscles contracting powerfully, propelling it through the air. The dog's landing is graceful yet forceful, its paws striking the ground with enough impact to send a faint tremor through the earth. The surrounding plants sway gently in response to the dog's swift movements, their leaves brushing softly against each other. The scene is captured in a medium shot, the camera remaining steady, allowing the viewer to fully absorb the dynamic interplay between the dog's actions and the peaceful garden environment. The visual style is naturalistic and highly detailed, with soft, natural lighting that enhances the textures of the dog's fur, the stone path, and the vibrant greenery, all rendered in crisp, 8K quality.")
    parser.add_argument("--video_negative_prompts", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    parser.add_argument("--audio_prompt", type=str, default="The sharp, resonant report of a cap gun firing echoes through the air, accompanied by the faint rustle of dry grass and bushes disturbed by the concussion.")
    parser.add_argument("--audio_negative_prompts", type=str, default="Low quality, unclear and noisy.")
    parser.add_argument("--save_file", type=str, default="test.mp4")
    parser.add_argument("--ckpt_path", type=str, default="/mnt/task_runtime/t2av/code_base/checkpoints/joint_t2av/model-step=20000.ckpt")
    args = parser.parse_args()
    video_prompt = args.video_prompt
    video_negative_prompts = args.video_negative_prompts
    audio_prompt = args.audio_prompt
    audio_negative_prompts = args.audio_negative_prompts
    save_file = args.save_file
    ckpt_path = args.ckpt_path
    model = JointDiT_T2AV(config_path="config/sample.yaml")
    model.load_ckpt(ckpt_path)
    print("Loaded checkpoint successfully")
    model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts, save_name=save_file)
