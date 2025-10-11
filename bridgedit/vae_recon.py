from diffusers.models import AutoencoderKLWan
import torch
import torchvision
import numpy as np
from diffusers.video_processor import VideoProcessor
from moviepy import VideoClip
from torchvision import transforms
def load_vae(vae_path):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        local_files_only=True,
        ignore_mismatched_sizes=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    return vae

def resize_crop_video(tensor, target_height, target_width):
    """将视频张量调整为目标尺寸并中心裁剪"""
    # tensor shape: (C, T, H, W)
    C, T, H, W = tensor.shape
    
    # 计算缩放因子
    scale = max(target_width / W, target_height / H)
    new_H, new_W = int(H * scale), int(W * scale)
    
    # 调整大小
    resized = torch.nn.functional.interpolate(
        tensor.permute(1, 0, 2, 3),  # (T, C, H, W)
        size=(new_H, new_W),
        mode='bilinear',
        align_corners=False
    ).permute(1, 0, 2, 3)  # 恢复为 (C, T, H, W)
    
    # 中心裁剪
    start_y = (new_H - target_height) // 2
    start_x = (new_W - target_width) // 2
    cropped = resized[:, :, start_y:start_y+target_height, start_x:start_x+target_width]
    
    return cropped

def encode_videos(vae, video_path, device):
    vae.requires_grad_(False)
    video_frame_transform = transforms.Compose([
        transforms.Lambda(lambda x: x/255 * 2 - 1),
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        transforms.Resize((480, 832), antialias=True),  
    ])
    # 加载视频并调整尺寸
    video_tuple = torchvision.io.read_video(video_path, pts_unit='sec')
    video_tensor = video_tuple[0]  # (T, H, W, C)
    # video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
    video_tensor = video_frame_transform(video_tensor).permute(1, 0, 2, 3)  # 调整为目标尺寸
    # print("video_tensor", video_tensor.min(), video_tensor.max())
    # 添加批次维度并发送到设备
    video_tensor = video_tensor.unsqueeze(0).to(vae.dtype).to(device)
    
    # 编码
    video_latents = vae.encode(video_tensor).latent_dist.mode()
    
    # 归一化潜变量
    video_latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, video_latents.dtype)
    video_latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, video_latents.dtype)
    
    # 正确归一化：(latent - mean) / std
    video_latents = (video_latents - video_latents_mean) / video_latents_std
    print("video_latents_after_vae", video_latents.min(), video_latents.max())
    return video_tensor, video_latents.detach()

def decode_videos(vae, video_latents, device):
    # 反归一化潜变量: latent = normalized*std + mean
    video_latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, video_latents.dtype)
    video_latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, video_latents.dtype)
    
    video_latents = video_latents * video_latents_std + video_latents_mean
    
    # 解码
    with torch.no_grad():
        video_tensor = vae.decode(video_latents, return_dict=False)[0]
    print("video_tensor_after_decode", video_tensor.min(), video_tensor.max())
    # 后处理
    
    video_processor = VideoProcessor(vae_scale_factor=8)
    video_np = video_processor.postprocess_video(video_tensor, output_type="np")
    # 准备输出视频
    cur_video = video_np[0]
    print("cur_video", cur_video.min(), cur_video.max())
    video_np = (cur_video * 255).astype(np.uint8)

    fps = 30
    
    # 创建并保存视频
    def make_frame(t):
        frame_index = min(int(t * fps), video_np.shape[0] - 1)
        return video_np[frame_index]
    
    video_clip = VideoClip(make_frame, duration=len(video_np)/fps)
    video_clip.write_videofile('recon_mode_mean_std.mp4', fps=fps, codec="libx264")
    
    return video_tensor

def reconstruct_loss_video(video_tensor1, video_tensor2):
    print("video_tensor1", video_tensor1.shape, video_tensor1.min(), video_tensor1.max())
    print("video_tensor2", video_tensor2.shape, video_tensor2.min(), video_tensor2.max())
    video_tensor1 =  video_tensor1[:,:,:149,:]
    loss = torch.nn.functional.mse_loss(video_tensor1, video_tensor2)
    print("reconstruct_loss_video", loss)
    return loss

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    vae_path = "/mnt/task_runtime/models/Wan2.1-T2V-1.3B-Diffusers/vae"
    video_path = "/mnt/task_runtime/dataset/vgg_filter/train/__0Fp4K-2Ew_000060.mp4"
    
    # 加载VAE
    vae = load_vae(vae_path).to(device)
    
    # 处理视频
    video_tensor, video_latents = encode_videos(vae, video_path, device)
    video_tensor_recon = decode_videos(vae, video_latents, device)
    loss = reconstruct_loss_video(video_tensor, video_tensor_recon)
    
    print(f"Reconstructed video shape: {video_tensor_recon.shape}")
    print(f"Reconstructed Loss: {loss}")