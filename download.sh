#!/bin/bash
# Set token from command line argument
export HF_TOKEN=<hf_token>

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli is not installed. Please install it first."
    exit 1
fi

# Function to create directory if it doesn't exist
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create directory $1"
            exit 1
        fi
    fi
}

# Function to download model
download_model() {
    local repo=$1
    local dir=$2
    local repo_type=${3:-"model"}
    
    echo "Downloading $repo to $dir..."
    create_dir "$dir"
    
    huggingface-cli download "$repo" --local-dir "$dir" --repo-type "$repo_type" --token "$HF_TOKEN"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download $repo"
        exit 1
    fi
    echo "Successfully downloaded $repo"
}

# Download models
download_model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" "./Wan2.1-T2V-1.3B-Diffusers"
download_model "Wan-AI/Wan2.1-T2V-14B-Diffusers" "./Wan2.1-T2V-14B-Diffusers"
download_model "Wan-AI/Wan2.1-T2V-1.3B" "./Wan2.1-T2V-1.3B"
download_model "stabilityai/stable-audio-open-1.0" "./stable-audio-open-1.0" "model"
download_model "Xiaodong/FVD_I3D" "./i3d/"
download_model "lukewys/laion_clap" "./laion_clap"
download_model "stabilityai/stable-diffusion-3.5-medium" "./stable-diffusion-3.5-medium"
download_model "Qwen/Qwen2-Audio-7B-Instruct" "./Qwen2-Audio-7B-Instruct"
download_model "hpcai-tech/OpenSora-VAE-v1.2" "./OpenSora-VAE-v1.2"
download_model "Qwen/Qwen2.5-VL-72B-Instruct" "./Qwen2.5-VL-72B-Instruct"
download_model "Qwen/Qwen2.5-7B-Instruct" "./Qwen2.5-7B-Instruct"
download_model "Qwen/Qwen2.5-VL-32B-Instruct" "./Qwen2.5-VL-32B-Instruct"
download_model "Qwen/Qwen2.5-VL-7B-Instruct" "./Qwen2.5-VL-7B-Instruct"
download_model "Qwen/Qwen2.5-72B-Instruct" "./Qwen2.5-72B-Instruct"
download_model "Qwen/Qwen2.5-Omni-7B" "./Qwen2.5-Omni-7B"
mkdir -p clip
mkdir -p synchformer
mkdir -p imagebind
wget -O imagebind/imagebind_huge.pth https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
wget -O clip/ViT-L-14.pt https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth -O ./synchformer/synchformer_state_dict.pth
echo "All downloads completed successfully!"