#!/bin/bash

# Exit on error
set -e

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Warning: $1 is not installed"
        return 1
    fi
    return 0
}

# Function to check if a directory exists
check_dir() {
    if [ ! -d "$1" ]; then
        echo "Warning: Directory $1 does not exist"
        return 1
    fi
    return 0
}

echo "Running setup"

# 添加 alias
{
    echo "alias ll='ls -alF'"
    echo "alias conductor='aws --endpoint-url https://conductor.data.apple.com'"
    echo "alias tailf='tail -f'"
} >> ~/.bash_profile || echo "Warning: Failed to add aliases"

# 激活 bash_profile
source ~/.bash_profile || echo "Warning: Failed to source bash_profile"


# 创建 Conda 环境
# 检查是否安装了conda
echo "Checking for Conda installation..."

check_command conda || {
    echo "Warning: conda is not installed. Installing Miniconda..."
    
    # 下载 Miniconda 安装包
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # 安装 Miniconda
    bash miniconda.sh -b -p $HOME/miniconda
    
    # 设置环境变量，添加 Miniconda 到 PATH
    export PATH="$HOME/miniconda/bin:$PATH"
    
    # 初始化conda
    $HOME/miniconda/bin/conda init
    
    # 激活conda
    source ~/.bashrc

    echo "Miniconda installed successfully."
}

# 创建新的 Conda 环境

source /root/miniconda/etc/profile.d/conda.sh || echo "Warning: Failed to source conda.sh"
# 接受 main 仓库的条款 (主要用于 Python 包)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# 接受 r 仓库的条款 (如果你用 R 语言的话)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n bridgedit python=3.10 -y
conda activate bridgedit || echo "Warning: Failed to activate conda environment"

# 写入环境激活命令到 bash_profile
echo "conda activate bridgedit" >> ~/.bash_profile || echo "Warning: Failed to add conda activation to bash_profile"

# 安装 pip 包
echo "Installing Python packages..."
pip install --upgrade pip

# 安装所有 pip 包
pip_packages=(
    "matplotlib"
    "langdetect"
    "lingua-language-detector"
    "jupyter"
    "pandas"
    "tqdm"
    "jsonpath_ng"
    "python-Levenshtein"
    "pystemmer"
    "pyarrow"
    "-U huggingface_hub[cli]"
    "pafy==0.5.5"
    "youtube-dl"
    "nvitop"
    "blobfile"
    "dtaidistance"
    "laion_clap"
    "pyloudnorm"
    "openai"
    "torch==2.6.0"
    "torchvision==0.21.0"
    "torchaudio==2.6.0"
    "einops"
    "transformers"
    "diffusers"
    "accelerate"
    "pytorch-lightning"
    "xfuser>=0.4.1"
    "moviepy"
    "torchsde"
    "git+https://github.com/openai/CLIP.git"
    "decord"
    "av"
    "omegaconf"
    "dataset"
    "peft"
    "scikit-image"
    "deepspeed"
)

for package in "${pip_packages[@]}"; do
    echo "Installing $package..."
    pip install $package || echo "Warning: Failed to install $package"
done
pip uninstall xformers
pip install flash-attn==2.7.4.post1

# 安装系统包
echo "Installing system packages..."
apt-get install -y ffmpeg sox || echo "Warning: Failed to install ffmpeg or sox"

# CUDA 安装
echo "Installing CUDA..."
CUDA_INSTALLER="cuda_12.3.0_545.23.06_linux.run"
wget "https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/$CUDA_INSTALLER" || echo "Warning: Failed to download CUDA installer"

sh "$CUDA_INSTALLER" --silent --toolkit --toolkitpath=/usr/local/cuda-12.3 || echo "Warning: Failed to install CUDA"

# 将 CUDA 环境变量写入 ~/.bash_profile
{
    echo "export CUDA_HOME=/usr/local/cuda-12.3"
    echo "export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CUDA_HOME/lib64"
} >> ~/.bash_profile || echo "Warning: Failed to add CUDA environment variables"



# 配置 hvgc
conda create --name hvgc python=3.10 -y
conda activate hvgc
pip install transformers==4.52.0
pip install vllm==0.8.4
pip install qwen_vl_utils
# 可选：启动 Jupyter
# nohup /miniconda/envs/myenv/bin/jupyter notebook --port "$VIZ_PORT" --allow-root --no-browser --ip='*' --NotebookApp.token='' --NotebookApp.password='' &