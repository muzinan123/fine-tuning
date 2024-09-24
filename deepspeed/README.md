# DeepSpeed Framework Installation Guide
## Update GCC and G++ Versions (if needed)
First, add the necessary PPA repository, then update gcc and g++:

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7 g++-7
```

# Update the system's default gcc and g++ pointers:
```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc
```

## Create an Isolated Anaconda Environment
# If you want to isolate the environment, it's recommended to use the clone method to create a dedicated Anaconda environment for DeepSpeed:
```bash
conda create -n deepspeed --clone base
```

## Install Transformers and DeepSpeed
### Install Transformers from Source
# Follow the official documentation and install Transformers using the following command:
```bash
pip install git+https://github.com/huggingface/transformers
```

### Install DeepSpeed from Source
# Set the TORCH_CUDA_ARCH_LIST parameter according to your GPU's actual situation. If you need to use CPU Offload optimizer parameters, set the parameter DS_BUILD_CPU_ADAM=1; if you need to use NVMe Offload, set the parameter DS_BUILD_UTILS=1:
```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="7.5" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

## Note: Do not clone and install DeepSpeed source code within your project directory to avoid accidental commits.

### Training T5 Series Models Using DeepSpeed
 Single GPU training script: train_on_one_gpu.sh
 Distributed training script: train_on_multi_nodes.sh