################# Run on the machine where DeepSpeed is compiled and installed from source ######################
# Update GCC and G++ versions (if needed)
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7 g++-7
# Update system's default gcc and g++ pointers
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc

# Install DeepSpeed from source
# Set the TORCH_CUDA_ARCH_LIST parameter according to your GPU's actual situation (see previous page for how to check);
# If you need to use NVMe Offload, set the parameter DS_BUILD_UTILS=1;
# If you need to use CPU Offload optimizer parameters, set the parameter DS_BUILD_CPU_ADAM=1;
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="7.5" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 
python setup.py build_ext -j8 bdist_wheel
# This will generate a file like dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl,
# Install on other nodes: pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl.

# Install Transformers from source
# https://huggingface.co/docs/transformers/installation#install-from-source
pip install git+https://github.com/huggingface/transformers


################# launch.slurm script (modify template values as needed) ######################
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
