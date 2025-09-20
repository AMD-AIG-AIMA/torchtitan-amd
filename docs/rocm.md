# How run trainining on AMD GPU

## Single node training 

### Prepare docker
```bash
export DOCKER_IMAGE=${DOCKER_IMAGE:="docker.io/rocm/megatron-lm:v25.5_py310"}
export HOST_MOUNT=${HOST_MOUNT:="/root/nfs_models/your_folder_name"}
export CONTAINER_MOUNT=${CONTAINER_MOUNT:="/workspace"}
export CONTAINER_NAME=${CONTAINER_NAME:="titan_training_username"} 
docker pull $DOCKER_IMAGE
docker run -it --device /dev/dri --device /dev/kfd \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v $HOST_MOUNT:$CONTAINER_MOUNT \
    -w /workspace \
    --name $CONTAINER_NAME \
     $DOCKER_IMAGE 
```

### Step 1 : Git clone and install TorchTitan inside the container
torchtitan-amd [Github Link](https://github.com/AMD-AIG-AIMA/torchtitan-amd) 
```bash
cd /workspace
git clone https://github.com/AMD-AIG-AIMA/torchtitan-amd.git
cd torchtitan-amd
git checkout dev/primus_turbo
pip install torchao
pip3 install torch==2.9.0.dev20250825+rocm6.3 torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --force-reinstall
pip3 install -r requirements.txt
pip3 install -e .
```

### Step 2 : Git clone and install Primus-Turbo inside the container
Primus-Turbo [Github Link](https://github.com/AMD-AIG-AIMA/Primus-Turbo) 
```bash 
cd /workspace
git clone https://github.com/AMD-AIG-AIMA/Primus-Turbo.git --recursive
cd Primus-Turbo
git checkout dev/fp8
pip3 install -r requirements.txt
pip uninstall numpy -y && pip install numpy==1.26.4
pip3 install --no-build-isolation -e . -v # for developer install
python3 -m build --wheel --no-isolation   # build the whl package

pip3 install --extra-index-url https://test.pypi.org/simple ./dist/primus_turbo-XXX.whl # Please use the whl package name genereated under /dist folder

```


### Step 3: Go back to TorchTitan and run Llama 4 training 
```bash
cd /workspace/torchtitan-amd
export HF_TOKEN="your_hf_token"
python scripts/download_hf_assets.py --assets tokenizer --repo_id meta-llama/Llama-4-Scout-17B-16E --hf_token="$HF_TOKEN" # Please make sure to have the llam4 hf repo access
CONFIG_FILE="torchtitan/experiments/llama4/train_configs/llama4_17bx16e.toml" ./run_train.sh
```

### Other models supported in torchtitan 
You can also run other models supported in torchtian, such as llama3 and deepseek etc. For exmaple,
```bash 
export HF_TOKEN="your_hf_token"
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer --hf_token="$HF_TOKEN"
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh
```
```bash 
export HF_TOKEN="your_hf_token"
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token="$HF_TOKEN"
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh
```

## Multinode training 
To run multinode training in a Slurm environment, just use the following command (please modify the `run_slurm_pretrain.sh` file as necessary):
```bash 
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_token"
sbatch run_slurm_pretrain.sh
```
