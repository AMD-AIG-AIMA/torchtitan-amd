# How run trainining on AMD GPU

## Prepare docker
```bash
docker pull docker.io/rocm/megatron-lm:v25.5_py310

docker run -it --device /dev/dri --device /dev/kfd \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v /root/nfs_models/john:/home/john \
    -w /workspace \
    --name titan_test_your_name \
     docker.io/rocm/megatron-lm:v25.5_py310 
```
## Step 1 : Git clone TorchTitan and install

```bash
https://github.com/AMD-AIG-AIMA/torchtitan-amd 

git clone https://github.com/AMD-AIG-AIMA/torchtitan-amd.git
cd torchtitan-amd
git checkout dev/primus_turbo

pip install torchao
pip3 install torch==2.9.0.dev20250825+rocm6.3 torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --force-reinstall
pip3 install -r requirements.txt
pip3 install -e .
```


## Step 2 : Git clone Primus-Turbo and install
```bash 
https://github.com/AMD-AIG-AIMA/Primus-Turbo 

git clone https://github.com/AMD-AIG-AIMA/Primus-Turbo.git --recursive
cd Primus-Turbo
pip3 install -r requirements.txt
python3 -m build --wheel --no-isolation
pip3 install --extra-index-url https://test.pypi.org/simple ./dist/primus_turbo-XXX.whl # Please use the whl package name genereated under /dist folder

```


## Step 3: Go back to TorchTitan and run Llama 4 training 
```bash
cd torchtitan-amd
python scripts/download_hf_assets.py --assets tokenizer --repo_id meta-llama/Llama-4-Scout-17B-16E --hf_token="your_hf_token" # Please make sure to have the llam4 hf repo access
TORCHTITAN_USE_FP8_GROUPED_MM=1 CONFIG_FILE="torchtitan/experiments/llama4/train_configs/llama4_17bx16e.toml" ./run_train.sh
```

## Other models supported in torchtitan 
You can also run other models supported in torchtian, such as llama3, deepseek etc. For exmaple,
```bash 
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh
```