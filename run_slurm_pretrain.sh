#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=logs/slurm/pretraining.%j.out
#SBATCH --nodes=8                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=96                   # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
##SBATCH --reservation=vultr-mi325x-torch # modify based on your reservation settings

echo "get first node"
# Get the list of nodes and the first node (master node)
# node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
COORDINATOR_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
master_node=$COORDINATOR_IP
# node_array=(${node_list})
# master_node=${node_array[0]}

# Set environment variables for distributed training
export SLURM_MASTER_ADDR=$master_node
export SLURM_MASTER_PORT=29565

# Optional: Print out the values for debugging
echo "MASTER_ADDR=$SLURM_MASTER_ADDR"
echo "MASTER_PORT=$SLURM_MASTER_PORT"

# Define the Docker image
export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8
echo $NCCL_IB_HCA

export DOCKER_IMAGE=${DOCKER_IMAGE:-}"docker.io/rocm/megatron-lm:v25.5_py310"
# Pull docker image
docker pull $DOCKER_IMAGE
# Setup your keys for HF and WADNB
# export HF_TOKEN="your hf key"
# export WANDB_API_KEY="wandb key"

export TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")

echo "TIME_STAMP=$TIME_STAMP"

export HF_TOKEN="your_hf_token"    # please set your HF token here
# Define the mount points
export TITAN_DIR=${PWD}                                      # change this path to Megatron-LM inside the docker
export HOST_MOUNT=${HOST_MOUNT:="/root/nfs_models/folder"}     # change this path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${CONTAINER_MOUNT:="/home/folder"}      # change this path to development workspace path inside the docker

export CONFIG_FILE="torchtitan/experiments/llama4/train_configs/llama4_17bx16e.toml"     
export REPO_ID="meta-llama/Llama-4-Scout-17B-16E"                                         
# export CONFIG_FILE="torchtitan/experiments/llama4/train_configs/llama4_17bx128e.toml" 
# export REPO_ID="meta-llama/Llama-4-Maverick-17B-128E"                                       
export TURBO_WHEEL="../Primus-Turbo/dist/primus_turbo-0.1.0+0858dbd-cp310-cp310-linux_x86_64.whl" # path to your local turbo wheel file
export TORCH_VERSION="2.9.0.dev20250825+rocm6.3"                                          # torch version to install in the container

docker ps -aq | xargs -r docker rm -f

if [ -e "/etc/libibverbs.d/bnxt_re.driver" ]; then
  echo "/etc/libibverbs.d exists and using broadcom."
  export IB_MOUNT_OPTIONS="-v /usr/bin:/usr/bin -v /etc/libibverbs.d/:/etc/libibverbs.d -v /usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/ -v /usr/local/lib:/usr/local/lib"
  
else
  echo "/etc/libibverbs.d does not exist not using ."
  export IB_MOUNT_OPTIONS=""
fi

# Run the Docker container with the script
srun bash -c ' docker ps -aq | xargs -r docker rm -f ; \
docker run --rm \
 --env SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR \
 --env SLURM_MASTER_PORT=$SLURM_MASTER_PORT \
 --env SLURM_PROCID=$SLURM_PROCID \
 --env SLURM_NODEID=$SLURM_NODEID \
 --env SLURM_NNODES=$SLURM_NNODES \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v $HOST_MOUNT:$CONTAINER_MOUNT \
 ${IB_MOUNT_OPTIONS} \
 $DOCKER_IMAGE /bin/bash -c \
 "echo $(date) ; \
    cd $CONTAINER_MOUNT/torchtitan-amd ; \
    pip3 install torch==${TORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --force-reinstall ; \
    pip3 install -r requirements.txt ; \
    pip3 install -e . ; \
    pip3 install --extra-index-url https://test.pypi.org/simple ${TURBO_WHEEL}; \
    pip install torchao ; \
    pip uninstall numpy -y && pip install numpy==1.26.4; \ 
    python scripts/download_hf_assets.py --assets tokenizer --repo_id ${REPO_ID} --hf_token="$HF_TOKEN" ;\
    CONFIG_FILE=${CONFIG_FILE} bash run_multinode_train.sh ; \
 echo $(date) 
 " 
 '
