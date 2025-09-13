#!/usr/bin/bash

set -ex
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1
# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_70b.toml"}

GPUS_PER_NODE=$NGPU
echo "getting SLURM_MASTER_ADDR:" ${SLURM_MASTER_ADDR}
echo "getting SLURM_MASTER_PORT:" ${SLURM_MASTER_PORT}
MASTER_ADDR=${SLURM_MASTER_ADDR}
MASTER_PORT=${SLURM_MASTER_PORT}
echo "SLURM_PROCID:$SLURM_PROCID"
echo "SLURM_NODEID:$SLURM_NODEID"
echo "SLURM_NNODES: $SLURM_NNODES"
NNODES=$SLURM_NNODES
NODE_RANK=${SLURM_NODEID}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node=${NGPU} \
         --rdzv_backend c10d \
         --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --local-ranks-filter ${LOG_RANK} \
        --role rank --tee 3 \
        torchtitan/train.py --job.config_file ${CONFIG_FILE}
