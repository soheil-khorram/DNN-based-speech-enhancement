#!/bin/bash

# Author: Soheil Khorram
# License: Simplified BSD

# A bash file designed to sweep over different configurations of the network
# Usage: ./run.sh dataset_path result_path
#             It reads train, development and test samples from the dataset_path and
#             stores the results (trained models and final metrics) in the result_path

export CUDA_VISIBLE_DEVICE=0
export MODEL='conv'
dataset=$1
res_dir=$2
layer_nums=(7)
kernel_sizes=(7)
kernel_nums=(64)
bridges=('nothing' 'add' 'mul')
paddings=('same' 'causal')
for layer_num in "${layer_nums[@]}"; do
    for kernel_size in "${kernel_sizes[@]}"; do
        for kernel_num in "${kernel_nums[@]}"; do
            for bridge in "${bridges[@]}"; do
                for padding in "${paddings[@]}"; do
                    exp_name=${MODEL}_ln_${layer_num}_ks_${kernel_size}_kn_${kernel_num}_brg_${bridge}_pd_${padding}
                    command="\
                        python main.py \
                            -data-dir $data_dir \
                            -dataset $dataset \
                            -res-dir $res_dir \
                            -layer-num $layer_num \
                            -kernel-size $kernel_size \
                            -kernel-num $kernel_num \
                            -apply-end-relu 0 \
                            -bridge $bridge \
                            -padding $padding \
                            -exp-name $exp_name"
                    echo -e $command
                    $command
                done
            done
        done
    done
done
