#!/bin/bash

export CUDA_VISIBLE_DEVICE=0
export MODEL='conv'
data_dir=/home/crss/Desktop/Mamun/Features
datasets=('SSN_0_dB' 'SSN_5_dB' 'SSN_10_dB')
res_dir=/home/crss/Desktop/Mamun/res
layer_nums=(7)
kernel_sizes=(7)
kernel_nums=(64)
bridges=('nothing' 'add' 'mul')
paddings=('same' 'causal')
for dataset in "${datasets[@]}"; do
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
done
