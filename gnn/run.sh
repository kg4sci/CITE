#!/bin/bash
# 获取开始时间
start_time=$(date +%s)

# 输出开始时间
echo "开始时间: $(date)"
# 指定 GPU，可以通过传参指定 GPU id（默认0）
#GPU=${1:-4}
GPU=3
export CUDA_VISIBLE_DEVICES=$GPU

#datasets=("cora" "pubmed" "citeseer" "wikics" "arxiv" "instagram" "reddit")
#models=("GAT" "SAGE" "GCN")
datasets=("chemistry")
models=("SAGE")
hidden_channels=(16 64 128 256)
num_layers=(2 3)
dropout=(0.3 0.5 0.6)

# 使用指定的 GPU 运行预处理脚本
# python st_embeddings.py#第一次要坐embedding 耗时1.3小时

for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for hidden in "${hidden_channels[@]}"
        do
            for num in "${num_layers[@]}"
            do
                for dr in "${dropout[@]}"
                do
                    python gnn.py --dataname $dataset --model $model --dropout $dr --hidden_channels $hidden --num_layers $num 
                done
            done
        done
    done
done
# 获取结束时间
end_time=$(date +%s)

# 输出结束时间
echo "结束时间: $(date)"

# 计算并输出总执行时间（秒）
execution_time=$((end_time - start_time))
echo "总共执行时间: $execution_time 秒"