# to fill in the following path to extract projector for the second tuning stage!
src_model=/root/autodl-tmp/2GraphGPT-main/checkpoints/stage_1
output_proj=/root/autodl-tmp/2GraphGPT-main/checkpoints/stage_1_projector/stage_1_projector.bin

python3.9 ./scripts/extract_graph_projector.py \
  --model_name_or_path ${src_model} \
  --output ${output_proj}