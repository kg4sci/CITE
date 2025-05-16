# to fill in the following path to extract projector for the first tuning stage!
src_model=./checkpoints0501/stage_1
output_proj=./checkpoints0501/stage_1_projector/stage_1_projector.bin

python3 ./scripts/extract_graph_projector.py \
  --model_name_or_path ${src_model} \
  --output ${output_proj}