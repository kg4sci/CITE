# to fill in the following path to extract projector for the second tuning stage!
export PYTHONPATH=$(pwd)/graphgpt:$PYTHONPATH


output_model=./checkpoints/stage_2
datapath=./data/eval/arxiv_test_instruct_cot.json
graph_data_path=./graph_data/all_graph_data.pt
res_path=./output_stage_2_arxiv_nc
start_id=0
end_id=20000
num_gpus=1

python3.9 ../../graphgpt/eval/run_graphgpt.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}