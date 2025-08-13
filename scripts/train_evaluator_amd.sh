# Number of GPUs to use
num_gpus=$(rocminfo | grep -c 'Device Type:             GPU')

# check if needs multi_gpu option
if [ "$num_gpus" == 1 ]; then
    gpu_option=""
else
    gpu_option="--multi_gpu"
fi

pip install -r requirements_amd.txt
accelerate config default

TOKENIZERS_PARALLELISM=true accelerate launch $gpu_option --num_processes=$num_gpus --mixed_precision="bf16" train_evaluator.py \
  --output_dir="ckpts/evaluation_classifier" \
  --gradient_checkpointing \
  --use_heuristics \
  --use_amd