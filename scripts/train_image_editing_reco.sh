# Number of GPUs to use
num_gpus=$(nvidia-smi --list-gpus | wc -l)

# check if needs multi_gpu option
if [ "$num_gpus" == 1 ]; then
    gpu_option=""
else
    gpu_option="--multi_gpu"
fi

pip install -r requirements.txt
accelerate config default

TOKENIZERS_PARALLELISM=true accelerate launch $gpu_option --num_processes=$num_gpus --mixed_precision="fp16" train_image_editing_reco.py \
  --pretrained_model_name_or_path="vinesmsuic/magicbrush-jul7" \
  --train_batch_size=64 \
  --num_train_epochs=200 \
  --validation_epochs=20 \
  --checkpointing_epochs=20 \
  --enable_xformers_memory_efficient_attention \
  --use_ema \
  --scorer_ckpt_path="ckpts/evaluation_classifier" \
  --decoder_type="classifier"