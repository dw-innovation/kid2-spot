## 4bit pre quantized models we support for 4x faster downloading + no OOMs.
#fourbit_models = [
#    "unsloth/llama-3-8b-bnb-4bit",
#    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 15 trillion tokens model 2x faster!
#    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
#    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
#    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # We also uploaded 4bit for 405b!
#    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",  # New Mistral 12b 2x faster!
#    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
#    "unsloth/mistral-7b-v0.3-bnb-4bit",  # Mistral v3 2x faster!
#    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
#    "unsloth/Phi-3-mini-4k-instruct",  # Phi-3 2x faster!d
#    "unsloth/Phi-3-medium-4k-instruct",
#    "unsloth/gemma-2-9b-bnb-4bit",
#    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
#]  # More models at https://huggingface.co/unsloth

#### Parameter set 1 ####
#LORA_R=16
#LORA_ALPHA=16
#LORA_RANDOM_STATE=3407
#EARLY_STOPPING=10
#EVAL_STEPS=50
#SAVE_STEPS=50
#AUTO_BATCH_SIZE=1
#BATCH_SIZE=8
#LEARNING_RATE=2e-4
#WEIGHT_DECAY=0.01
#LR_SCHEDULER='linear'
##########################

#### Parameter set 4 ####
#LORA_R=32
#LORA_ALPHA=64
#LORA_RANDOM_STATE=3407
#EARLY_STOPPING=10
#EVAL_STEPS=200
#SAVE_STEPS=200
#AUTO_BATCH_SIZE=1
#BATCH_SIZE=8
#LEARNING_RATE=1e-5
#WEIGHT_DECAY=0.01
#LR_SCHEDULER='cosine'
##########################

DATE=18112024
PARAMETER_VERSION=6
MODEL='llama-3-8b' # llama-3-8b # Meta-Llama-3.1-8B
VERSION_NAME='v17-1-2'
EPOCHS=10 # Default: 3
LORA_R=16
LORA_ALPHA=16
LORA_RANDOM_STATE=3407
EARLY_STOPPING=10
EVAL_STEPS=50
SAVE_STEPS=50
AUTO_BATCH_SIZE=1
BATCH_SIZE=32
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
LR_SCHEDULER='linear'
MAX_SEQ_LENGTH=2048  # Choose any! We auto support RoPE Scaling internally!
DTYPE="-1"  # Leave at -1 for auto-detection, set to Float16 for Tesla T4, V100, or Bfloat16 for Ampere+
LOAD_IN_4BIT=1  # Use 1 for True, 0 for False

OUTPUT_NAME="spot_${MODEL}_ep${EPOCHS}_training_ds_${VERSION_NAME}_param-${PARAMETER_VERSION}"
MODEL_NAME="unsloth/${MODEL}-bnb-4bit"
TRAIN_PATH="data/train_${VERSION_NAME}.tsv"
DEV_PATH="data/dev_${VERSION_NAME}.tsv"

CUDA_VISIBLE_DEVICES="1" screen -L -Logfile logs/${OUTPUT_NAME}_${DATE}.txt python -m train_unsloth \
  --output_name $OUTPUT_NAME \
  --model_name $MODEL_NAME \
  --train_path $TRAIN_PATH \
  --dev_path $DEV_PATH \
  --epochs $EPOCHS \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --random_state $LORA_RANDOM_STATE \
  --early_stopping $EARLY_STOPPING \
  --eval_steps $EVAL_STEPS \
  --save_steps $SAVE_STEPS \
  --auto_batch_size $AUTO_BATCH_SIZE \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --lr_scheduler $LR_SCHEDULER \
  --max_seq_length $MAX_SEQ_LENGTH \
  --dtype $DTYPE \
  --load_in_4bit $LOAD_IN_4BIT \
  --train \
  --test