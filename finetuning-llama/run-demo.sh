source activate-conda.sh
activate_conda
conda activate lora_env

LLM_PATH='/home/intel-admin/plischwe/schoolnet/llama3.1-8b-Instruct-INT4'
DEV=$3
LORA_PATH=$2
TRANSCRIPTION=$1

python lora_inf.py --transcription $TRANSCRIPTION --llm_path $LLM_PATH --lora_path $LORA_PATH --device $DEV
