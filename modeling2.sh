#!/usr/bin/env bash

# Function to train the model for a list of dataset types
train_model() {
  local dataset_types=("${!1}")  # Receive array of dataset types by name reference
  local model=$2
  local embedder=$3
  local prompt_type=$4
  local max_len=$5
  local N=$6
  local top_k=$7
  local top_k_reverse=$8

  # Create neccessary directories
  mkdir -p log

  # Create log file
  if [[ " ${dataset_types[*]} " =~ "Easy" ]]; then
    echo "[+] Model: Easy     , $model, $embedder, $prompt_type, $max_len, $N, $top_k, $top_k_reverse" >> running.log
  elif [[ " ${dataset_types[*]} " =~ "Challenge" ]]; then
    echo "[+] Model: Challenge, $model, $embedder, $prompt_type, $max_len, $N, $top_k, $top_k_reverse" >> running.log
  fi

  for dataset_type in "${dataset_types[@]}"; do
    local output_path="output_${dataset_type}_${model##*/}"  # Extract the model name and use it in the output path
    local data_path="data/ARC-${dataset_type}.jsonl"

    # Create the output directory
    mkdir -p "${output_path}"

    # Run the training process
    python eval_fewshot.py --data_path "$data_path" --device_id "0,1" \
                           --model "$model" --embedder "$embedder" --start_index 0 --end_index 9999 \
                           --max_len $max_len --output_path "${output_path}" --overwrite False \
                           --prompt_type "$prompt_type" --N $N --top_k $top_k --top_k_reverse $top_k_reverse | tee "log/${output_path}.log"
    # Print the result
    echo -n '[-] ' >> running.log
    python acc.py --prediction_path "${output_path}" 2>&1 >> running.log
  done
}

# Function to clean up generated files
clean_up() {
  echo -n "This will clean up all generated files. Are you sure? (Y/N) "
  read -r response
  case "$response" in
    [Yy])
      echo "Cleaning up generated files..."
      rm -rf output_*
      rm -rf log
      rm -f running.log
      # Add any additional file or directory clean up commands here
      ;;
    *)
      echo "Clean up cancelled."
      ;;
  esac
}

# Check if the first parameter is "clean" and call the clean_up function
if [ "${1}" == "clean" ]; then
  clean_up
  exit 0
fi

# Variables to hold the parameter values
MODEL="model/phi-2"
EMBEDDER="model/bge-small-en-v1.5"
PROMPT_TYPE="v2.0"
MAX_LEN=2048
N=8
TOP_K=True
TOP_K_REVERSE=True

# Arrays of dataset types
EASY_DATASET_TYPES=("Easy-train" "Easy-validation" "Easy-test")
CHALLENGE_DATASET_TYPES=("Challenge-train" "Challenge-validation" "Challenge-test")

# Delete the log file if it exists
rm -rf running.log

# Call the function with parameters for ARC-Easy and ARC-Challenge dataset types
## Model: phi-2, embedder: bge-small-en-v1.5, prompt_type: v2.0, max_len: 2048, N: 8, top_k: True, top_k_reverse: True
train_model EASY_DATASET_TYPES[@] "$MODEL" "$EMBEDDER" "$PROMPT_TYPE" $MAX_LEN $N $TOP_K $TOP_K_REVERSE
train_model CHALLENGE_DATASET_TYPES[@] "$MODEL" "$EMBEDDER" "$PROMPT_TYPE" $MAX_LEN $N $TOP_K $TOP_K_REVERSE
