# Potential Improvements

Model Performance = 0.3 * acc_easy + 0.7 * acc_challenge

- Advanced models: phi-1.5, phi-2.0
  - model
  - embedder
- Hyper-parameter tuning
  - max_len
  - prompt_type
  - N
  - top_k
  - top_k_reverse

# Preparing

- llm: https://huggingface.co/microsoft/phi-1_5
- embedder: https://huggingface.co/BAAI/bge-small-en-v1.5

```bash
python download.py # dir: model/
```

# Training & Validating & Testing

## ARC-Easy & phi 1.5

- Training

```bash
python eval_fewshot.py --data_path "data/ARC-Easy-train.jsonl" --device_id "0,1" \
                       --model "model/phi-1_5" --embedder "model/bge-small-en-v1.5" --start_index 0 --end_index 9999 \
                       --max_len 1024 --output_path "output_easy_train_phi_1_5" --overwrite False \
                       --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False
python acc.py --prediction_path "output_easy_train_phi_1_5"
```

- Validating

```bash
python eval_fewshot.py --data_path "data/ARC-Easy-validation.jsonl" --device_id "0,1" \
                       --model "model/phi-1_5" --embedder "model/bge-small-en-v1.5" --start_index 0 --end_index 9999 \
                       --max_len 1024 --output_path "output_easy_val_phi_1_5" --overwrite False \
                       --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False
python acc.py --prediction_path "output_easy_val_phi_1_5"
```

- Testing

```bash
python eval_fewshot.py --data_path "data/ARC-Easy-test.jsonl" --device_id "0,1" \
                       --model "model/phi-1_5" --embedder "model/bge-small-en-v1.5" --start_index 0 --end_index 9999 \
                       --max_len 1024 --output_path "output_easy_test_phi_1_5" --overwrite False \
                       --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False
python acc.py --prediction_path "output_easy_test_phi_1_5"
```
