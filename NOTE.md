# Potential Improvements

Model Performance = 0.3 * acc_easy + 0.7 * acc_challenge

- Model & Embedder:
  - model: phi-1.5, phi-2.0
  - embedder: bge-small-en-v1.5
- Hyper-parameter tuning
  - max_len: 516, 1024, 2048
  - prompt_type: v2.0, v2.1
  - N: 4, 8, 16
  - top_k: True, False
  - top_k_reverse: False, True

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
                       --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False | tee log/output_easy_train_phi_1_5.log
{'N': 8,
 'data_path': 'data/ARC-Easy-train.jsonl',
 'device_id': '0,1',
 'embedder': 'model/bge-small-en-v1.5',
 'end_index': 9999,
 'max_len': 1024,
 'model': 'model/phi-1_5',
 'output_path': 'output_easy_train_phi_1_5',
 'overwrite': False,
 'prompt_type': 'v2.0',
 'start_index': 0,
 'top_k': True,
 'top_k_reverse': False}
Loaded model/phi-1_5.
loaded model/bge-small-en-v1.5.
load 2251 demonstrations from data/ARC-Easy-train.jsonl
...
Saving results to output_easy_train_phi_1_5/9000.jsonl
Saving results to output_easy_train_phi_1_5/9001.jsonl

python acc.py --prediction_path "output_easy_train_phi_1_5"
(output_easy_train_phi_1_5) Acc: 0.9911150599733451 (2231 / 2251)
```

- Validating

```bash
python eval_fewshot.py --data_path "data/ARC-Easy-validation.jsonl" --device_id "0,1" \
                       --model "model/phi-1_5" --embedder "model/bge-small-en-v1.5" --start_index 0 --end_index 9999 \
                       --max_len 1024 --output_path "output_easy_val_phi_1_5" --overwrite False \
                       --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False | tee log/output_easy_val_phi_1_5.log

python acc.py --prediction_path "output_easy_val_phi_1_5"
(output_easy_val_phi_1_5) Acc: 0.6403508771929824 (365 / 570)
```

- Testing

```bash
python eval_fewshot.py --data_path "data/ARC-Easy-test.jsonl" --device_id "0,1" \
                       --model "model/phi-1_5" --embedder "model/bge-small-en-v1.5" --start_index 0 --end_index 9999 \
                       --max_len 1024 --output_path "output_easy_test_phi_1_5" --overwrite False \
                       --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False | tee log/output_easy_test_phi_1_5.log

python acc.py --prediction_path "output_easy_test_phi_1_5"
(output_easy_test_phi_1_5) Acc: 0.6675084175084175 (1586 / 2376)
```

## ARC-Challenge & phi 1.5

- Training

```bash
python eval_fewshot.py --data_path "data/ARC-Challenge-train.jsonl" --device_id "0,1" \
                       --model "model/phi-1_5" --embedder "model/bge-small-en-v1.5" --start_index 0 --end_index 9999 \
                       --max_len 1024 --output_path "output_challenge_train_phi_1_5" --overwrite False \
                       --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False | tee log/output_challenge_train_phi_1_5.log

python acc.py --prediction_path "output_challenge_train_phi_1_5"
(output_challenge_train_phi_1_5) Acc: 0.9535299374441466 (1067 / 1119)
```

- Validating

```bash
python eval_fewshot.py --data_path "data/ARC-Challenge-validation.jsonl" --device_id "0,1" \
                       --model "model/phi-1_5" --embedder "model/bge-small-en-v1.5" --start_index 0 --end_index 9999 \
                       --max_len 1024 --output_path "output_challenge_val_phi_1_5" --overwrite False \
                       --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False | tee log/output_challenge_val_phi_1_5.log

python acc.py --prediction_path "output_challenge_val_phi_1_5"
(output_challenge_val_phi_1_5) Acc: 0.46153846153846156 (138 / 299)
```

- Testing

```bash
python eval_fewshot.py --data_path "data/ARC-Challenge-test.jsonl" --device_id "0,1" \
                       --model "model/phi-1_5" --embedder "model/bge-small-en-v1.5" --start_index 0 --end_index 9999 \
                       --max_len 1024 --output_path "output_challenge_test_phi_1_5" --overwrite False \
                       --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False | tee log/output_challenge_test_phi_1_5.log

python acc.py --prediction_path "output_challenge_test_phi_1_5"
```