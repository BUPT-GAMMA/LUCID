# LUCID

This repository contains the research code for **LUCID**, a hallucination detector for large-language-model-based knowledge graph (KG) reasoning.

Paper: [Detecting Hallucinations for Large Language Model-based Knowledge Graph Reasoning](http://www.shichuan.org/doc/233.pdf)  

## What this guide reproduces

Given a question `Q`, a retrieved KG subgraph `S_Q`, and an LLM response `R_Q`, LUCID predicts whether `R_Q` contains a hallucination:

```text
D(Q, S_Q, R_Q) -> probability of hallucination
```

The paper evaluates three KG reasoning frameworks on three datasets:

- Reasoning frameworks: **Readi**, **ToG**, and **StructGPT**
- Evaluation datasets: **GrailQA**, **WebQSP**, and **QALD-10**
- Training setting: the paper trains the GINE detector with Readi outputs on **CWQ**
- Open-source generation model: **Qwen2.5-7B-Instruct**
- Relation/query encoder: **sentence-transformers/all-MiniLM-L6-v2**

The repository includes Readi label files under `benchmark/data/`, but it does not include all original benchmark data, all framework outputs, model checkpoints, or the external dataset loader used by the paper. Therefore, the complete paper tables require obtaining those inputs separately.

## Reproduction pipeline

```text
KG reasoning framework outputs
        │
        ▼
Answer evaluation and hallucination labels
        │
        ▼
LLM attention extraction
        │
        ▼
Entity/relation attention matrices
        │
        ├── question-relation cosine similarity
        ▼
Feature-enriched KG subgraphs
        │
        ▼
GINE training
        │
        ▼
Hallucination probabilities and metrics
```

Run all commands from the repository root, `/path/to/LUCID`.

## 1. Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install a CUDA-compatible PyTorch build first.
pip install transformers sentence-transformers numpy pandas scipy scikit-learn \
    matplotlib networkx tqdm SPARQLWrapper ollama
```

Install PyTorch Geometric using the official command for your installed PyTorch and CUDA versions. The attention extraction and GINE training stages are intended for GPU execution.


## 2. Extract LLM attention features

Configure the following values in `core/main_embedding.py`:

```python
input_path = "run_data/raw/outputs.jsonl"
output_path = "run_data/attention/attention.jsonl"
saveid_path = "run_data/attention/finished_ids.txt"
```

Set the local Hugging Face identifier for the selected model. Then run:

```bash
python3 core/main_embedding.py --model_name <model_name>
```

For every sample, the intended output is one JSON object containing:

```json
{
  "id": "sample-id",
  "attention_scores": [
    {
      "layer_id": 0,
      "head_id": 0,
      "span_scores": {
        "type": "relation",
        "content": []
      }
    }
  ]
}
```

Important: in the current snapshot, `main_embedding.py` constructs `sample_attention_scores` but does not write it to `output_path`. Add the JSONL write operation before running this stage, or restore it from the version used for the paper.

## 3. Aggregate attention matrices

Configure `core/cal_embedding.py`:

```python
input_path = "run_data/attention/attention.jsonl"
output_path = "run_data/embeddings/embeddings.jsonl"
right_path = "run_data/labels/right.jsonl"
error_path = "run_data/labels/error.jsonl"
```

Then run:

```bash
python3 core/cal_embedding.py
```

The output stores `entity_content` and `relation_content`. Each entry contains the entity/relation text and a layer-by-head attention matrix.

## 4. Compute question-relation similarity

Configure `process_data/cal_question_relation_sim.py`:

```python
file_path = "run_data/raw/outputs.jsonl"
```

Also replace the empty path in `cal_sim()` with the desired output file, for example:

```python
with open("run_data/similarity/relation_similarity.jsonl", "a") as f:
```

Then run:

```bash
python3 process_data/cal_question_relation_sim.py
```

Each sample should produce a record with an `id` and a `similarity` list containing relation names and cosine similarity scores.

## 5. Train the GINE detector

The paper configuration is:

| Parameter | Paper value |
| --- | ---: |
| GINE layers | 2 |
| Hidden channels | 512 |
| Learning rate | `1e-3` |
| Epochs | 300 |
| Batch size | 32 |
| Loss | Binary cross-entropy |
| Threshold | ROC geometric mean |

In `train/train_gine.py`, configure:

```python
jsonl_path = "run_data/raw/train_outputs.jsonl"
embedding_path = "run_data/embeddings/train_embeddings.jsonl"
label_path = "run_data/labels/train_labels.jsonl"
sim_file_path = "run_data/similarity/train_similarity.jsonl"
save_dir = "run_data/checkpoints"
```

Set `learning_rate = 1e-3` to match the paper. The current source default is `1e-4`. Also replace both empty `save_dir` assignments in the training loop; `os.makedirs("")` is invalid.

The paper trains on Readi/CWQ and evaluates on the other framework/dataset combinations. To reproduce a specific table, use exactly the same framework and dataset split as that table.

Run:

```bash
python3 train/train_gine.py --use_edge_weight True
```

## 6. Evaluate a checkpoint

Configure the paths in `train/load_gine.py` for the test data, embeddings, labels, similarity file, and checkpoint. Then run:

```bash
python3 train/load_gine.py --use_edge_weight True
```

The evaluator reports:

- Accuracy (ACC)
- ROC-AUC (AUC)
- Precision
- Recall
- F1
- Pearson correlation coefficient (PCC)
- AVG = `(ACC + AUC + PCC) / 3`

The current evaluator refers to `thresh` without defining it, and the checkpoint does not persist the optimal threshold. For a faithful reproduction, save the training threshold selected by the ROC geometric-mean criterion and pass the same threshold to evaluation.

