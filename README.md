# Adaptive Speculative Reasoning

Welcome to the Adaptive Speculative Reasoning project! This repository contains the infrastructure, prompt designs, evaluation harnesses, and speculative model heads to explore structured Chain-of-Thought (CoT) generation, step-wise confidence scoring, and automated correctness verification.

---

## Current Project Phase
We have successfully completed:
1. **Phase 1 (Generation Infrastructure & Prompting)**
2. **Phase 2 (Confidence Scoring & NLI Correctness Checking)**

We are now currently working on a mitigation plan owing to Token Log Probability's failure to correctly score step level confidence.
Thus we have another sub-phase, mb sorry :
 - **Phase 2.1 (UHead Training & Core Model Migration)**
---

## 1. UHead Architecture
* **UHead** is a lightweight, multi-layer perceptron (MLP) binary classifier head trained on top of the frozen main model's final-layer token hidden states to predict the probability of step-wise correctness.
* The architecture utilizes sigmoid activation to yield a certainty score in the range `[0, 1]`.
* Basically training a very basic neural network to look at the hidden state vectors of the final token of each step at the last layer [-1] to determine whether or not the model is confident in this reasoning step. Should've done this first, would've saved time and effort ;_;

---

## 2. Core Codebase Migrations & Integrations

### Transition from Ollama to Hugging Face
* **Motivation**: The previous setup called the Ollama API, which is an opaque HTTP endpoint. Ollama does not expose token-level hidden states, making it impossible to perform step-wise UHead scoring or hook activation.
* **Unified Model Loader ([model_loader.py](file:///Users/vbaalaadityaa/Downloads/summer-research/adaptive-speculative-reasoning/src/confidence/model_loader.py))**: Loads `Mistral-7B-Instruct` locally. To match the low memory footprint of Ollama (~4–5 GB), it implements 4-bit NF4 quantization using `bitsandbytes` (falling back to float16 on CPU/macOS if unavailable).
* **Shared Singleton Instance**: Text generation ([model_client.py](file:///Users/vbaalaadityaa/Downloads/summer-research/adaptive-speculative-reasoning/src/model_client.py)) and hidden state extraction share the exact same model instance. This guarantees that hidden-state distributions are identical at both training time and inference time.
* **Forward Hooks ([extractor.py](file:///Users/vbaalaadityaa/Downloads/summer-research/adaptive-speculative-reasoning/src/uhead/extractor.py))**: Introduces a context manager that registers PyTorch forward hooks on the model's final layer, extracting the hidden state of the final token in each reasoning step.
* **Compatibility Stubs**: Kept Ollama stubs (`check_ollama_running`, `list_available_models`) as mocks in `model_client.py` so that external scripts do not break.

### Remote & Pull Changes (Nithilan's commits)
* **Timeout Extensions**: Raised Ollama-client/generation timeouts from `120s` to `600s` to prevent timeout failures during slow/multi-step reasoning generation.
* **Platform/Device Auto-mapping**: Standardized device mapping across CPU, CUDA, and MPS via `device_map="auto"` inside Hugging Face causal loaders.

---

## 3. Overall Flow of Operations

Below is the sequential flow of commands to run the entire pipeline:

### Step A: Generate Base Reasoning Chain-of-Thought (CoT)
Generates structured reasoning steps for a dataset and caches formatting/fallback results.
```bash
# Run batch generation on 50 questions
python src/batch_runner.py --dataset hotpotqa --n 50
```

### Step B: Run Unsupervised Confidence Scoring
Calculates numerical baseline confidence metrics.
```bash
# Evaluate step confidence using raw next-token log-probabilities
python src/confidence/run_scoring.py --dataset hotpotqa --n 50 --method logprob
```

### Step C: Run Correctness Verification (NLI Grading)
Evaluates step logic using a Natural Language Inference (DeBERTa-v3-mnli) cross-encoder model against gold-standard premises.
```bash
# Label steps as 'correct' (entailed), 'incorrect' (contradicted), or 'ambiguous' (neutral)
python src/correctness/run_correctness.py --dataset hotpotqa
```

### Step D: Train and Fine-tune UHead

#### 1. Cache PRM800K Hidden States
Pre-extracts hidden states from the large-scale math reasoning dataset to accelerate training.
```bash
python src/uhead/train.py --build-cache --cache-limit 50000
```

#### 2. Pretrain UHead (Stage 1)
Trains the UHead MLP classifier on the cached PRM800K hidden states.
```bash
python src/uhead/train.py --pretrain --epochs 3
```

#### 3. Build Domain Fine-Tuning Caches
Caches the hidden states of domain datasets (GSM8K, HotpotQA) using the correctness labels graded by the NLI step (Step C). Ambiguous/unknown labels are automatically filtered out.
```bash
python src/uhead/train.py --finetune-cache --dataset gsm8k
python src/uhead/train.py --finetune-cache --dataset hotpotqa
```

#### 4. Fine-tune UHead (Stage 2)
Fine-tunes the pretrained UHead on target-domain labels.
```bash
python src/uhead/train.py --finetune --epochs 2
```

---

## 4. Walkthrough for sir :

```bash
source venv/bin/activate
pip install -r requirements.txt
```

#### A. Run Correctness Verification (NLI Grading)
Evaluates step logic using a Natural Language Inference (DeBERTa-v3-mnli) cross-encoder model against gold-standard premises.
```bash
# Label steps as 'correct' (entailed), 'incorrect' (contradicted), or 'ambiguous' (neutral)
python src/correctness/run_correctness.py --dataset musique
```

#### 1. Cache PRM800K Hidden States
Pre-extracts hidden states from the large-scale math reasoning dataset to accelerate training.
```bash
python src/uhead/train.py --build-cache --cache-limit 50000
```

#### 2. Pretrain UHead (Stage 1)
Trains the UHead MLP classifier on the cached PRM800K hidden states.
```bash
python src/uhead/train.py --pretrain --epochs 3
```

#### 3. Build Domain Fine-Tuning Caches
Caches the hidden states of domain datasets (GSM8K, HotpotQA) using the correctness labels graded by the NLI step (Step C). Ambiguous/unknown labels are automatically filtered out.
```bash
python src/uhead/train.py --finetune-cache --dataset musique
```

#### 4. Fine-tune UHead (Stage 2)
Fine-tunes the pretrained UHead on target-domain labels.
```bash
python src/uhead/train.py --finetune --epochs 2


## 5. Output Summary
* **Base Generation**: `outputs/<dataset>_confidence_scores.json`
* **NLI Labels**: `outputs/<dataset>_correctness_labels.json`
* **Pretrained Checkpoint**: `outputs/uhead_pretrained.pt`
* **Fine-tuned Checkpoint**: `outputs/uhead_finetuned.pt`
