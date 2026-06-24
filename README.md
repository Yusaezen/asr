# Adaptive Speculative Reasoning

This repository contains the infrastructure, prompt designs, and generation harnesses to explore and evaluate structured Chain-of-Thought (CoT) generation across different reasoning datasets.

## Current Project Phase
We have successfully completed the foundational infrastructure and prompt design phases. The core generation pipelines are now running, and we are moving into the **Manual QA & Data Curation** phase. 

---

## What Has Been Built (Intern 1 & 2 Tasks Completed)

### 1. Infrastructure & Constrained Generation (Baala)
- **Local Model Setup:** A small draft model is integrated via API/HF in `src/model_client.py`.
- **Generation Harness (`src/generation_harness.py`):** The core pipeline that accepts a question, runs the model, and returns a cleanly parsed list of steps.
- **Structured Output Generation (`src/schema.py`):** Uses JSON schemas/grammar-constrained decoding to force the model to separate its reasoning steps reliably.
- **Step Parser (`src/step_parser.py`):** Consumes the raw model output and strictly enforces the parsed step list.
- **Fallback Logic (`src/fallback.py`):** Custom engineering that handles and segments outputs when structured generation alone isn't sufficient.

### 2. Prompt Design & Pilot Generation (Nithilan)
- **Few-Shot Prompting (`src/prompts.py`):** Tailored few-shot exemplars designed specifically for step-formatted CoT generation. Handled per dataset:
  - **GSM8K (`src/load_gsm8k.py`):** Focuses on numeric, math-based step formulation.
  - **HotpotQA (`src/load_hotpotqa.py`) & MuSiQue (`src/load_musique.py`):** Focuses on multi-hop bridging steps.
- **Sentence-Boundary Fallback (`src/fallback.py`):** Additional segmenter logic for edge cases where the structured format fails to isolate steps.
- **Pilot Generations:** Pilot batches have been run across all three datasets and their outputs are saved in the `outputs/` directory.

*(Note: Intern 2 will continue to iterate on prompt wording based on findings from the Manual QA phase.)*

---

## Next Steps: Manual QA & Data Curation Guide

With the pilot generation complete, the immediate next goal is to systematically evaluate the model's output quality. **If you are stepping in to run QA (Intern 3 / Intern 2 iterating on prompts), please follow this workflow:**

### 1. Inspect Pilot Outputs
Navigate to the `outputs/` folder. This contains the generated step lists from our pilot runs across GSM8K, HotpotQA, and MuSiQue.
You will manually inspect the output files to evaluate the model on the following criteria:
- **Atomic Steps:** Is each step a single, logical thought?
- **Separability:** Are steps properly separated, or are multiple steps merged together?
- **Completeness:** Are there any missing logical steps?
- **Consistency:** Does the output strictly follow the requested JSON/formatting constraints?

### 2. Categorize Failure Types
When you find a failure, do not just mark it as failed. Categorize the failure into one of the following buckets so the Prompt/Infrastructure teams can fix it:
- `NO_SEPARATION`: The model dumped all text into a single step.
- `MALFORMED_JSON`: The model failed to adhere to the schema constraint, breaking the parser.
- `MIXED_CONTENT`: The model included bridging thoughts and numeric operations in a confusing/tangled way.
- `OTHER`: Upto your expertise :D

**Feedback Loop:** Log these failures and pass them back to the prompt engineering / infra side. If you are Intern 2, you will use these findings to directly iterate on `src/prompts.py` and `src/fallback.py`.

### 3. Maintain the Pass/Fail Tracking Sheet if possible/required
Keep a simple spreadsheet (e.g., Google Sheets or a local CSV) tracking the pilot batch. For each sample, log:
- `Dataset` (GSM8K, HotpotQA, MuSiQue)
- `Sample ID`
- `Pass/Fail`
- `Failure Category` (if applicable)
- `Notes`

---

## Running the Codebase

If you need to re-run the batch generation or test a new prompt:

1. **Activate the Environment:**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run a Batch:**
   You can run the batch scripts directly to process new questions or re-test failed ones after tweaking the prompts.
   ```bash
   python src/batch_runner.py --dataset gsm8k --samples 50
   ```

3. **Check Outputs:**
   Results will be saved sequentially into the `outputs/` folder for review.

Incase any troubles regarding model setup is encountererd, please let us know!