# banking77-intent-classifier
# Banking77 Intent Classifier — DistilBERT + LoRA Fine-Tuning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![LoRA](https://img.shields.io/badge/PEFT-LoRA-green)

## Problem Statement

Customer support systems receive thousands of messages daily. 
Manually routing each message to the correct department is 
slow, expensive, and error-prone. This project builds an 
automated intent classifier that categorizes customer banking 
queries into 77 distinct intents — enabling instant, accurate 
routing without human intervention.

**Real-world challenge:** How do you fine-tune a transformer 
model for production use when you have no GPU, no expensive 
API costs, and limited compute resources?

**Solution:** Parameter Efficient Fine-Tuning using LoRA — 
training only 1.17% of model parameters while retaining 
full model capability.

---

## Dataset

**Banking77** — Industry standard customer support benchmark

| Property | Value |
|---|---|
| Training examples | 10,003 |
| Test examples | 3,080 |
| Intent classes | 77 |
| Average text length | 59.5 characters |
| Class imbalance ratio | 5.34x |

Sample intents: `lost_or_stolen_card`, `declined_card_payment`, 
`change_pin`, `top_up_by_card`, `fiat_currency_support`

---

## Architecture & Technical Decisions

### Why DistilBERT over BERT-base?

| Model | Parameters | Performance | Memory |
|---|---|---|---|
| BERT-base | 110M | 100% | High |
| DistilBERT | 66M | 97% | 40% less |

DistilBERT retains 97% of BERT's performance through knowledge 
distillation while using 40% fewer parameters. Critical for 
training on free Colab T4 GPU without memory crashes.

### Why LoRA?

Full fine-tuning of 66M parameters is expensive and risks 
catastrophic forgetting — where the model overwrites pretrained 
knowledge with task-specific patterns.

LoRA freezes all pretrained weights and introduces two small 
adapter matrices alongside each attention layer:
Original matrix W:  768 × 768 = 589,824 parameters
LoRA Matrix A:      768 × 8   =   6,144 parameters
LoRA Matrix B:        8 × 768 =   6,144 parameters
Reduction: 98.8% fewer trainable parameters

**Result:**
Total parameters:     67,012,685
Trainable parameters:    797,261  (1.17%)

### LoRA Configuration

```python
LoraConfig(
    r=8,                              # rank — sweet spot for this task
    lora_alpha=16,                    # scaling factor (2x rank)
    target_modules=["q_lin", "v_lin"],# query and value attention matrices
    lora_dropout=0.1,                 # regularization
    task_type=TaskType.SEQ_CLS        # sequence classification
)
```

### Handling Class Imbalance

Data exploration revealed a 5.34x imbalance between most and 
least frequent classes (187 vs 35 examples). Training without 
correction causes the model to ignore rare intents entirely.

**Fix: Inverse frequency weighted loss**

```python
class_weights = 1.0 / label_counts
# Rare class → high weight → misclassifying it costs more
# Common class → low weight → model cannot ignore rare classes
```

---

## Training Configuration

```python
TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=32,
    learning_rate=2e-5,          # small to prevent catastrophic forgetting
    warmup_steps=100,            # gradual lr increase at start
    weight_decay=0.01,           # regularization
    fp16=True,                   # half precision — 2x memory saving
    eval_strategy="epoch",
    load_best_model_at_end=True
)
```

**Why learning rate 2e-5?**
Large learning rates aggressively overwrite pretrained weights. 
2e-5 gently nudges existing knowledge toward the task without 
destroying what BERT learned during pretraining.

---

## Results

### Training Curve

| Epoch | Training Loss | Validation Loss | Accuracy |
|---|---|---|---|
| 1 | 3.9726 | 3.5859 | 38.76% |
| 2 | 2.5550 | 2.2843 | 61.14% |
| 3 | 1.9706 | 1.7091 | 68.63% |
| 4 | 1.6654 | 1.4714 | 71.03% |
| 5 | 1.5524 | 1.4026 | 71.73% |

**Final Test Accuracy: 72.69%**

Baseline (random): 1.3% — model achieves 56x improvement over random.

### Per-Class Performance

**Top 5 Best Performing Intents:**

| Intent | F1 Score |
|---|---|
| verify_top_up | 1.000 |
| age_limit | 0.976 |
| passcode_forgotten | 0.941 |
| edit_personal_details | 0.940 |
| get_physical_card | 0.925 |

**Top 5 Worst Performing Intents:**

| Intent | F1 Score |
|---|---|
| topping_up_by_card | 0.170 |
| why_verify_identity | 0.333 |
| request_refund | 0.353 |
| supported_cards_and_currencies | 0.353 |
| top_up_by_bank_transfer_charge | 0.370 |

### Failure Mode Analysis

Poor performing intents share a common pattern — semantic 
overlap. A customer saying "I want to add money to my card" 
could legitimately belong to:

- `topping_up_by_card`
- `top_up_by_bank_transfer_charge`  
- `top_up_by_cash`
- `transfer_into_account`

The model distributes confidence across all similar intents 
rather than making one strong prediction. This is a dataset 
limitation — Banking77's 77 classes contain genuinely 
ambiguous boundaries that even human annotators struggle with.

**This explains why confidence scores are moderate:**
"I want to change my PIN" → change_pin (47.46%) — clear intent
"I lost my card"          → card_not_working (14.40%) — ambiguous

---

## Inference Demo

```python
test_queries = [
    "I lost my card and need a replacement",
    "Why was my payment declined?",
    "How do I add money to my account?",
    "I want to change my PIN number",
    "What currencies do you support?"
]
```

**Results:**
Text: Why was my payment declined?
Intent: declined_card_payment
Confidence: 19.94%
Text: I want to change my PIN number
Intent: change_pin
Confidence: 47.46%
Text: What currencies do you support?
Intent: fiat_currency_support
Confidence: 35.15%

---

## What I Would Improve With More Resources

1. **Larger rank r** — r=16 or r=32 would give model more 
   capacity to learn complex intent boundaries

2. **More data for rare classes** — data augmentation or 
   synthetic generation for intents with only 35 examples

3. **Intent merging** — semantically overlapping intents 
   like `topping_up_by_card` and `top_up_by_bank_transfer_charge` 
   could be merged into parent categories

4. **Larger base model** — RoBERTa-base or DeBERTa would 
   likely push accuracy above 80%

5. **Contrastive learning** — train model to explicitly 
   push similar intents apart in embedding space

---

## Stack

| Component | Tool |
|---|---|
| Base Model | distilbert-base-uncased |
| Fine-tuning | HuggingFace PEFT + LoRA |
| Training | PyTorch + HuggingFace Trainer |
| Dataset | HuggingFace Datasets |
| Evaluation | sklearn + evaluate |
| Compute | Google Colab T4 GPU (free) |

---

## Project Structure
banking77-intent-classifier/
├── notebook.ipynb    # Full pipeline: data → train → eval → inference
└── README.md         # This file

---

## Author

**Syed Muhammad Aneeb Ur Rehman**  
AI/ML Engineer | Full-Stack Developer  
[LinkedIn](https://linkedin.com/in/syedaneeb15) | 
[GitHub](https://github.com/aneebnaqvi15)
