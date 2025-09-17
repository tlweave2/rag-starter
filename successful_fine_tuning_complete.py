# COMPLETE SUCCESSFUL MOTORCYCLE FINE-TUNING PROJECT
# This is the exact code that worked for your successful training

# ============================================
# PROJECT OVERVIEW AND OUTCOMES
# ============================================
"""
WHAT WE BUILT:
- Fine-tuned Qwen 2.5-7B model on motorcycle manual Q&A data
- Trained on 850 examples + 75 validation examples  
- Model learned structured technical response format
- Successfully deployed to HuggingFace Hub: tlweave2/motorcycle-manual-assistant

TRAINING RESULTS:
- Training completed in ~5-6 minutes on A100 80GB
- Final training loss: 1.64 (good convergence)
- Model learned: Problem → Diagnosis → Solution → Parts format
- Consistent page citations: [Page X, Service Manual]
- Professional technical language and manual section references

FILES CREATED:
1. Training data: dyna_llm_training_1000_strict_train.jsonl (850 examples)
2. Validation data: dyna_llm_training_1000_strict_val.jsonl (75 examples)  
3. Trained model: motorcycle-lora-final/ (local Colab)
4. HF Model: tlweave2/motorcycle-manual-assistant (cloud hosted)

NEXT STEPS FOR COMPLETE SYSTEM:
- Integrate with RAG pipeline for document retrieval
- Deploy complete system to HuggingFace Spaces
- Add vector database of motorcycle manual chunks
- Create web interface for end users
"""

# ============================================
# CELL 1: Package Installation
# ============================================
!pip install transformers accelerate peft datasets

# ============================================
# CELL 2: Load Model and Data (A100 Full Precision)
# ============================================
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# CONFIRMED WORKING: A100 80GB with full precision (no quantization)
model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # Half precision for memory efficiency
)

print(f"Model loaded - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
# EXPECTED OUTPUT: ~14.19 GB GPU memory usage

# Load motorcycle training data (THESE FILES MUST EXIST IN /content/)
train_data = []
with open('/content/dyna_llm_training_1000_strict_train.jsonl', 'r') as f:
    for line in f:
        train_data.append(json.loads(line.strip()))

val_data = []
with open('/content/dyna_llm_training_1000_strict_val.jsonl', 'r') as f:
    for line in f:
        val_data.append(json.loads(line.strip()))

print(f"Training: {len(train_data)} examples")
print(f"Validation: {len(val_data)} examples")
# EXPECTED OUTPUT: Training: 850 examples, Validation: 75 examples

# ============================================
# CELL 3: LoRA Setup and Data Processing
# ============================================
# CONFIRMED WORKING: LoRA configuration for Qwen 2.5-7B
lora_config = LoraConfig(
    r=16,                    # Rank: 16 gives good quality/speed balance
    lora_alpha=32,          # Alpha: 2x rank is optimal
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,      # Light dropout for regularization
    bias="none",            # No bias training for efficiency
    task_type="CAUSAL_LM"   # Causal language modeling
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# EXPECTED OUTPUT: trainable params: 40,370,176 || all params: 7,655,986,688 || trainable%: 0.5273

# Format data into instruction-following format
def format_prompt(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"}

train_dataset = Dataset.from_list(train_data).map(format_prompt)
eval_dataset = Dataset.from_list(val_data).map(format_prompt)

# CRITICAL: Tokenization with fixed padding that prevents tensor dimension errors
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # CRITICAL: Fixed padding prevents tensor errors
        max_length=512,        # WORKING LENGTH: 512 tokens fits well in memory
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # For causal LM
    return tokenized

# Apply tokenization
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

print("Data prepared")

# ============================================
# CELL 4: Training Configuration and Start
# ============================================
# WORKING HYPERPARAMETERS: These settings produced successful training
training_args = TrainingArguments(
    output_dir="./motorcycle-lora",
    num_train_epochs=3,                    # 3 epochs sufficient for 850 examples
    per_device_train_batch_size=4,        # Batch size that fits in A100 memory
    per_device_eval_batch_size=4,         # Matching eval batch size
    gradient_accumulation_steps=4,        # Effective batch size: 4*4=16
    warmup_steps=100,                     # Warmup for stable training
    learning_rate=2e-4,                   # Good learning rate for LoRA
    fp16=True,                           # Half precision for speed
    logging_steps=50,                    # Log every 50 steps
    eval_strategy="steps",               # FIXED: Was evaluation_strategy
    eval_steps=200,                      # Evaluate every 200 steps
    save_strategy="steps",               # Save checkpoints
    save_steps=200,
    save_total_limit=2,                  # Keep only 2 checkpoints
    load_best_model_at_end=True,        # Load best performing model
    report_to=[]                         # Disable wandb tracking
)

# Create trainer with working configuration
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=default_data_collator  # Simple collator that worked
)

print("Starting training...")
trainer.train()
# EXPECTED: Training completes in 5-6 minutes with decreasing loss

# ============================================
# CELL 5: Save and Test Model
# ============================================
# Save trained model locally
model.save_pretrained("./motorcycle-lora-final")
tokenizer.save_pretrained("./motorcycle-lora-final")

# Test function for the fine-tuned model
def test_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,           # Generate up to 256 new tokens
            temperature=0.3,              # Low temperature for focused responses
            do_sample=True,               # Use sampling for natural responses
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# Test with motorcycle question
test_prompt = """### Instruction:
Starter grinding noise - what should I check?

### Input:
Context: Starter troubleshooting

### Response:
"""

print("Testing model:")
print(test_model(test_prompt))

# EXPECTED OUTPUT FORMAT:
"""
Problem → Starter grinding noise
Diagnosis → Refer to the manual content on this page: Starter troubleshooting
Solution → Follow the factory procedure exactly. See manual section Electrical → Starting System for complete procedure.
Parts if needed → See manual section Electrical → Starting System and the model-year Parts Catalog for required parts/special tools.
[Page 32, Service Manual]
"""

print("Training complete!")

# ============================================
# CELL 6: Upload to HuggingFace Hub
# ============================================
from huggingface_hub import notebook_login

# Login to HuggingFace
notebook_login()

# Upload model to HuggingFace Hub (COMPLETED SUCCESSFULLY)
model.push_to_hub("tlweave2/motorcycle-manual-assistant")
tokenizer.push_to_hub("tlweave2/motorcycle-manual-assistant")

print("Model uploaded to HuggingFace Hub!")
print("Model URL: https://huggingface.co/tlweave2/motorcycle-manual-assistant")

# ============================================
# HOW TO LOAD YOUR MODEL LATER
# ============================================
"""
# Loading your fine-tuned model from HuggingFace:
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("tlweave2/motorcycle-manual-assistant")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "tlweave2/motorcycle-manual-assistant")

# Then use test_model() function above to generate responses
"""

# ============================================
# NEXT STEPS FOR RAG INTEGRATION
# ============================================
"""
COMPLETE RAG SYSTEM ARCHITECTURE:
1. Vector Database: Store motorcycle manual chunks with embeddings
2. Retrieval: Use semantic search to find relevant manual sections
3. Fine-tuned Model: Format retrieved content with proper citations
4. Web Interface: Deploy on HuggingFace Spaces with Gradio

DEPLOYMENT FILES NEEDED:
- app.py (Gradio interface)
- requirements.txt (dependencies)
- manual_chunks.json (processed manual sections)
- manual_embeddings.faiss (vector index)

Your fine-tuned model at tlweave2/motorcycle-manual-assistant is ready
to be the formatting/citation layer in a complete RAG system.
"""