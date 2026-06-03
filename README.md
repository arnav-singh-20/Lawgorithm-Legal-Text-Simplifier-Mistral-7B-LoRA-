# Lawgorithm — Legal Text Simplifier (Mistral-7B + LoRA)

Fine-tuned **Mistral-7B-Instruct-v0.2** using **QLoRA** to simplify complex legal clauses into plain, easy-to-understand language — while preserving legal meaning.

## Example

**Input:**
The indemnified party shall be held harmless from all liabilities arising from the indemnifying party's negligent acts or omissions.

**Output:**
The party being protected won't be responsible for any damages caused by the other party's carelessness.

## Tech Stack

- `mistralai/Mistral-7B-Instruct-v0.2` — base model
- `PEFT` + `LoRA` (r=16, alpha=32) — parameter-efficient fine-tuning
- `BitsAndBytes` — 4-bit quantization (QLoRA)
- `Hugging Face Datasets` — dataset loading and formatting

## Training Config

| Config | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Quantization | 4-bit NF4 double quant |
| Dataset size | 500 samples |
| Base model | Mistral-7B-Instruct-v0.2 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "path/to/saved/adapter")

# Run inference
def ask_simplify(text):
    prompt = f"""You are a legal assistant. Rewrite the following clause in simple, easy-to-understand language while preserving its legal meaning.

Clause:
{text}

Simplified Version:"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=150, temperature=0.0, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

ask_simplify("Your legal clause here")
```

## Why This Project

Legal documents are inaccessible to most people. This model bridges that gap by translating dense legal language into plain English — useful for contract review, legal aid tools, and platforms teaching law.

## Author

**Arnav Singh**  
[LinkedIn](https://www.linkedin.com/in/-singharnav/) • [Portfolio](https://arnav-singh-14q4.onrender.com/)
