# Iris - Llama3.2 FineTome Assistant (ID2223 Lab 2)

## 1. Overview

- Course: ID2223 HT2025
- Student: <你的名字>
- Goal: Fine-tune a small Llama model on the FineTome instruction dataset using LoRA,
  then deploy it on CPU with a Gradio UI (Hugging Face Spaces).

Links:
- Fine-tuned model (LoRA): https://huggingface.co/yunyuan01/llama32-1b-finetome-lora
- Fine-tuned model (merged 16-bit): https://huggingface.co/yunyuan01/llama32-1b-finetome-merged16
- Gradio UI (Space “iris”): <你的 Space URL>


## 2. Task 1 – Fine-tuning pipeline

### 2.1 Dataset

- Name: `mlabonne/FineTome-100k`
- Format: ShareGPT-style `conversations`, converted to Llama 3 chat template
  using Unsloth's `standardize_sharegpt` and `get_chat_template`.

### 2.2 Base model and LoRA setup

- Base model: `unsloth/Llama-3.2-1B-Instruct`
- LoRA rank r=16, target modules: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj
- Training settings (Colab T4 GPU):
  - max_seq_length=2048
  - batch_size=2, gradient_accumulation_steps=4
  - learning_rate=2e-4, warmup_steps=5
  - max_steps=100 (with checkpoints saved every 10 steps)

### 2.3 Checkpointing & resuming

- Checkpoints saved to Google Drive: `/id2223_lab2/outputs_step50/…`
- Demonstrated resuming training from `checkpoint-50` to `checkpoint-100`
  in a new Colab runtime.

### 2.4 Saving & hosting the model

- Saved LoRA adapters + tokenizer to `llama32-1b-finetome-lora`
- Merged LoRA into base model (16-bit) and uploaded to Hugging Face:
  - `yunyuan01/llama32-1b-finetome-merged16`

### 2.5 CPU inference + UI

- Inference on CPU in HF Spaces using `AutoModelForCausalLM.from_pretrained`
- Built a Gradio `ChatInterface` app (Space: `yunyuan01/iris`) that:
  - Builds Llama-3 chat prompts from dialogue history
  - Runs generation with `max_new_tokens=128`
  - Communicates the value as a “FineTome teaching assistant”

### 2.6 Attempted GGUF export (CPU-friendly format)

The lab instructions mention converting the fine-tuned model to a more
CPU-friendly format such as **GGUF**.  
I tried to do this using Unsloth’s built-in helper
`model.push_to_hub_gguf(...)`, starting from my LoRA checkpoint on the
Hugging Face Hub:

from unsloth import FastLanguageModel

1) Load the fine-tuned LoRA model from HF
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "yunquan01/llama32-1b-finetome-lora",
    max_seq_length = 2048,   # I also tried 1024 later
    dtype          = None,
    load_in_4bit   = True,
)

2) Try to convert + upload a quantized GGUF version
model.push_to_hub_gguf(
    "yunquan01/llama32-1b-finetome-gguf",
    tokenizer,
    quantization_method = "q4_k_m",
)

What is supposed to happen:

1. Unsloth merges the 4-bit base model and LoRA adapters into a full
   16-bit model (`model.safetensors`, ~2.5 GB).
2. It then calls the `llama.cpp` conversion tools to quantize this model
   to **GGUF** with the selected method (`q4_k_m`) and uploads the
   resulting `.gguf` file to a new HF repo
   (`yunyuan01/llama32-1b-finetome-gguf`).

However, on **Google Colab free** this step repeatedly failed due to
memory limits:

- With `max_seq_length = 2048`, the kernel crashed when converting the
  merged 16-bit weights to GGUF (RAM usage hit the limit and the session
  was restarted).
- I tried again with a smaller context,
  `max_seq_length = 1024`, but the Colab runtime still crashed at the
  same stage (after printing “Converting model to GGUF format…” and
  installing `llama.cpp`).

Because of these resource limitations, the GGUF conversion never
finished and the repository `yunyuan01/llama32-1b-finetome-gguf` was not
created successfully (404 on Hugging Face).

Even though I could not produce a final `.gguf` file, this experiment
shows that I understand the intended CPU-oriented workflow:

- **Train** a LoRA model on GPU.  
- **Merge** it into a full 16-bit model.  
- **Quantize to GGUF** (e.g. `q4_k_m`) using `llama.cpp` tools so that
  the model can be served efficiently on CPU-only backends such as
  `llama.cpp` or `llama-cpp-python`.


## 3. Task 2 – Improving scalability and performance

### 3.1 Model-centric improvements

Ideas and/or experiments:
- Increase max_steps from 100 → 300/500 to improve alignment.
- Try different LoRA ranks (e.g. r=8, r=32) to trade off quality vs. speed.
- Tune generation hyperparameters (temperature, top_p) specifically for
  my UI use case (教学解释类问答).

(如果你实际做了实验，就在这里写实验设置 + loss/主观对比结果。)

### 3.2 Data-centric improvements

Ideas and/or experiments:
- Filter FineTome to keep only “education / explanation” style samples to
  better match my UI scenario.
- Augment the training data with another small instruction dataset
  (e.g. math / programming instructions) to improve domain coverage.
- Remove extremely long or noisy samples to reduce training noise.

(同样，如果你真的跑了一个“加数据/滤数据”的版本，就写上对比例子。)

### 3.3 Alternative foundation LLMs

To satisfy the requirement of “a couple of different open-source foundation LLMs”, I plan to:
- Fine-tune another ~1B open-source model (e.g. <模型名>) on a small subset of FineTome.
- Compare:
  - CPU inference latency in the Space
  - Subjective quality on a small evaluation set of questions

Then choose the best model for the final UI.

