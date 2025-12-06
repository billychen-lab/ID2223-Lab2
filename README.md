# Iris - Llama3.2 FineTome Assistant (ID2223 Lab 2)

## 1. Overview

- Course: ID2223 HT2025
- Student: Yunquan Chen(yunquan@kth.se) Sibo Zhang(siboz@kth.se)
- Goal: Fine-tune a small Llama model on the FineTome instruction dataset using LoRA,
  then deploy it on CPU with a Gradio UI (Hugging Face Spaces).

Links:
- Fine-tuned model (LoRA): https://huggingface.co/yunyuan01/llama32-1b-finetome-lora
- Fine-tuned model (merged 16-bit): https://huggingface.co/yunyuan01/llama32-1b-finetome-merged16
- Gradio UI (Space “iris”): https://huggingface.co/spaces/yunquan01/iris


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

```python
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
```

What is supposed to happen:

1. Unsloth merges the 4-bit base model and LoRA adapters into a full
   16-bit model (`model.safetensors`, ~2.5 GB).
2. It then calls the `llama.cpp` conversion tools to quantize this model
   to **GGUF** with the selected method (`q4_k_m`) and uploads the
   resulting `.gguf` file to a new HF repo
   (`yunquan01/llama32-1b-finetome-gguf`).

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
finished and the repository `yunquan01/llama32-1b-finetome-gguf` was not
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

To see how much we can improve the answers **without changing the data**, we
did a small decoding experiment on the 1B model
(`yunquan01/llama32-1b-finetome-merged16`). I compared two sets of
generation hyperparameters on three of my Boolean-logic teaching
questions (Q1–Q3):

- **Setting A**: `temperature = 0.2`, `top_p = 0.7`
- **Setting B**: `temperature = 0.7`, `top_p = 0.9`

Qualitatively, I observed the following:

- With **Setting A** the model was very conservative and deterministic.  
  The answers were short and sometimes over-simplified or even misleading.  
  For example, for Q1 it only said that Boolean operators mean “equals” and
  “not equals”; for Q2 it incorrectly focused on a “left-to-right vs
  right-to-left” evaluation order. These replies are easy to grade as
  “wrong” but they also give me almost no useful explanation to show to a
  student.

- With **Setting B** the model produced much longer, more conversational
  answers. It tried to give more detail and sometimes mentioned concrete
  expressions such as `x > 0` or `!x`. For Q3 (“what is NOT?”) the answer
  under Setting B actually described the idea of “the opposite value” and
  explicitly mentioned that the result is `True` or `False`, which is much
  closer to what I want in a teaching scenario. However, the higher
  temperature also made the model more likely to ramble or introduce
  slightly incorrect math-style notation.

Overall, this experiment confirmed that **decoding parameters strongly
affect the teaching style** of the model:

- Low temperature → short, rigid, but often incomplete explanations.  
- Higher temperature + larger `top_p` → more fluent and richer
  explanations, at the cost of occasional hallucinated details.

For my final UI I kept a configuration close to **Setting B**
(`temperature = 0.7`, `top_p = 0.9`), because my goal is to act as a
friendly tutor: it is more important that the model gives a detailed,
easy-to-understand explanation than that every response is perfectly
deterministic.


### 3.2 Data-centric improvements

Ideas and/or experiments:
- Filter FineTome to keep only “education / explanation” style samples to
  better match my UI scenario.
- Augment the training data with another small instruction dataset
  (e.g. math / programming instructions) to improve domain coverage.
- Remove extremely long or noisy samples to reduce training noise.

(同样，如果你真的跑了一个“加数据/滤数据”的版本，就写上对比例子。)


### 3.3 Alternative foundation LLMs: 1B vs 3B

To understand the impact of model size, we manually compared our 1B and 3B
fine-tuned Llama models on five teaching-oriented questions about Boolean
logic (Q1–Q5). For each answer we assigned 1–5 points on several dimensions
(correctness, clarity, quality of examples, etc.), so that the maximum
total score per answer was 20.

The results show a clear pattern.  
For the **more conceptual questions** (Q1–Q3: “What is a Boolean
operator?”, “AND vs OR with real-life examples”, “NOT with two examples”),
the 1B model often produced short or partially incorrect explanations and
very few concrete examples, giving total scores around **8/20**. The 3B
model, in contrast, gave longer explanations with multiple everyday
examples and much clearer wording, scoring **14–16/20** on the same
questions. This suggests that the larger 3B model is significantly better
at verbal reasoning and pedagogical explanation.

For the **more concrete questions** (Q4–Q5: interpreting specific `if`
conditions such as `x > 3 && x < 10` or `age < 18 || isStudent`), both
models performed almost perfectly, with scores of **19–20/20**, and their
outputs were very similar. In these simpler cases, the extra capacity of
the 3B model does not provide much additional benefit over the 1B model.

On average over the five questions, the 3B model reached about **16.8/20**
while the 1B model achieved **12.6/20**. However, the 3B model is also
heavier and slower, which matters for CPU-only inference in a Hugging Face
Space. Overall, this comparison highlights a clear quality–efficiency
trade-off: the 3B model is preferable when I care most about explanation
quality, while the 1B model remains attractive when latency and memory
constraints are more important.


