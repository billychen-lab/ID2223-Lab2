import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# 你的模型仓库名
MODEL_ID = "yunquan01/llama32-3b-finetome-merged16"

device = torch.device("cpu")

# 加载 tokenizer 和模型（CPU）
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    low_cpu_mem_usage=True,
)
model.to(device)
model.eval()

SYSTEM_PROMPT = (
    "You are a helpful assistant fine-tuned on the FineTome instruction dataset. "
    "Answer clearly and concisely."
)

MAX_TURNS = 4  # 只保留最近 4 轮对话，防止上下文太长


def respond(message, history):
    """
    ChatInterface 默认：
    - message: 当前用户输入 (str)
    - history: 形如 [[user1, bot1], [user2, bot2], ...] 的列表
    我们把它转成 Llama-3 的 messages 格式。
    """
    # 裁剪历史
    if len(history) > MAX_TURNS:
        history = history[-MAX_TURNS:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for user_text, bot_text in history:
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if bot_text:
            messages.append({"role": "assistant", "content": bot_text})

    # 当前这轮用户输入
    messages.append({"role": "user", "content": message})

    # 应用 Llama-3 chat 模板
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,            # 生成短一点，减轻 CPU 压力
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="Iris - Llama3.2 FineTome Assistant",
    description="Chat with my fine-tuned Llama-3.2-3B model trained on the FineTome instruction dataset.",
)

if __name__ == "__main__":
    demo.launch()


