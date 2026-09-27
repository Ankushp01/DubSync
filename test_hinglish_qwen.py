import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "MihaiPopa-1/Qwen3-0.6B-English-Hinglish"

TEXT = """Dangerous ho sakti hai kyonki ismen jo markery hoti hai na vo tens of thousand times zyaada hoti hai jitna WHO recommend karta hai aur ye creams bhale hi aapko ghora bana dengi but ye aapki skin ko completely destroy kar sakti hai uska barrier destroy kar sakti hai but usse badtar aapki kidney tak ko fail kar sakti hai kyonki ye markery sidha absorb ho jaati hai bodyke ports ke through aur aapki kidney mein jaakar settle ho jaati hai. Yahi."""


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.eval()

messages = [
    {
        "role": "user",
        "content": f"Translate Hinglish to English: {TEXT}",
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

print("Translating...")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
    )

generated = outputs[0][inputs["input_ids"].shape[-1]:]

result = tokenizer.decode(
    generated,
    skip_special_tokens=True,
).strip()

print("\n" + "=" * 80)
print("APEX HINGLISH")
print("=" * 80)
print(TEXT)

print("\n" + "=" * 80)
print("QWEN HINGLISH → ENGLISH")
print("=" * 80)
print(result)

print("=" * 80)