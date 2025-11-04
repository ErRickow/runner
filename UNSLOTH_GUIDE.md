# 🚀 Unsloth Guide - Inference Murah dengan T4 GPU!

## 💰 Kenapa Unsloth + T4?

Unsloth adalah library optimized untuk inference LLM yang **2x lebih cepat** dan pakai **60% lebih sedikit memory** dibanding transformers biasa.

### Perbandingan Biaya (Modal Pricing)

| GPU Type | vRAM | Cost/Hour | Best For | Engine |
|----------|------|-----------|----------|--------|
| **T4** | 16GB | **~$0.60/hour** | Models <7B | **Unsloth** ← Paling Murah! |
| A10G | 24GB | ~$1.10/hour | Models 7-13B | vLLM |
| A100 | 40GB | ~$3.00/hour | Models >13B | vLLM |

**Hemat 5x lipat!** T4 + Unsloth hanya $0.60/jam vs A100 $3/jam!

## 🎯 Model yang Tersedia

Semua model ini jalan sempurna di T4 (16GB) dengan 4-bit quantization:

### 1. **Phi-2** (2.7B) - Ultra Cepat! ⚡
```bash
Model: "phi-2" atau "microsoft/phi-2"
Size: 2.7B parameters
Speed: Sangat cepat!
Quality: Bagus untuk tasks sederhana
Best for: Chatbots, Q&A, code completion
```

### 2. **Llama-3.2-3B** (3B) - Latest Llama! 🔥
```bash
Model: "llama-3.2-3b" atau "unsloth/Llama-3.2-3B-Instruct"
Size: 3B parameters
Speed: Cepat
Quality: Excellent, latest model
Best for: General purpose, chat, reasoning
```

### 3. **Mistral-7B** (7B) - Popular Choice! ⭐
```bash
Model: "mistral-7b" atau "unsloth/mistral-7b-instruct-v0.3"
Size: 7B parameters
Speed: Moderat (tapi murah!)
Quality: Sangat bagus
Best for: Complex tasks, long context
```

### 4. **Gemma-2B** (2B) - Google Model! 🌟
```bash
Model: "gemma-2b" atau "unsloth/gemma-2b-it"
Size: 2B parameters
Speed: Sangat cepat!
Quality: Bagus
Best for: Fast responses, simple tasks
```

### 5. **Qwen-2.5-3B** (3B) - Coding & Multilingual! 🌏
```bash
Model: "qwen-3b" atau "unsloth/Qwen2.5-3B-Instruct"
Size: 3B parameters
Speed: Cepat
Quality: Excellent untuk code & multilingual
Best for: Coding, Chinese, multilingual tasks
```

## 📡 Cara Pakai

### Setup (Sama seperti biasa)

```bash
# 1. Setup Modal
poetry run modal token set --token-id YOUR_ID --token-secret YOUR_SECRET

# 2. Setup secrets
poetry run modal secret create huggingface HUGGINGFACE_TOKEN=your_token
poetry run modal secret create ext-api-key RUNNER_API_KEY=your_api_key

# 3. Download model Unsloth (opsional, auto-download saat pertama pakai)
poetry run modal run runner::download

# 4. Deploy
poetry run modal deploy runner
```

### API Request - Format OpenAI

```python
import requests

API_URL = "https://your-workspace--runner-completion.modal.run/v1/completions"
API_KEY = "your-api-key"

# Pakai Unsloth model (T4 GPU - Murah!)
response = requests.post(
    API_URL,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    },
    json={
        "model": "phi-2",  # atau "llama-3.2-3b", "mistral-7b", dll
        "prompt": "Jelaskan apa itu Unsloth dalam 3 kalimat",
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": False
    }
)

print(response.json())
```

### Streaming Response

```python
import requests
import json

response = requests.post(
    API_URL,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    },
    json={
        "model": "llama-3.2-3b",  # Latest Llama!
        "prompt": "Write a short story about AI",
        "max_tokens": 300,
        "temperature": 0.8,
        "stream": True  # Enable streaming
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data == '[DONE]':
                print("\n✅ Done!")
                break
            try:
                chunk = json.loads(data)
                text = chunk["choices"][0]["text"]
                print(text, end='', flush=True)
            except:
                pass
```

### Dengan OpenAI Python Library

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-workspace--runner-completion.modal.run/v1",
    api_key="your-api-key",
)

# Non-streaming
completion = client.completions.create(
    model="mistral-7b",  # Unsloth model
    prompt="Translate to Indonesian: Hello, how are you?",
    max_tokens=100,
)
print(completion.choices[0].text)

# Streaming
stream = client.completions.create(
    model="qwen-3b",  # Great for coding!
    prompt="Write a Python function to calculate fibonacci",
    max_tokens=200,
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].text, end='', flush=True)
```

## 🔧 Model Aliases

Bisa pakai nama yang lebih simple:

| Alias | Full Model Name |
|-------|-----------------|
| `phi-2` | `microsoft/phi-2` |
| `llama-3.2-3b` | `unsloth/Llama-3.2-3B-Instruct` |
| `mistral-7b` | `unsloth/mistral-7b-instruct-v0.3` |
| `gemma-2b` | `unsloth/gemma-2b-it` |
| `qwen-3b` | `unsloth/Qwen2.5-3B-Instruct` |

## 💡 Tips Menghemat Biaya

### 1. Pakai Model Terkecil yang Cukup
```python
# ✅ Good: Pakai Phi-2 untuk simple tasks
model="phi-2"  # $0.60/hour

# ❌ Overkill: Pakai Mistral-7B untuk simple tasks
model="mistral-7b"  # Masih $0.60/hour tapi lebih lambat
```

### 2. Set `container_idle_timeout` Rendah
Di `unsloth_unified.py`, model-model sudah di-set dengan timeout rendah:
- Phi-2: 5 menit
- Gemma-2B: 3 menit
- Yang lain: 10 menit

Ini otomatis shutdown container kalau idle → hemat biaya!

### 3. Disable `keep_warm` di Development
```python
keep_warm=None  # Don't keep containers warm during development
```

### 4. Use Concurrent Requests
T4 bisa handle multiple requests:
```python
# Unsloth settings
concurrent_inputs=2  # untuk model 7B
concurrent_inputs=3  # untuk model 3B
concurrent_inputs=4  # untuk model 2B
```

## 📊 Performance Comparison

Berdasarkan benchmarks Unsloth:

| Task | Standard | Unsloth | Speedup |
|------|----------|---------|---------|
| Inference | 1.0x | **2.0x** | 2x faster |
| Memory | 100% | **40%** | 60% less |
| Cost (T4) | N/A | **$0.60/hr** | 5x cheaper than A100 |

## 🎯 Use Cases per Model

### Phi-2 (2.7B) - Best for:
- ✅ Simple chatbots
- ✅ Q&A systems
- ✅ Text completion
- ✅ Code snippets
- ❌ Complex reasoning
- ❌ Long context (max 2048 tokens)

### Llama-3.2-3B (3B) - Best for:
- ✅ General chat
- ✅ Reasoning tasks
- ✅ Content generation
- ✅ Summarization
- ✅ Latest features
- ✅ 4096 token context

### Mistral-7B (7B) - Best for:
- ✅ Complex instructions
- ✅ Long-form content
- ✅ Multi-step reasoning
- ✅ 4096 token context
- ⚠️ Slightly slower (but still cheap!)

### Gemma-2B (2B) - Best for:
- ✅ Ultra-fast responses
- ✅ High throughput
- ✅ Simple tasks
- ✅ Budget-conscious apps
- ❌ Not for complex tasks

### Qwen-2.5-3B (3B) - Best for:
- ✅ **Coding tasks** (excellent!)
- ✅ Multilingual (Chinese, English, etc)
- ✅ Math problems
- ✅ Technical writing
- ✅ 4096 token context

## 🚀 Example: Building a Budget Chatbot

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-workspace--runner-completion.modal.run/v1",
    api_key="your-api-key",
)

def chat(user_message, model="llama-3.2-3b"):
    """
    Simple chat function using Unsloth + T4
    Cost: ~$0.60/hour (vs $3/hour for A100!)
    """
    prompt = f"User: {user_message}\n\nAssistant:"

    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        stop=["User:", "\n\n"]
    )

    return response.choices[0].text.strip()


# Usage
print(chat("What is Unsloth?"))
print(chat("How can I save money on AI inference?"))
print(chat("Write a haiku about clouds"))
```

## 📈 Monitoring Costs

```python
# Estimasi biaya per request
tokens_generated = 200
time_per_request = 2  # seconds
hourly_cost = 0.60  # USD for T4

cost_per_request = (time_per_request / 3600) * hourly_cost
print(f"Cost per request: ${cost_per_request:.4f}")  # ~$0.0003

# Untuk 1000 requests
print(f"Cost for 1000 requests: ${cost_per_request * 1000:.2f}")  # ~$0.33
```

## 🆚 vLLM vs Unsloth

| Feature | vLLM (A10G/A100) | Unsloth (T4) |
|---------|------------------|--------------|
| Speed | Fastest | 2x faster than transformers |
| Memory | Optimized | 60% less than transformers |
| Cost | $1.10-$3/hour | **$0.60/hour** ← Winner! |
| Models | 7B+ | <7B |
| Batch Size | Large | Moderate |
| Best For | Production, high throughput | Development, budget apps |

## 💰 Cost Calculator

```python
# Monthly cost estimation
hours_per_day = 8
days_per_month = 30
hourly_rate = 0.60  # T4 + Unsloth

monthly_cost = hours_per_day * days_per_month * hourly_rate
print(f"Monthly cost (8hrs/day): ${monthly_cost}")  # $144/month

# Compare with A100
a100_rate = 3.00
a100_monthly = hours_per_day * days_per_month * a100_rate
print(f"A100 monthly cost: ${a100_monthly}")  # $720/month

print(f"Savings: ${a100_monthly - monthly_cost}/month")  # $576/month saved!
```

## 🎓 Tips & Tricks

1. **Start Small**: Mulai dengan Phi-2 atau Gemma-2B untuk testing
2. **Scale Up**: Kalau butuh quality lebih, naik ke Llama-3.2-3B atau Mistral-7B
3. **Combine Models**: Pakai model kecil untuk simple tasks, model besar untuk complex tasks
4. **Monitor Usage**: Track berapa request per model untuk optimize costs
5. **Use Streaming**: Better UX dan bisa stop early kalau tidak perlu full response

## 🔗 Resources

- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **Modal Pricing**: https://modal.com/pricing
- **Supported Models**: https://github.com/unslothai/unsloth#-finetune-for-free

---

**Happy Inferencing! 🚀**

Dengan Unsloth + T4, sekarang kamu bisa run AI models dengan biaya 5x lebih murah! 💰
