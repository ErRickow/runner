# OpenRouter Runner - Setup dan Panduan Penggunaan

## 📝 Tentang Project Ini

Project ini adalah runner untuk menjalankan LLM models menggunakan vLLM di infrastructure Modal. Sudah diupdate ke:
- **vLLM 0.6.3+** (dari 0.2.6) - Performa jauh lebih baik!
- **OpenAI-compatible API** - Format standard industri
- **Dependencies terbaru** (2024/2025)

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10 atau 3.11**
```bash
python --version  # Pastikan >= 3.10
```

2. **Poetry** (package manager)
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Atau via pip
pip install poetry
```

3. **Modal Account**
- Daftar di https://modal.com
- Gratis untuk testing, ada free tier

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd runner

# 2. Install dependencies
poetry install

# 3. Setup Modal
poetry run modal token new

# Ikuti instruksi untuk login ke Modal
# Ini akan membuat token di ~/.modal.toml
```

### Configuration

Buat file `.env` untuk development:

```bash
# .env
DD_ENV=development
RUNNER_API_KEY=your-secret-api-key-here

# Optional: Datadog monitoring
DD_API_KEY=your-datadog-api-key
DD_SITE=datadoghq.com
```

## 🏃‍♂️ Running the Service

### Development Mode (Local Testing)

```bash
# Serve locally (Modal akan handle GPU di cloud)
poetry run modal serve modal/runner/api.py

# Service akan running di: https://your-workspace--runner-api.modal.run
```

### Production Deployment

```bash
# Deploy ke Modal
poetry run modal deploy modal/runner/api.py

# Lihat logs
poetry run modal app logs runner

# Check status
poetry run modal app list
```

## 📡 API Usage

### Endpoint Options

Ada 3 endpoint yang tersedia:

1. **`POST /v1/completions`** - OpenAI-compatible (RECOMMENDED)
2. **`POST /completion`** - Legacy format (backward compatibility)
3. **`POST /`** - Legacy format (backward compatibility)

### OpenAI-Compatible Format (Recommended)

#### Non-Streaming Request

```bash
curl -X POST "https://your-workspace--runner-api.modal.run/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "microsoft/phi-2",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
  }'
```

**Response Format:**
```json
{
  "id": "cmpl-abc123...",
  "object": "text_completion",
  "created": 1699564800,
  "model": "microsoft/phi-2",
  "choices": [
    {
      "text": " there was a kingdom...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 90,
    "total_tokens": 100
  }
}
```

#### Streaming Request

```bash
curl -X POST "https://your-workspace--runner-api.modal.run/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "microsoft/phi-2",
    "prompt": "Write a poem about AI",
    "max_tokens": 200,
    "temperature": 0.8,
    "stream": true
  }'
```

**Streaming Response (SSE format):**
```
data: {"id":"cmpl-123","object":"text_completion.chunk","created":1699564800,"model":"microsoft/phi-2","choices":[{"text":"In","index":0,"finish_reason":null}]}

data: {"id":"cmpl-123","object":"text_completion.chunk","created":1699564800,"model":"microsoft/phi-2","choices":[{"text":" circuits","index":0,"finish_reason":null}]}

data: {"id":"cmpl-123","object":"text_completion.chunk","created":1699564800,"model":"microsoft/phi-2","choices":[{"text":" deep","index":0,"finish_reason":null}]}

...

data: {"id":"cmpl-123","object":"text_completion.chunk","created":1699564800,"model":"microsoft/phi-2","choices":[{"text":"","index":0,"finish_reason":"stop"}]}

data: [DONE]
```

### Python Client Example

```python
import requests
import json

API_URL = "https://your-workspace--runner-api.modal.run/v1/completions"
API_KEY = "your-api-key"

# Non-streaming
response = requests.post(
    API_URL,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    },
    json={
        "model": "microsoft/phi-2",
        "prompt": "Explain quantum computing in simple terms:",
        "max_tokens": 150,
        "temperature": 0.7,
    }
)

result = response.json()
print(result["choices"][0]["text"])

# Streaming
response = requests.post(
    API_URL,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    },
    json={
        "model": "microsoft/phi-2",
        "prompt": "Tell me a joke",
        "max_tokens": 100,
        "stream": True,
    },
    stream=True,
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]  # Remove 'data: ' prefix
            if data == '[DONE]':
                break
            chunk = json.loads(data)
            text = chunk["choices"][0]["text"]
            print(text, end='', flush=True)
```

### OpenAI Python Library

Karena API sudah compatible dengan OpenAI, bisa pakai library mereka:

```python
from openai import OpenAI

# Initialize dengan custom base URL
client = OpenAI(
    base_url="https://your-workspace--runner-api.modal.run/v1",
    api_key="your-api-key",
)

# Non-streaming
completion = client.completions.create(
    model="microsoft/phi-2",
    prompt="Translate to French: Hello, how are you?",
    max_tokens=50,
)
print(completion.choices[0].text)

# Streaming
stream = client.completions.create(
    model="microsoft/phi-2",
    prompt="Write a haiku about coding",
    max_tokens=100,
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].text, end='', flush=True)
```

## 🎯 Available Models

Model yang sudah terdaftar di system:

| Model Name | Quantization | GPU | Max Context |
|------------|--------------|-----|-------------|
| `microsoft/phi-2` | GPTQ | A10G | 2048 |
| `Intel/neural-chat-7b-v3-1` | GPTQ | A10G | 4096 |
| `KoboldAI/LLaMA2-13B-Psyfighter2` | GPTQ | A10G | 4096 |

### Adding New Models

Edit `modal/runner/containers/vllm_unified.py`:

```python
# Tambah model baru
_my_model = "organization/model-name"
VllmContainer_MyModel = _make_container(
    name="VllmContainer_MyModel",
    model_name=_my_model,
    gpu=modal.gpu.A100(count=1),  # Pilih GPU yang sesuai
    concurrent_inputs=4,
    max_containers=5,
    # quantization="GPTQ",  # Jika pakai quantized model
)

# Jika pakai quantized model, tambahkan mapping
QUANTIZED_MODELS = {
    "organization/original-name": _my_model,
}
```

## 🔧 API Parameters

### Completion Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model ID |
| `prompt` | string/array | required | Input text |
| `max_tokens` | integer | 16 | Maximum tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature (0-2) |
| `top_p` | float | 1.0 | Nucleus sampling |
| `top_k` | integer | -1 | Top-k sampling (-1 = disabled) |
| `n` | integer | 1 | Number of completions |
| `stream` | boolean | false | Enable streaming |
| `stop` | string/array | null | Stop sequences |
| `presence_penalty` | float | 0.0 | Presence penalty (-2 to 2) |
| `frequency_penalty` | float | 0.0 | Frequency penalty (-2 to 2) |
| `repetition_penalty` | float | 1.0 | Repetition penalty |
| `best_of` | integer | null | Generate best_of completions |
| `logprobs` | integer | null | Return log probabilities |
| `seed` | integer | null | Random seed |

### vLLM-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_p` | float | 0.0 | Minimum probability threshold |
| `ignore_eos` | boolean | false | Ignore end-of-sequence token |
| `use_beam_search` | boolean | false | Use beam search |
| `skip_special_tokens` | boolean | true | Skip special tokens in output |

## 🧪 Testing

```bash
# Run all tests
poetry run pytest modal/tests/ -v

# Run specific test
poetry run pytest modal/tests/runner/endpoints/test_completion.py -v

# Run with coverage
poetry run pytest modal/tests/ --cov=modal/runner --cov-report=html
```

### Integration Tests

Pastikan service sudah running dan `.env.dev` sudah dikonfigurasi:

```bash
# .env.dev
API_URL=https://your-workspace--runner-api.modal.run/completion
RUNNER_API_KEY=your-api-key
```

```bash
# Run integration tests
poetry run pytest modal/tests/ -v -m integration
```

## 📊 Monitoring

### Logs

```bash
# View logs in real-time
poetry run modal app logs runner --follow

# View logs for specific function
poetry run modal app logs runner.VllmContainer_MicrosoftPhi2
```

### Metrics

Jika menggunakan Datadog:

```python
# Metrics automatically sent:
- Generation latency
- Tokens per second (TPS)
- Request counts
- Error rates
- GPU utilization
```

## 🔍 Troubleshooting

### Common Issues

**1. Modal Authentication Failed**
```bash
# Re-authenticate
poetry run modal token new
```

**2. GPU Out of Memory**
```python
# Reduce concurrent_inputs or max_num_seqs
_make_container(
    ...,
    concurrent_inputs=2,  # Reduce this
    # Add to vllm_opts:
    gpu_memory_utilization=0.85,  # Reduce from 0.90
)
```

**3. Model Not Found**
```bash
# Download model manually
poetry run modal run modal/runner/shared/download.py::download_model --model-name "microsoft/phi-2"
```

**4. Import Errors**
```bash
# Reinstall dependencies
poetry install --sync
```

**5. vLLM Version Conflicts**
```bash
# Check vLLM version in container
poetry run modal shell runner.VllmContainer_MicrosoftPhi2
# Inside container:
pip show vllm
```

## 🔐 Security

### API Key Management

```bash
# Generate secure API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set in Modal secrets
poetry run modal secret create runner-api-key \
  RUNNER_API_KEY=your-generated-key
```

### Best Practices

1. **Never commit** `.env` files
2. **Use Modal secrets** for production
3. **Rotate API keys** regularly
4. **Enable rate limiting** di production
5. **Monitor logs** untuk suspicious activity

## 📈 Performance Optimization

### GPU Selection

```python
# Untuk model kecil (<7B params)
gpu=modal.gpu.A10G(count=1)

# Untuk model medium (7B-13B)
gpu=modal.gpu.A100(count=1, memory=40)

# Untuk model besar (>13B)
gpu=modal.gpu.A100(count=2, memory=80)  # Tensor parallel
```

### Batch Processing

```python
_make_container(
    ...,
    concurrent_inputs=8,  # Handle 8 requests simultaneously
    max_num_seqs=256,     # Max sequences in batch
)
```

### Cold Start Optimization

```python
_make_container(
    ...,
    keep_warm=1,  # Keep 1 container warm (costs money!)
    container_idle_timeout=20 * 60,  # 20 minutes
)
```

## 🆕 Migration dari Format Lama

Jika sudah pakai format lama, ini cara migrate:

### Format Lama (Deprecated)
```json
{
  "id": "req-123",
  "prompt": "Hello",
  "model": "microsoft/phi-2",
  "stream": false,
  "params": {
    "max_tokens": 100,
    "temperature": 0.7
  }
}
```

### Format Baru (OpenAI-Compatible)
```json
{
  "model": "microsoft/phi-2",
  "prompt": "Hello",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

**Good news:** Format lama masih didukung untuk backward compatibility!

## 📚 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Modal Documentation](https://modal.com/docs)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/completions)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

```bash
# Format code
poetry run ruff format .

# Lint
poetry run ruff check .

# Type check
poetry run mypy modal/

# Run pre-commit hooks
poetry run pre-commit run --all-files
```

## 📄 License

Lihat file LICENSE untuk detail.

## 💬 Support

Jika ada masalah atau pertanyaan:

1. Check dokumentasi di atas
2. Lihat issues di GitHub
3. Contact: support@openrouter.ai

---

**Last Updated**: November 2024
**Version**: 2.0.0
