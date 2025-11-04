# Review Kode Deprecated dan Update yang Diperlukan

## Ringkasan
Kode ini sudah 2 tahun tidak diupdate dan menggunakan standar lama. Berikut adalah review lengkap dari masalah yang ditemukan dan solusinya.

## 🚨 Masalah Utama yang Ditemukan

### 1. **vLLM Version Sangat Outdated** (KRITIS)
**Lokasi**: `modal/runner/engines/vllm.py:29`
- **Masalah**: Menggunakan vLLM v0.2.6 (rilis ~2 tahun lalu)
- **Versi Terbaru**: v0.6.3+ (November 2024)
- **Dampak**:
  - Kehilangan banyak fitur baru (continuous batching, PagedAttention optimization, dll)
  - Performa jauh lebih lambat
  - Tidak support model-model baru (Llama 3, Mistral v0.3, dll)
  - Bug dan security issues yang sudah diperbaiki di versi baru

### 2. **Response Format Tidak Sesuai OpenAI Standard**
**Lokasi**: `modal/shared/protocol.py:51-56`

**Format Sekarang (Custom)**:
```python
{
    "text": "...",
    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    "finish_reason": "stop",
    "done": true
}
```

**Format OpenAI Standard yang Benar**:
```python
{
    "id": "chatcmpl-123",
    "object": "text_completion" / "chat.completion",
    "created": 1677858242,
    "model": "gpt-3.5-turbo",
    "choices": [{
        "text": "...",  # untuk /v1/completions
        "message": {...},  # untuk /v1/chat/completions
        "index": 0,
        "logprobs": null,
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

### 3. **Request Format Tidak Standard**
**Lokasi**: `modal/shared/protocol.py:32-37`

**Sekarang**:
```python
{
    "id": "...",
    "prompt": "...",
    "stream": false,
    "params": {...},
    "model": "..."
}
```

**OpenAI Standard**:
```python
{
    "model": "gpt-3.5-turbo",
    "messages": [...],  # untuk chat completions
    "prompt": "...",    # untuk text completions
    "temperature": 1.0,
    "max_tokens": 100,
    "stream": false
    # params di top-level, bukan nested
}
```

### 4. **Streaming Response Format Salah**
**Lokasi**: `modal/runner/engines/vllm.py:114`

**Sekarang**: Mengirim full response object dalam SSE
```
data: {"text":"hello","usage":{...},"finish_reason":null,"done":false}
```

**OpenAI Standard**: Harus format seperti ini
```
data: {"id":"...","object":"text_completion.chunk","choices":[{"text":"hello","index":0,"finish_reason":null}]}

data: [DONE]
```

### 5. **Missing Standard Fields**
- ❌ Tidak ada field `object` (wajib di OpenAI API)
- ❌ Tidak ada field `created` (timestamp)
- ❌ Tidak ada field `system_fingerprint` (untuk reproducibility)
- ❌ Tidak ada field `total_tokens` di usage
- ❌ Tidak ada `choices` array (OpenAI selalu return array)

### 6. **Deprecated API Patterns**
**Lokasi**: `modal/runner/engines/vllm.py:33-34`
```python
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
```
- vLLM 0.6+ mengubah banyak API internal
- `AsyncEngineArgs` dan `AsyncLLMEngine` sudah berubah signifikan

### 7. **Dependencies Outdated**
**Lokasi**: `pyproject.toml`
- `modal = "^0.62.124"` → Terbaru: 0.64+
- `fastapi = "^0.108.0"` → Terbaru: 0.115+
- `sentry-sdk = "1.39.1"` → Terbaru: 2.17+
- `ray = "^2.9.0"` → Terbaru: 2.38+

### 8. **Error Response Format**
**Lokasi**: `modal/shared/protocol.py:78-82`

**Sekarang**: Plain text error
```python
PlainTextResponse(content=message, status_code=status_code)
```

**OpenAI Standard**: JSON dengan structure khusus
```python
{
    "error": {
        "message": "...",
        "type": "invalid_request_error",
        "param": null,
        "code": null
    }
}
```

## 📋 Daftar Update yang Harus Dilakukan

### Priority 1 - KRITIS (Breaking Changes)

1. **Update vLLM ke versi terbaru (0.6.3+)**
   - File: `modal/runner/engines/vllm.py`
   - Update import statements
   - Update API calls sesuai dokumentasi terbaru

2. **Ubah Response Format ke OpenAI Standard**
   - File: `modal/shared/protocol.py`
   - Tambah `ResponseChoice`, `CompletionResponse`, `ChatCompletionResponse`
   - Update `ResponseBody` → hapus, ganti dengan standard format

3. **Ubah Request Format**
   - File: `modal/shared/protocol.py`
   - Support `/v1/completions` endpoint (text completion)
   - Support `/v1/chat/completions` endpoint (chat format)
   - Flatten params ke top-level

4. **Fix Streaming Format**
   - File: `modal/runner/engines/vllm.py`
   - Gunakan format SSE yang benar
   - Kirim `data: [DONE]` di akhir stream

### Priority 2 - Compatibility

5. **Update Dependencies**
   - File: `pyproject.toml`
   - Update semua dependencies ke versi stabil terbaru

6. **Add Missing Fields**
   - Tambah `object`, `created`, `system_fingerprint`
   - Tambah `total_tokens` di usage
   - Wrap response dalam `choices` array

7. **Error Handling Improvements**
   - Standardize error response format
   - Add proper error codes

### Priority 3 - Nice to Have

8. **Add New Parameters**
   - `response_format` untuk JSON mode
   - `seed` untuk reproducibility
   - `tools` dan `tool_choice` untuk function calling
   - `logit_bias` untuk token probability manipulation

9. **Documentation**
   - Update README dengan setup instructions
   - Tambah API documentation
   - Tambah migration guide dari format lama

## 🔧 Setup Instructions (Untuk Pembaruan)

### Prerequisites
```bash
# Install Poetry jika belum ada
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Setup Modal
poetry run modal token new
```

### Running Locally
```bash
# Development mode
poetry run modal serve modal/runner/api.py

# Deploy to Modal
poetry run modal deploy modal/runner/api.py
```

### Testing
```bash
# Run tests
poetry run pytest modal/tests/

# Test specific endpoint
poetry run pytest modal/tests/runner/endpoints/test_completion.py -v
```

## 📊 Comparison Table

| Aspek | Format Lama | OpenAI Standard |
|-------|------------|-----------------|
| Response wrapper | `{text, usage, finish_reason}` | `{choices: [{text, ...}], usage}` |
| Error format | Plain text | JSON dengan error object |
| Streaming | Custom SSE format | OpenAI SSE dengan [DONE] |
| Fields | Minimal | object, created, id, model, dll |
| Endpoint | `/completion` | `/v1/completions`, `/v1/chat/completions` |
| Parameters | Nested dalam `params` | Top-level (temperature, max_tokens, dll) |

## 🎯 Recommended Action Plan

1. ✅ Buat branch baru untuk updates
2. ✅ Update vLLM ke versi terbaru
3. ✅ Refactor protocol.py dengan OpenAI format
4. ✅ Update engine untuk gunakan format baru
5. ✅ Update tests untuk test format baru
6. ✅ Update dependencies
7. ✅ Test thoroughly
8. ✅ Deploy ke staging
9. ✅ Monitor errors
10. ✅ Deploy ke production

## 📚 References
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/completions)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI Streaming Guide](https://platform.openai.com/docs/api-reference/streaming)
