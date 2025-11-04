# OpenRouter Runner (Updated 2024)

> **⚠️ IMPORTANT UPDATE:** This repository has been modernized and updated to current standards (November 2024)

## 🎯 What's New

This repo was originally from the early days of OpenRouter. It has now been **completely updated** with:

- ✅ **vLLM 0.6.3+** (from 0.2.6) - Latest optimizations and performance
- ✅ **OpenAI-compatible API** - Standard format yang dipakai industri
- ✅ **Modern dependencies** - All packages updated to 2024/2025 versions
- ✅ **Backward compatibility** - Legacy format masih didukung
- ✅ **Better error handling** - OpenAI-standard error responses
- ✅ **Improved streaming** - Proper SSE format with [DONE] marker

## 🚀 Quick Start

```bash
# Install dependencies
poetry install

# Setup Modal authentication
poetry run modal token new

# Run in development
poetry run modal serve modal/runner/api.py
```

## 📖 Documentation

- **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** - Panduan lengkap setup, API usage, dan troubleshooting (BAHASA INDONESIA)
- **[DEPRECATION_REVIEW.md](./DEPRECATION_REVIEW.md)** - Detail tentang deprecated code dan changes yang dilakukan

## 📡 API Endpoints

### OpenAI-Compatible (Recommended)
```bash
POST /v1/completions
```

### Legacy (Still Supported)
```bash
POST /completion
POST /
```

## 🔍 Example Usage

```python
import requests

response = requests.post(
    "https://your-workspace--runner-api.modal.run/v1/completions",
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "model": "microsoft/phi-2",
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "temperature": 0.7
    }
)

print(response.json())
```

## 📊 What Changed

| Component | Before (2 years ago) | After (Now) |
|-----------|---------------------|-------------|
| vLLM | 0.2.6 | 0.6.3.post1 |
| Python | 3.10 | 3.10-3.12 |
| FastAPI | 0.108.0 | 0.115.5 |
| Modal | 0.62.124 | 0.64.142 |
| Response Format | Custom | OpenAI Standard |
| API Endpoints | /completion only | /v1/completions + legacy |
| Streaming | Custom SSE | OpenAI SSE with [DONE] |

## 🎓 Learn More

For current production models and features, check out:
- [OpenRouter Models](https://openrouter.ai/models)
- [OpenRouter Docs](https://openrouter.ai/docs/quickstart)

## 📄 License

See LICENSE file for details.

---

**Status**: ✅ Updated and maintained (Nov 2024)
**Version**: 2.0.0
