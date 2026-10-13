# Model-Modal
A serverless, OpenAI-compatible inference API for Qwen models, hosted on [Modal](https://modal.com/). This project uses FastAPI to serve Hugging Face transformers with GPU acceleration, complete with Bearer token authentication and native Server-Sent Events (SSE) streaming.

Perfect for connecting local coding agents like [Pi](https://pi.dev/) to powerful cloud GPUs without managing infrastructure.

## Features

- **Serverless GPUs**: Spin up A10, L40S, or H100 GPUs on demand.
- **OpenAI-Compatible**: Speaks the standard `/v1/chat/completions` protocol.
- **Native Streaming**: Full SSE streaming support for real-time token generation in terminal clients.
- **Secure**: Bearer token authentication via Modal Secrets.
- **Warm Containers**: Keeps models loaded in VRAM with a 15-minute scale-down window to eliminate cold starts.
- **Dynamic Generation**: Automatically respects `max_tokens`, `temperature`, and `stream` parameters from client requests.

## Prerequisites

- A [Modal](https://modal.com/) account.
- The Modal Python package installed and authenticated (`modal setup`).
- [Pi CLI](https://pi.dev/) (v1.2.0+) for agent integration.

## 🛠️ Setup & Deployment

### 0. Clone this repository and install all dependencies on a local box:
```
git clone 
python3 -m venv venv
source venv/bin/activate
pip install modal fastapi uvicorn transformers 
```

### 1. Create your API Secret
Secure your endpoint by creating a Modal secret. Replace `your-super-secure-secret-token-here` with a strong password.

```bash
modal secret create my-web-auth-token API_SECRET_TOKEN="your-super-secure-secret-token-here"
```

### 2. Deploy to Modal
Deploy the FastAPI application to Modal's cloud infrastructure. The initial deployment will build the container image and download the model weights into the GPU.

```bash
modal deploy main.py
```

## Pi CLI Integration

To connect your local Pi agent to your Modal deployment, add your endpoint as a custom provider in Pi's configuration file.

Open your Pi models configuration:
```bash
nano ~/.pi/agent/models.json
```

Add the following JSON configuration (replace `name` in the URL with your Modal workspace name, and `your-token` with your secret):

```json
{
  "$schema": "https://pi.dev/schemas/models.schema.json",
  "providers": {
    "ollama": {
      "baseUrl": "https://name--qwen-inference-api-fastapi-app.us.modal.run/v1",
      "api": "openai-completions",
      "apiKey": "your-token",
      "models": [
        { "id": "qwen3x" }
      ]
    }
  }
}
```

Once saved, you can select the model inside Pi using the `/model` command.

## API Usage

You can interact with the API directly using `curl` or any OpenAI-compatible SDK.

**Streaming Request Example:**
```bash
curl -X POST "https://name--qwen-inference-api-fastapi-app.us.modal.run/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "model": "qwen3x",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Write a short poem about deploying code to the cloud."}
    ]
  }'
```

## ⚙️ Model Configurations

You can easily swap out the model and GPU tier in `main.py` depending on your performance and budget requirements:

| Model | GPU Tier | Use Case |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` (180B) | **H100** | Maximum performance, complex reasoning |
| `Qwen/Qwen3.8-27B` | **L40S** | Optimal balance of speed and intelligence |
| `Qwen/Qwen3.5-0.8B` | **A10** | Cheapest, fastest, lightweight tasks |

*To change the model, update the `model` parameter in the `pipeline()` function and the `gpu` parameter in the `@app.function` decorator, then run `modal deploy main.py` again.*

## Monitoring & Logs

To view real-time logs from your Modal container (including model loading status and generation errors):

```bash
modal app logs qwen-inference-api
```

To force the container to scale down to zero (saving GPU costs) immediately after testing:

```bash
modal app stop qwen-inference-api
```
---
2026 [ ivan deus ]
