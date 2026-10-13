# main.py (with Token Auth)
# Modal QWEN 3.8 Qwen/Qwen3.8-Flash-Next : 180B  : H100
# Qwen/Qwen3.8-27B   : optimal  : L40S
# Qwen/Qwen3.5-0.8B   :  cheapest  : A10
### modal secret create my-web-auth-token API_SECRET_TOKEN="your-super-secure-secret-token-here"
### modal deploy main.py
## standard OpenAI chat completion schemas so Pi can communicate with it natively
###
import os
import modal
import json
import uuid
import time
from contextlib import asynccontextmanager
from threading import Thread

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from transformers import pipeline, TextIteratorStreamer

app = modal.App("qwen-inference-api")

# Added 'accelerate' for proper GPU memory mapping with device_map="auto"
image = modal.Image.debian_slim().uv_pip_install(
    "transformers[torch]", "fastapi", "uvicorn", "accelerate"
)

auth_scheme = HTTPBearer()

# Global variable to hold the model pipeline
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    print("Loading Qwen model into GPU memory... This may take a few minutes.")
    
    chatbot = pipeline(
        "text-generation", 
        model="Qwen/Qwen3.5-0.8B",
        device_map="auto", 
        torch_dtype="auto"
    )
    print("Model loaded successfully! Ready to accept requests.")
    yield

web_app = FastAPI(lifespan=lifespan)

@app.function(
    gpu="a10", 
    image=image, 
    scaledown_window=900, 
    secrets=[modal.Secret.from_name("my-web-auth-token")]
)
@modal.asgi_app()
def fastapi_app():
    return web_app

@web_app.post("/v1/chat/completions")
def chat_completions(request: dict, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    expected_token = os.environ.get("API_SECRET_TOKEN")
    if not expected_token or token.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    messages = request.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="The 'messages' array is required.")
        
    # Extract Pi's generation preferences dynamically
    stream = request.get("stream", False)
    max_tokens = request.get("max_tokens", 1024) or request.get("max_completion_tokens", 1024)
    temperature = request.get("temperature", 0.7)
    model_name = request.get("model", "qwen3x")

    try:
        if stream:
            # Initialize the text streamer to yield tokens as they generate
            streamer = TextIteratorStreamer(
                chatbot.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            generation_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                return_full_text=False,
                streamer=streamer,
            )
            
            # Run the pipeline in a background thread so FastAPI can yield chunks as they arrive
            thread = Thread(target=chatbot, args=(messages,), kwargs=generation_kwargs)
            thread.start()
            
            def event_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4()}"
                created = int(time.time())
                
                # Yield chunks as the model generates them
                for new_text in streamer:
                    if new_text:
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": new_text},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send the final chunk with the finish_reason (This is what Pi was waiting for!)
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                
                thread.join()

            return StreamingResponse(event_stream(), media_type="text/event-stream")
            
        else:
            # Non-streaming fallback (Standard JSON)
            result = chatbot(messages, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, return_full_text=False)
            response_content = result[0]["generated_text"]
            
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop"
                }]
            }
            
    except Exception as e:
        # Catch inference errors (like context length exceeded) and return them cleanly
        raise HTTPException(status_code=500, detail=str(e))

