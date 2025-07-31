import torch
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading on Startup ---
# Use FastAPI's 'lifespan' to load models once when the app starts.
# This is much more efficient than loading them on every request.
models = {}

# Simple model status tracking
model_status = {
    'phi3': 'not_loaded',
    'medgemma': 'not_loaded',
    'server_start_time': time.time()
}

def get_optimal_device():
    """Detect the best available device for model inference."""
    if torch.backends.mps.is_available():
        logger.info("Using Metal Performance Shaders (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model loading and cleanup during app lifecycle."""
    try:
        logger.info("Loading models... This may take a few minutes.")
        
        # Detect optimal device
        device = get_optimal_device()

        # 1. Load Phi-3-mini using the pipeline API for general chat
        try:
            logger.info("Loading microsoft/Phi-3-mini-4k-instruct...")
            model_status['phi3'] = 'loading'
            start_time = time.time()
            
            models['phi3'] = pipeline(
                "text-generation",
                model="microsoft/Phi-3-mini-4k-instruct",
                device=device.type,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            
            load_time = time.time() - start_time
            model_status['phi3'] = 'ready'
            logger.info(f"✅ Phi-3 model loaded successfully in {load_time:.2f}s")
        except Exception as e:
            model_status['phi3'] = 'error'
            logger.error(f"Failed to load Phi-3 model: {e}")
            raise

        # 2. Load MedGemma for specialized medical Q&A (multimodal model)
        try:
            logger.info("Loading google/medgemma-4b-it...")
            model_status['medgemma'] = 'loading'
            start_time = time.time()
            
            medgemma_processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
            medgemma_model = AutoModelForImageTextToText.from_pretrained(
                "google/medgemma-4b-it",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            models['medgemma'] = {
                'processor': medgemma_processor, 
                'model': medgemma_model,
                'device': device
            }
            
            load_time = time.time() - start_time
            model_status['medgemma'] = 'ready'
            logger.info(f"✅ MedGemma model loaded successfully in {load_time:.2f}s")
        except Exception as e:
            model_status['medgemma'] = 'error'
            logger.error(f"Failed to load MedGemma model: {e}")
            raise
        
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    
    yield # The server is now running and handling requests

    # This code runs ONCE when the server is shutting down.
    logger.info("Clearing models from memory...")
    models.clear()


# --- FastAPI App Initialization ---
app = FastAPI(
    lifespan=lifespan,
    title="Dual LLM API",
    description="Serves Phi-3 and MedGemma models."
)

# Allow requests from our frontend (which will be on a different "origin")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models for Request/Response ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp")

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="User prompt")
    system_prompt: str = Field(default="", max_length=1000, description="System prompt to guide model behavior")
    max_new_tokens: int = Field(default=512, ge=1, le=1024, description="Maximum tokens to generate")
    conversation_history: List[ChatMessage] = Field(default=[], description="Previous conversation messages")
    conversation_id: Optional[str] = Field(default=None, description="Unique conversation identifier")

class PromptResponse(BaseModel):
    response: str = Field(..., description="Model generated response")

def prepare_conversation_context(conversation_history: List[ChatMessage], current_prompt: str, system_prompt: str, max_history_messages: int = 10):
    """
    Prepare conversation context using sliding window approach.
    Keeps the most recent exchanges while managing token limits.
    """
    # Build messages array starting with system prompt
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history (sliding window of last N messages)
    recent_history = conversation_history[-max_history_messages:] if conversation_history else []
    
    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})
    
    # Add current user message
    messages.append({"role": "user", "content": current_prompt})
    
    return messages

def prepare_medgemma_conversation_context(conversation_history: List[ChatMessage], current_prompt: str, system_prompt: str, max_history_messages: int = 6):
    """
    Prepare conversation context for MedGemma using its multimodal message format.
    """
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
    
    # Add conversation history (sliding window)
    recent_history = conversation_history[-max_history_messages:] if conversation_history else []
    
    for msg in recent_history:
        messages.append({
            "role": msg.role,
            "content": [{"type": "text", "text": msg.content}]
        })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": current_prompt}]
    })
    
    return messages

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Server is running. Models are loaded."}

@app.get("/health")
def health_check():
    """Simple health check endpoint to monitor server and model status."""
    uptime = time.time() - model_status['server_start_time']
    
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "models": {
            "phi3": model_status['phi3'],
            "medgemma": model_status['medgemma']
        },
        "timestamp": time.time()
    }

@app.post("/generate/phi3", response_model=PromptResponse)
def generate_phi3(request: PromptRequest):
    """Generate response using Phi-3 model for general conversations with conversation history."""
    start_time = time.time()
    try:
        logger.info(f"Received Phi-3 prompt: {request.prompt[:100]}...")
        logger.info(f"Conversation history length: {len(request.conversation_history)}")
        
        if 'phi3' not in models:
            raise HTTPException(status_code=503, detail="Phi-3 model not available")
        
        pipe = models['phi3']
        
        # Prepare conversation context with history
        messages = prepare_conversation_context(
            request.conversation_history, 
            request.prompt, 
            request.system_prompt,
            max_history_messages=8  # Keep last 8 messages for Phi-3
        )
        
        logger.info(f"Prepared {len(messages)} messages for Phi-3")
        
        output = pipe(
            messages,
            max_new_tokens=request.max_new_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=0.7,
            pad_token_id=pipe.tokenizer.eos_token_id,
            use_cache=False  # Fix for DynamicCache issue
        )
        
        response_text = output[0]['generated_text'].strip()
        
        execution_time = time.time() - start_time
        logger.info(f"Generated Phi-3 response in {execution_time:.2f}s: {response_text[:100]}...")
        
        return {"response": response_text}
        
    except Exception as e:
        logger.error(f"Error generating Phi-3 response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.post("/generate/medgemma", response_model=PromptResponse)
def generate_medgemma(request: PromptRequest):
    """Generate response using MedGemma model for medical questions with conversation history."""
    start_time = time.time()
    try:
        logger.info(f"Received MedGemma prompt: {request.prompt[:100]}...")
        logger.info(f"Conversation history length: {len(request.conversation_history)}")
        
        if 'medgemma' not in models:
            raise HTTPException(status_code=503, detail="MedGemma model not available")
        
        processor = models['medgemma']['processor']
        model = models['medgemma']['model']
        device = models['medgemma']['device']

        # Prepare conversation context with history using MedGemma's multimodal format
        messages = prepare_medgemma_conversation_context(
            request.conversation_history,
            request.prompt,
            request.system_prompt,
            max_history_messages=6  # Keep last 6 messages for MedGemma to manage context length
        )
        
        logger.info(f"Prepared {len(messages)} messages for MedGemma")
        
        # Process the input using the processor
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=request.max_new_tokens,
                do_sample=False  # Use deterministic generation for stability
            )
            generation = generation[0][input_len:]
        
        # Decode the response
        response_text = processor.decode(generation, skip_special_tokens=True).strip()
        
        execution_time = time.time() - start_time
        logger.info(f"Generated MedGemma response in {execution_time:.2f}s: {response_text[:100]}...")
        
        return {"response": response_text}
        
    except Exception as e:
        logger.error(f"Error generating MedGemma response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
