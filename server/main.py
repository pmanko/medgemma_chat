import torch
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading on Startup ---
# Use FastAPI's 'lifespan' to load models once when the app starts.
# This is much more efficient than loading them on every request.
models = {}

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
            models['phi3'] = pipeline(
                "text-generation",
                model="microsoft/Phi-3-mini-4k-instruct",
                device=device.type,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            logger.info("✅ Phi-3 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Phi-3 model: {e}")
            raise

        # 2. Load MedGemma for specialized medical Q&A (multimodal model)
        try:
            logger.info("Loading google/medgemma-4b-it...")
            medgemma_processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
            medgemma_model = AutoModelForImageTextToText.from_pretrained(
                "google/medgemma-4b-it",
                torch_dtype=torch.float16,
                device_map="auto" if device.type == "mps" else device
            )
            models['medgemma'] = {
                'processor': medgemma_processor, 
                'model': medgemma_model,
                'device': device
            }
            logger.info("✅ MedGemma model loaded successfully")
        except Exception as e:
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
class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="User prompt")
    max_new_tokens: int = Field(default=256, ge=1, le=1024, description="Maximum tokens to generate")

class PromptResponse(BaseModel):
    response: str = Field(..., description="Model generated response")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Server is running. Models are loaded."}

@app.post("/generate/phi3", response_model=PromptResponse)
def generate_phi3(request: PromptRequest):
    """Generate response using Phi-3 model for general conversations."""
    try:
        logger.info(f"Received Phi-3 prompt: {request.prompt[:100]}...")
        
        if 'phi3' not in models:
            raise HTTPException(status_code=503, detail="Phi-3 model not available")
        
        pipe = models['phi3']
        messages = [{"role": "user", "content": request.prompt}]
        
        output = pipe(
            messages,
            max_new_tokens=request.max_new_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=0.7,
        )
        
        response_text = output[0]['generated_text'].strip()
        logger.info(f"Generated Phi-3 response: {response_text[:100]}...")
        
        return {"response": response_text}
        
    except Exception as e:
        logger.error(f"Error generating Phi-3 response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.post("/generate/medgemma", response_model=PromptResponse)
def generate_medgemma(request: PromptRequest):
    """Generate response using MedGemma model for medical questions."""
    try:
        logger.info(f"Received MedGemma prompt: {request.prompt[:100]}...")
        
        if 'medgemma' not in models:
            raise HTTPException(status_code=503, detail="MedGemma model not available")
        
        processor = models['medgemma']['processor']
        model = models['medgemma']['model']
        device = models['medgemma']['device']

        # MedGemma uses a chat message format
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": request.prompt}]
            }
        ]
        
        # Process the input using the processor
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            generation = model.generate(
                **inputs, 
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
            generation = generation[0][input_len:]
        
        # Decode the response
        response_text = processor.decode(generation, skip_special_tokens=True).strip()
        
        logger.info(f"Generated MedGemma response: {response_text[:100]}...")
        
        return {"response": response_text}
        
    except Exception as e:
        logger.error(f"Error generating MedGemma response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
