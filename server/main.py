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

def prepare_conversation_context(conversation_history: List[ChatMessage], current_prompt: str, system_prompt: str, max_history_messages: int = 8):
    """
    Prepare conversation context using sliding window approach with optional summarization.
    Keeps the most recent exchanges while managing token limits.
    """
    # Build messages array starting with system prompt
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Handle long conversations with summarization for Phi-3 too
    if len(conversation_history) > max_history_messages * 2:  # If we have more than double the limit
        logger.info(f"Long Phi-3 conversation ({len(conversation_history)} messages), creating summary for context preservation...")
        
        # Create summary of earlier conversation using Phi-3 itself for consistency
        summary = create_conversation_summary(conversation_history, 'phi3', max_messages_to_summarize=15)
        if summary:
            messages.append({"role": "system", "content": f"Previous conversation context: {summary}"})
            logger.info("Added Phi-3-generated summary to context")
        
        # Add only the most recent exchanges
        recent_history = conversation_history[-max_history_messages:]
        logger.info(f"Using last {len(recent_history)} messages plus summary for Phi-3")
    else:
        # Use regular sliding window for shorter conversations
        recent_history = conversation_history[-max_history_messages:] if conversation_history else []
    
    # Add conversation history
    for msg in recent_history:
        messages.append({"role": msg.role, "content": msg.content})
    
    # Add current user message
    messages.append({"role": "user", "content": current_prompt})
    
    return messages

def generate_medgemma_summary(conversation_history: List[ChatMessage], max_messages_to_summarize: int = 10) -> str:
    """
    Use MedGemma itself to generate intelligent medical conversation summaries.
    This leverages MedGemma's medical knowledge to create better context preservation.
    """
    if not conversation_history or len(conversation_history) <= 4:
        return ""
    
    # Get messages to summarize (exclude the most recent ones)
    messages_to_summarize = conversation_history[:-4]  # Keep last 4 messages intact
    if len(messages_to_summarize) > max_messages_to_summarize:
        messages_to_summarize = messages_to_summarize[-max_messages_to_summarize:]
    
    if not messages_to_summarize:
        return ""
    
    try:
        if 'medgemma' not in models:
            logger.warning("MedGemma not available for summarization, falling back to simple summary")
            return create_simple_summary(messages_to_summarize)
        
        processor = models['medgemma']['processor']
        model = models['medgemma']['model']
        
        # Build conversation text for summarization
        conversation_text = ""
        for msg in messages_to_summarize:
            role_prefix = "Patient" if msg.role == "user" else "Assistant"
            conversation_text += f"{role_prefix}: {msg.content}\n"
        
        # Create summarization prompt
        summarization_messages = [{
            "role": "user",
            "content": [{
                "type": "text", 
                "text": f"""Please create a concise medical summary of this conversation focusing on:
1. Key symptoms and patient concerns
2. Important medical advice or recommendations given
3. Any treatments, medications, or diagnoses mentioned
4. Relevant medical history or context

Keep the summary under 200 words and focus on medically relevant information.

Conversation to summarize:
{conversation_text}

Summary:"""
            }]
        }]
        
        # Generate summary using MedGemma
        inputs = processor.apply_chat_template(
            summarization_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Ensure we don't exceed context limits for summarization
        input_len = inputs["input_ids"].shape[-1]
        if input_len > 800:  # Leave room for generation
            inputs["input_ids"] = inputs["input_ids"][:, -800:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -800:]
            input_len = 800
        
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=150,  # Limit summary length
                do_sample=False
            )
            generation = generation[0][input_len:]
        
        # Decode the summary
        summary = processor.decode(generation, skip_special_tokens=True).strip()
        
        logger.info(f"Generated MedGemma summary: {summary[:100]}...")
        return f"MEDICAL CONTEXT SUMMARY: {summary}"
        
    except Exception as e:
        logger.warning(f"Failed to generate MedGemma summary: {e}")
        return create_simple_summary(messages_to_summarize)

def create_simple_summary(messages_to_summarize: List[ChatMessage]) -> str:
    """
    Fallback simple summary when MedGemma summarization fails.
    """
    user_messages = [msg.content for msg in messages_to_summarize if msg.role == "user"]
    assistant_messages = [msg.content for msg in messages_to_summarize if msg.role == "assistant"]
    
    summary_parts = []
    
    if user_messages:
        user_summary = " ".join(user_messages)[:200]
        summary_parts.append(f"Patient reported: {user_summary}")
    
    if assistant_messages:
        assistant_summary = " ".join(assistant_messages)[:200]
        summary_parts.append(f"Medical advice: {assistant_summary}")
    
    if summary_parts:
        return f"CONVERSATION CONTEXT: {' | '.join(summary_parts)}"
    
    return ""

def generate_phi3_summary(conversation_history: List[ChatMessage], max_messages_to_summarize: int = 15) -> str:
    """
    Use Phi-3 itself to generate intelligent conversation summaries.
    This leverages Phi-3's general knowledge to create coherent context preservation.
    """
    if not conversation_history or len(conversation_history) <= 4:
        return ""
    
    # Get messages to summarize (exclude the most recent ones)
    messages_to_summarize = conversation_history[:-4]  # Keep last 4 messages intact
    if len(messages_to_summarize) > max_messages_to_summarize:
        messages_to_summarize = messages_to_summarize[-max_messages_to_summarize:]
    
    if not messages_to_summarize:
        return ""
    
    try:
        if 'phi3' not in models:
            logger.warning("Phi-3 not available for summarization, falling back to simple summary")
            return create_simple_summary(messages_to_summarize)
        
        pipe = models['phi3']
        
        # Build conversation text for summarization
        conversation_text = ""
        for msg in messages_to_summarize:
            role_prefix = "User" if msg.role == "user" else "Assistant"
            conversation_text += f"{role_prefix}: {msg.content}\n"
        
        # Create summarization prompt for Phi-3
        summarization_messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise conversation summaries."},
            {"role": "user", "content": f"""Please create a concise summary of this conversation focusing on:
1. Key topics and user concerns discussed
2. Important information or advice provided
3. Any specific requests, problems, or solutions mentioned
4. Relevant context that should be remembered

Keep the summary under 300 words and focus on the most important information.

Conversation to summarize:
{conversation_text}

Summary:"""}
        ]
        
        # Generate summary using Phi-3
        output = pipe(
            summarization_messages,
            max_new_tokens=200,  # Limit summary length
            return_full_text=False,
            do_sample=True,
            temperature=0.3,  # Lower temperature for more focused summaries
            pad_token_id=pipe.tokenizer.eos_token_id,
            use_cache=False  # Fix for DynamicCache issue
        )
        
        summary = output[0]['generated_text'].strip()
        
        logger.info(f"Generated Phi-3 summary: {summary[:100]}...")
        return f"CONVERSATION CONTEXT: {summary}"
        
    except Exception as e:
        logger.warning(f"Failed to generate Phi-3 summary: {e}")
        return create_simple_summary(messages_to_summarize)

def create_conversation_summary(conversation_history: List[ChatMessage], model_name: str, max_messages_to_summarize: int = 10) -> str:
    """
    Create a conversation summary using the specified model for consistency.
    Uses the same model that's handling the current conversation.
    """
    if model_name == 'medgemma':
        return generate_medgemma_summary(conversation_history, max_messages_to_summarize)
    elif model_name == 'phi3':
        return generate_phi3_summary(conversation_history, max_messages_to_summarize)
    else:
        logger.warning(f"Unknown model '{model_name}' for summarization, using simple summary")
        messages_to_summarize = conversation_history[:-4] if len(conversation_history) > 4 else []
        if len(messages_to_summarize) > max_messages_to_summarize:
            messages_to_summarize = messages_to_summarize[-max_messages_to_summarize:]
        return create_simple_summary(messages_to_summarize)

def prepare_medgemma_conversation_context(conversation_history: List[ChatMessage], current_prompt: str, system_prompt: str, max_history_messages: int = 4):
    """
    Prepare conversation context for MedGemma using its multimodal message format.
    Uses intelligent summarization when conversation gets too long.
    """
    messages = []
    
    # Add system prompt if provided (keep it short for MedGemma)
    if system_prompt:
        # Truncate system prompt if too long
        truncated_system = system_prompt[:200] if len(system_prompt) > 200 else system_prompt
        if len(system_prompt) > 200:
            logger.info(f"Truncated system prompt from {len(system_prompt)} to 200 characters for MedGemma")
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": truncated_system}]
        })
    
    # Handle conversation history with summarization
    if len(conversation_history) > max_history_messages * 2:  # If we have more than double the limit
        logger.info(f"Long MedGemma conversation ({len(conversation_history)} messages), creating summary for context preservation...")
        
        # Create summary of earlier conversation using MedGemma's medical expertise
        summary = create_conversation_summary(conversation_history, 'medgemma', max_messages_to_summarize=10)
        if summary:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": summary}]
            })
            logger.info("Added MedGemma-generated medical summary to context")
        
        # Add only the most recent exchanges
        recent_history = conversation_history[-max_history_messages:]
        logger.info(f"Using last {len(recent_history)} messages plus summary for MedGemma")
    else:
        # Use regular sliding window for shorter conversations
        recent_history = conversation_history[-max_history_messages:] if conversation_history else []
    
    # Add recent conversation history
    for msg in recent_history:
        # Truncate very long messages to prevent context overflow
        truncated_content = msg.content[:400] if len(msg.content) > 400 else msg.content
        messages.append({
            "role": msg.role,
            "content": [{"type": "text", "text": truncated_content}]
        })
    
    # Add current user message (also truncate if very long)
    truncated_prompt = current_prompt[:500] if len(current_prompt) > 500 else current_prompt
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": truncated_prompt}]
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
        # Use intelligent summarization to handle longer conversations
        messages = prepare_medgemma_conversation_context(
            request.conversation_history,
            request.prompt,
            request.system_prompt,
            max_history_messages=4  # Slightly increased since we now have smart summarization
        )
        
        logger.info(f"Prepared {len(messages)} messages for MedGemma")
        
        # Process the input using the processor
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        input_len = inputs["input_ids"].shape[-1]
        logger.info(f"Input sequence length: {input_len} tokens")
        
        # MedGemma has a context limit - truncate if necessary
        MAX_CONTEXT_LENGTH = 1024
        if input_len > MAX_CONTEXT_LENGTH:
            logger.warning(f"Input length ({input_len}) exceeds context limit ({MAX_CONTEXT_LENGTH}), truncating...")
            inputs["input_ids"] = inputs["input_ids"][:, -MAX_CONTEXT_LENGTH:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -MAX_CONTEXT_LENGTH:]
            input_len = MAX_CONTEXT_LENGTH
        
        # Move to device with proper dtype
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=min(request.max_new_tokens, MAX_CONTEXT_LENGTH - input_len),
                do_sample=False  # Use deterministic generation for stability
            )
            generation = generation[0][input_len:]
        
        # Decode the response
        response_text = processor.decode(generation, skip_special_tokens=True).strip()
        
        execution_time = time.time() - start_time
        logger.info(f"Generated MedGemma response in {execution_time:.2f}s: {response_text[:100]}...")
        
        return {"response": response_text}
        
    except RuntimeError as e:
        if "size" in str(e) and "tensor" in str(e).lower():
            logger.error(f"MedGemma tensor size error (likely context length issue): {e}")
            raise HTTPException(status_code=400, detail="Input too long for MedGemma model. Please try a shorter message or clear conversation history.")
        else:
            logger.error(f"Runtime error in MedGemma: {e}")
            raise HTTPException(status_code=500, detail=f"Model runtime error: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating MedGemma response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
