import torch
import logging
import time
import gc
import warnings
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration - Easy to add/remove/disable models
MODEL_CONFIG = {
    "general": {
        "enabled": True,
        "model_id": "google/gemma-3-1b-it",  # Change this to switch models
        "display_name": "Gemma 3 1B",
        "description": "Mobile-optimized general chat (English-only)",
        "icon": "📱",
        "download_size": "~2.5GB",
        "expected_response_time": "1-3 seconds",
        "max_tokens": 1000,
        "model_kwargs": {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "device_map": "auto",  # Optimize device placement
        },
        "trust_remote_code": False,
        "use_cache": True,  # Set to False if model has caching issues
    },
    "medgemma": {
        "enabled": True,
        "model_id": "google/medgemma-4b-it",
        "display_name": "MedGemma 4B",
        "description": "Specialized medical Q&A",
        "icon": "🏥",
        "download_size": "~18GB",
        "expected_response_time": "20-40 seconds",
        "max_tokens": 800,
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        },
        "trust_remote_code": False,
        "use_cache": True,
    },
    # Add new models here:
    # "coding": {
    #     "enabled": False,
    #     "model_id": "google/codegemma-2b",
    #     "display_name": "CodeGemma 2B",
    #     ...
    # }
}

# Default system prompts optimized for each model based on best practices
DEFAULT_SYSTEM_PROMPTS = {
    "general": MODEL_CONFIG["general"]["display_name"] + " - You are a helpful, accurate, and concise AI assistant optimized for mobile and edge devices. Provide clear, well-structured responses. Use markdown formatting for better readability when appropriate. Note: This model is English-only.",
    
    "medgemma": "You are a medical AI assistant. Provide accurate, evidence-based medical information. Always include appropriate disclaimers when discussing health topics. Format responses with clear structure using markdown. Keep responses comprehensive but concise."
}

# Response guidance to ensure proper formatting and prevent cutoffs
RESPONSE_GUIDANCE = {
    "medgemma": """

Response guidelines:
- Aim for 100-200 words for comprehensive medical explanations
- Use markdown formatting (**bold**, *emphasis*, bullet points, headers)
- Structure responses: Overview → Details → Key takeaways
- Always complete your thoughts - don't cut off mid-sentence
- Include medical disclaimers when appropriate ("Consult a healthcare professional")
- Use clear headings to organize complex information""",
    
    "general": """

Response guidelines:
- Provide complete, well-structured answers
- Use markdown formatting for clarity (**bold**, *emphasis*, bullet points)
- Aim for thoroughness while remaining concise
- Always finish your complete thought before ending"""
}

# System prompt optimization constants
SYSTEM_PROMPT_LIMITS = {
    "medgemma": {
        "max_total_tokens": 800,  # Conservative limit for system prompt (out of 8192 total)
        "default_priority": 1,    # Keep default prompts
        "user_priority": 2,       # User prompts are important
        "guidance_priority": 3    # Response guidance is MOST important (prevents cutoffs)
    },
    "general": {
        "max_total_tokens": 1000,  # Phi-3 can handle more
        "default_priority": 1,
        "user_priority": 2,
        "guidance_priority": 3
    }
}

# Memory management utilities
def intelligent_prompt_optimization(user_prompt: str, model_name: str) -> str:
    """
    Intelligent system prompt optimization based on token limits and priority.
    Uses hierarchical truncation to preserve the most important guidance.
    Based on prompt optimization best practices for LLM performance.
    """
    # Get model limits and components
    limits = SYSTEM_PROMPT_LIMITS.get(model_name, SYSTEM_PROMPT_LIMITS["general"])
    default_prompt = DEFAULT_SYSTEM_PROMPTS.get(model_name, "")
    response_guidance = RESPONSE_GUIDANCE.get(model_name, "")
    
    # Rough token estimation (4 chars ≈ 1 token for English)
    def estimate_tokens(text: str) -> int:
        return len(text) // 4
    
    # Build components with priorities
    components = [
        {"text": default_prompt, "priority": limits["default_priority"], "name": "default"},
        {"text": response_guidance, "priority": limits["guidance_priority"], "name": "guidance"},
    ]
    
    # Add user prompt if provided
    if user_prompt and user_prompt.strip():
        user_text = f"\n\nAdditional instructions: {user_prompt.strip()}"
        components.append({"text": user_text, "priority": limits["user_priority"], "name": "user"})
    
    # Sort by priority (highest priority first)
    components.sort(key=lambda x: x["priority"], reverse=True)
    
    # Build prompt, respecting token limits
    final_components = []
    total_tokens = 0
    max_tokens = limits["max_total_tokens"]
    
    for component in components:
        component_tokens = estimate_tokens(component["text"])
        
        if total_tokens + component_tokens <= max_tokens:
            # Fits completely
            final_components.append(component)
            total_tokens += component_tokens
        elif total_tokens < max_tokens:
            # Partial fit - intelligently truncate
            remaining_tokens = max_tokens - total_tokens
            remaining_chars = remaining_tokens * 4
            
            if component["name"] == "user" and remaining_chars > 50:
                # Truncate user prompt gracefully
                truncated_text = component["text"][:remaining_chars-20] + "... [truncated]"
                final_components.append({**component, "text": truncated_text})
                logger.info(f"Truncated user prompt from {component_tokens} to ~{remaining_tokens} tokens for {model_name}")
                break
            elif component["name"] == "default" and remaining_chars > 100:
                # Truncate default prompt if necessary
                truncated_text = component["text"][:remaining_chars-20] + "..."
                final_components.append({**component, "text": truncated_text})
                logger.info(f"Truncated default prompt from {component_tokens} to ~{remaining_tokens} tokens for {model_name}")
                break
            # Never truncate response guidance - it's most important!
        else:
            # No room left
            break
    
    # Rebuild in logical order (default -> user -> guidance)
    ordered_texts = []
    component_dict = {comp["name"]: comp["text"] for comp in final_components}
    
    if "default" in component_dict:
        ordered_texts.append(component_dict["default"])
    if "user" in component_dict:
        ordered_texts.append(component_dict["user"])
    if "guidance" in component_dict:
        ordered_texts.append(component_dict["guidance"])
    
    combined_prompt = "".join(ordered_texts)
    
    logger.debug(f"Optimized system prompt for {model_name}: {estimate_tokens(combined_prompt)} tokens (~{len(combined_prompt)} chars)")
    return combined_prompt

# Keep backward compatibility
def combine_system_prompts(user_prompt: str, model_name: str) -> str:
    """Backward compatibility wrapper for intelligent prompt optimization."""
    return intelligent_prompt_optimization(user_prompt, model_name)

def cleanup_memory(aggressive=False):
    """Clean up GPU/MPS memory after model inference to prevent memory leaks.
    
    Args:
        aggressive: If True, performs more thorough cleanup (slower but frees more memory)
    """
    try:
        if aggressive:
            # Full cleanup for when memory usage is critical
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()  # Apple Silicon MPS memory cleanup
                torch.mps.synchronize()  # Wait for all MPS operations
                
            # More aggressive cleanup - collect multiple times
            for _ in range(3):
                gc.collect()
            # Clear any cached allocators
            if hasattr(torch.mps, 'clear_cache'):
                torch.mps.clear_cache()
                
            # Log memory status for aggressive cleanup
            if torch.backends.mps.is_available():
                allocated = torch.mps.current_allocated_memory() / 1024**3
                reserved = torch.mps.driver_allocated_memory() / 1024**3
                logger.info(f"MPS Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        else:
            # Minimal cleanup - just empty cache without expensive synchronization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()  # Just empty cache, no sync
                
        logger.debug("Memory cleanup completed")
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

# --- Model Loading on Startup ---
# Use FastAPI's 'lifespan' to load models once when the app starts.
# This is much more efficient than loading them on every request.
models = {}

# Simple model status tracking
model_status = {
    'general': 'not_loaded',
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

def load_text_generation_model(model_key: str, config: dict, device):
    """Load a text generation model based on configuration."""
    if not config.get("enabled", True):
        logger.info(f"⏭️ Skipping {config['display_name']} (disabled in config)")
        return None
    
    try:
        logger.info(f"Loading {config['display_name']} ({config['model_id']})...")
        model_status[model_key] = 'loading'
        start_time = time.time()
        
        # Log model details
        logger.info(f"  📊 Size: {config.get('download_size', 'Unknown')}")
        logger.info(f"  ⚡ Expected response time: {config.get('expected_response_time', 'Unknown')}")
        
        # Create pipeline - handle device_map vs device conflict
        model_kwargs = config.get("model_kwargs", {})
        pipeline_kwargs = {
            "task": "text-generation",
            "model": config["model_id"],
            "trust_remote_code": config.get("trust_remote_code", False),
            "model_kwargs": model_kwargs,
        }
        
        # Only add device if device_map is not specified (they're mutually exclusive)
        if "device_map" not in model_kwargs:
            pipeline_kwargs["device"] = device.type
            
        model = pipeline(**pipeline_kwargs)
        
        load_time = time.time() - start_time
        model_status[model_key] = 'ready'
        logger.info(f"✅ {config['display_name']} loaded successfully in {load_time:.2f}s")
        return model
        
    except Exception as e:
        model_status[model_key] = 'error'
        logger.error(f"Failed to load {config['display_name']}: {e}")
        if config.get("required", True):
            raise
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model loading and cleanup during app lifecycle."""
    try:
        logger.info("Loading models... This may take a few minutes.")
    
        # Detect optimal device
        device = get_optimal_device()

        # Load general chat model
        if "general" in MODEL_CONFIG:
            models['general'] = load_text_generation_model('general', MODEL_CONFIG['general'], device)

        # Load MedGemma (multimodal model)
        if "medgemma" in MODEL_CONFIG and MODEL_CONFIG["medgemma"].get("enabled", True):
            try:
                config = MODEL_CONFIG["medgemma"]
                logger.info(f"Loading {config['display_name']} ({config['model_id']})...")
                model_status['medgemma'] = 'loading'
                start_time = time.time()
                
                # Log model details
                logger.info(f"  📊 Size: {config.get('download_size', 'Unknown')}")
                logger.info(f"  ⚡ Expected response time: {config.get('expected_response_time', 'Unknown')}")
                
                # Use fast processor for better performance with medical images
                medgemma_processor = AutoProcessor.from_pretrained(
                    config["model_id"],
                    use_fast=True  # Enable fast image processor for better performance
                )
                logger.info("✅ Using fast image processor for MedGemma (optimized for performance)")
                
                # Load model with configuration
                model_kwargs = config.get("model_kwargs", {}).copy()
                model_kwargs["device_map"] = "auto"  # MedGemma needs device_map instead of device
                
                medgemma_model = AutoModelForImageTextToText.from_pretrained(
                    config["model_id"],
                    **model_kwargs
                )
                
                models['medgemma'] = {
                    'processor': medgemma_processor, 
                    'model': medgemma_model,
                    'device': device
                }
                
                load_time = time.time() - start_time
                model_status['medgemma'] = 'ready'
                logger.info(f"✅ {config['display_name']} loaded successfully in {load_time:.2f}s")
            except Exception as e:
                model_status['medgemma'] = 'error'
                logger.error(f"Failed to load {config['display_name']}: {e}")
                if config.get("required", True):
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
    max_new_tokens: int = Field(default=512, ge=1, le=2048, description="Maximum tokens to generate (higher values may be slower)")
    conversation_history: List[ChatMessage] = Field(default=[], description="Previous conversation messages")
    conversation_id: Optional[str] = Field(default=None, description="Unique conversation identifier")
    
    def get_optimized_max_tokens(self, model_name: str = "general") -> int:
        """Get performance-optimized max_new_tokens that works WITH summary system."""
        # Model-specific limits based on context handling capabilities
        if model_name == "medgemma":
            # MedGemma has stricter context limits (1024 tokens total)
            # Increased token limit for longer, more complete responses
            base_limit = min(self.max_new_tokens, 1200)  # Increased from 800
            summary_threshold = 8  # MedGemma summarizes at >8 messages
        else:
            # Phi-3 can handle longer sequences better
            base_limit = min(self.max_new_tokens, 1500) 
            summary_threshold = 16  # Phi-3 summarizes at >16 messages
        
        # Only apply history penalty if we're BEFORE summarization kicks in
        # After summarization, context is efficiently managed
        if len(self.conversation_history) <= summary_threshold:
            # Light penalty for building conversations (better UX for quick responses)
            history_penalty = min(len(self.conversation_history) * 5, 50)  # Max 50 token reduction
        else:
            # Summarization is handling context - no additional penalty needed
            history_penalty = 0
            logger.debug(f"Using full token budget - conversation summarization is managing context")
        
        # Apply performance optimization
        optimized = max(base_limit - history_penalty, 128)  # Minimum 128 tokens
        
        if optimized != self.max_new_tokens:
            logger.debug(f"Optimized max_new_tokens: {self.max_new_tokens} → {optimized} for {model_name} ({len(self.conversation_history)} msgs)")
        
        return optimized

class PromptResponse(BaseModel):
    response: str = Field(..., description="Model generated response")

def prepare_conversation_context(conversation_history: List[ChatMessage], current_prompt: str, system_prompt: str, max_history_messages: int = 8):
    """
    Prepare conversation context using sliding window approach with optional summarization.
    Keeps the most recent exchanges while managing token limits.
    """
    # Build messages array starting with combined system prompt
    messages = []
    
    # Use intelligent system prompt optimization
    optimized_system_prompt = intelligent_prompt_optimization(system_prompt, "general")
    if optimized_system_prompt:
        messages.append({"role": "system", "content": optimized_system_prompt})
    
    # Handle long conversations with summarization for Phi-3 too
    if len(conversation_history) > max_history_messages * 2:  # If we have more than double the limit
        logger.info(f"Long Phi-3 conversation ({len(conversation_history)} messages), creating summary for context preservation...")
        
        # Create summary of earlier conversation using Phi-3 itself for consistency
        summary = create_conversation_summary(conversation_history, 'general', max_messages_to_summarize=15)
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

def generate_general_summary(conversation_history: List[ChatMessage], max_messages_to_summarize: int = 15) -> str:
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
        if 'general' not in models:
            logger.warning("Phi-3 not available for summarization, falling back to simple summary")
            return create_simple_summary(messages_to_summarize)
        
        pipe = models['general']
        
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
    elif model_name == 'general':
        return generate_general_summary(conversation_history, max_messages_to_summarize)
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
    
    # Use intelligent system prompt optimization
    optimized_system_prompt = intelligent_prompt_optimization(system_prompt, "medgemma")
    if optimized_system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": optimized_system_prompt}]
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
    """Root endpoint showing available models and their status."""
    available_models = {}
    for key, config in MODEL_CONFIG.items():
        if config.get("enabled", True):
            available_models[key] = {
                "display_name": config["display_name"],
                "description": config["description"],
                "status": model_status.get(key, "not_loaded"),
                "endpoint": f"/generate/{key}"
            }
    
    return {
        "status": "Server is running",
        "available_models": available_models
    }

@app.post("/memory/cleanup")
def force_memory_cleanup():
    """Force aggressive memory cleanup to free up resources."""
    try:
        before_process = 0
        before_mps = 0
        
        # Get before stats
        try:
            process = psutil.Process()
            before_process = process.memory_info().rss / 1024**3
            
            if torch.backends.mps.is_available():
                before_mps = torch.mps.current_allocated_memory() / 1024**3
        except:
            pass
        
        # Perform aggressive cleanup
        cleanup_memory(aggressive=True)
        
        # Get after stats
        after_process = 0
        after_mps = 0
        try:
            process = psutil.Process()
            after_process = process.memory_info().rss / 1024**3
            
            if torch.backends.mps.is_available():
                after_mps = torch.mps.current_allocated_memory() / 1024**3
        except:
            pass
        
        return {
            "status": "success",
            "memory_freed": {
                "process_gb": round(before_process - after_process, 2),
                "mps_gb": round(before_mps - after_mps, 2)
            },
            "current_memory": {
                "process_gb": round(after_process, 2),
                "mps_gb": round(after_mps, 2)
            }
        }
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        return {"status": "error", "detail": str(e)}

@app.get("/health")
def health_check():
    """Enhanced health check endpoint with performance monitoring."""
    uptime = time.time() - model_status['server_start_time']
    
    # Memory usage info for performance monitoring
    memory_info = {}
    try:
        # System memory info
        process = psutil.Process()
        memory_info["process_memory_gb"] = round(process.memory_info().rss / 1024**3, 2)
        memory_info["process_memory_percent"] = round(process.memory_percent(), 1)
        
        # GPU/MPS memory info
        if torch.backends.mps.is_available():
            # Get MPS memory usage if available  
            current_allocated = torch.mps.current_allocated_memory() / 1024**3  # GB
            driver_allocated = torch.mps.driver_allocated_memory() / 1024**3  # GB
            memory_info["mps_allocated_gb"] = round(current_allocated, 2)
            memory_info["mps_reserved_gb"] = round(driver_allocated, 2)
        elif torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info["cuda_allocated_gb"] = round(current_allocated, 2)
    except Exception as e:
        logger.debug(f"Could not get memory info: {e}")
    
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "models": {
            "general": model_status['general'],
            "medgemma": model_status['medgemma']
        },
        "memory": memory_info,
        "performance_tips": {
            "optimal_max_tokens": "256-512 for speed, 1024+ for longer responses",
            "context_management": "Clear history if >20 messages for best performance"
        },
        "timestamp": time.time()
    }

@app.post("/generate/general", response_model=PromptResponse)
def generate_general(request: PromptRequest):
    """Generate response using general chat model with conversation history."""
    start_time = time.time()
    try:
        logger.info(f"Received general model prompt: {request.prompt[:100]}...")
        logger.info(f"Conversation history length: {len(request.conversation_history)}")
        
        if 'general' not in models:
            raise HTTPException(status_code=503, detail="General chat model not available")
        
        pipe = models['general']
        
        # Prepare conversation context with history
        messages = prepare_conversation_context(
            request.conversation_history, 
            request.prompt, 
            request.system_prompt,
            max_history_messages=8  # Keep last 8 messages for general model
        )
        
        logger.info(f"Prepared {len(messages)} messages for general model")
        
        # Get performance-optimized token limit
        optimized_max_tokens = request.get_optimized_max_tokens("general")
        
        # Performance-optimized generation for general model (fast greedy decoding)
        with torch.inference_mode():  # More efficient than torch.no_grad()
            # Temporarily disable the generation config defaults to avoid warnings
            original_generation_config = pipe.model.generation_config
            pipe.model.generation_config.do_sample = False
            pipe.model.generation_config.top_p = 1.0  # Disable top_p
            pipe.model.generation_config.top_k = 0    # Disable top_k
            
            try:
                output = pipe(
                    messages,
                    max_new_tokens=optimized_max_tokens,
                    return_full_text=False,
                    do_sample=False,  # Greedy decoding for maximum speed
                    pad_token_id=pipe.tokenizer.eos_token_id,
                    use_cache=MODEL_CONFIG['general'].get('use_cache', True),  # Use config setting
                    num_beams=1,  # Single beam for speed
                )
            finally:
                # Restore original config
                pipe.model.generation_config = original_generation_config
        
        response_text = output[0]['generated_text'].strip()
        
        # No automatic cleanup - let PyTorch/Python handle memory naturally
        
        execution_time = time.time() - start_time
        logger.info(f"Generated {MODEL_CONFIG['general']['display_name']} response in {execution_time:.2f}s: {response_text[:100]}...")
        
        return {"response": response_text}
        
    except Exception as e:
        logger.error(f"Error generating {MODEL_CONFIG.get('general', {}).get('display_name', 'general')} response: {e}")
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
        # MedGemma actual context window is 8192 tokens, use reasonable limit for generation
        MAX_CONTEXT_LENGTH = 2048  # Increased from 1024, allows for longer conversations
        if input_len > MAX_CONTEXT_LENGTH:
            logger.warning(f"Input length ({input_len}) exceeds context limit ({MAX_CONTEXT_LENGTH}), truncating...")
            inputs["input_ids"] = inputs["input_ids"][:, -MAX_CONTEXT_LENGTH:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -MAX_CONTEXT_LENGTH:]
            input_len = MAX_CONTEXT_LENGTH
        
        # Move to device with proper dtype
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        
        # Get performance-optimized token limit for MedGemma
        optimized_max_tokens = request.get_optimized_max_tokens("medgemma")
        final_max_tokens = min(optimized_max_tokens, MAX_CONTEXT_LENGTH - input_len)
        
        if final_max_tokens < 50:
            logger.warning(f"Very low token budget ({final_max_tokens}), context may be too long")
        
        # Performance-optimized generation for MedGemma
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=final_max_tokens,
                do_sample=False,  # Deterministic for medical consistency
                use_cache=True,  # Enable KV caching for performance
                pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None,
                # Optimize for Apple Silicon MPS
                num_beams=1,  # Greedy decoding for speed
            )
            generation = generation[0][input_len:]
        
        # Decode the response
        response_text = processor.decode(generation, skip_special_tokens=True).strip()
        
        # No automatic cleanup - let PyTorch/Python handle memory naturally
        
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
