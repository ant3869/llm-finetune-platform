"""
Inference Engine for Fine-Tuned Models.

Supports:
- Loading base model with LoRA adapters
- Interactive chat/completion
- Batch inference
- GGUF model loading via llama-cpp-python
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator
import torch

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Inference engine for testing fine-tuned models.
    
    Supports both:
    - HuggingFace models with LoRA adapters
    - GGUF models via llama-cpp-python
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_type = None  # "hf" or "gguf"
        self.model_path = None
        self.adapter_path = None
        
    def load_hf_model_with_adapter(
        self,
        base_model: str,
        adapter_path: str,
        load_in_4bit: bool = True
    ) -> bool:
        """
        Load a HuggingFace model with a LoRA adapter.
        
        Args:
            base_model: HuggingFace model ID or local path
            adapter_path: Path to the LoRA adapter
            load_in_4bit: Whether to use 4-bit quantization
            
        Returns:
            True if successful
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            
            logger.info(f"Loading base model: {base_model}")
            
            # Quantization config
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            # Load adapter
            logger.info(f"Loading adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
            self.model.eval()
            self.model_type = "hf"
            self.model_path = base_model
            self.adapter_path = adapter_path
            
            logger.info("Model loaded successfully with adapter")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_gguf_model(
        self,
        gguf_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
    ) -> bool:
        """
        Load a GGUF model using llama-cpp-python.
        
        Args:
            gguf_path: Path to the GGUF file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            
        Returns:
            True if successful
        """
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading GGUF model: {gguf_path}")
            
            self.model = Llama(
                model_path=gguf_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            
            self.model_type = "gguf"
            self.model_path = gguf_path
            self.tokenizer = None  # GGUF uses internal tokenizer
            
            logger.info("GGUF model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeated tokens
            stop_sequences: Sequences that stop generation
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_*_model() first.")
        
        if self.model_type == "hf":
            return self._generate_hf(
                prompt, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, stop_sequences
            )
        elif self.model_type == "gguf":
            return self._generate_gguf(
                prompt, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, stop_sequences
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _generate_hf(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_sequences: Optional[List[str]],
    ) -> str:
        """Generate using HuggingFace model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Apply stop sequences
        if stop_sequences:
            for stop in stop_sequences:
                if stop in generated:
                    generated = generated.split(stop)[0]
        
        return generated.strip()
    
    def _generate_gguf(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_sequences: Optional[List[str]],
    ) -> str:
        """Generate using GGUF model."""
        output = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stop=stop_sequences or [],
        )
        
        return output["choices"][0]["text"].strip()
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """
        Stream text generation token by token.
        
        Yields:
            Generated tokens one at a time
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")
        
        if self.model_type == "gguf":
            # GGUF supports native streaming
            for output in self.model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            ):
                yield output["choices"][0]["text"]
        else:
            # For HF models, generate all at once (streaming is more complex)
            result = self.generate(prompt, max_new_tokens, temperature, top_p)
            yield result
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat-style generation with message history.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Assistant response
        """
        # Format messages into prompt
        prompt = self._format_chat_prompt(messages, system_prompt)
        
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_sequences=["User:", "Human:", "\n\nUser", "\n\nHuman"],
        )
    
    def _format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format messages into a prompt string."""
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "system":
                parts.append(f"System: {content}")
        
        # Add prompt for assistant response
        parts.append("Assistant:")
        
        return "\n".join(parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"loaded": False}
        
        info = {
            "loaded": True,
            "type": self.model_type,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
        }
        
        if self.model_type == "hf":
            info["device"] = str(next(self.model.parameters()).device)
            info["dtype"] = str(next(self.model.parameters()).dtype)
        elif self.model_type == "gguf":
            info["context_size"] = self.model.n_ctx()
        
        return info
    
    def unload(self):
        """Unload the model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_type = None
        self.model_path = None
        self.adapter_path = None
        
        logger.info("Model unloaded")
