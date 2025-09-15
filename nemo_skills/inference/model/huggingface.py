# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_skills.utils import get_logger_name

from .base import BaseModel
from .utils import WrapperAutoTokenizer, trim_after_stop_phrases

# Add evaluation plugin path to sys.path
eval_path = Path(__file__).parents[5] / "evaluation"
if str(eval_path) not in sys.path:
    sys.path.insert(0, str(eval_path))

LOG = logging.getLogger(get_logger_name(__file__))


class HuggingFaceModel(BaseModel):
    """Direct HuggingFace model with optional attention plugin support."""
    
    MODEL_PROVIDER = "huggingface"
    
    def __init__(
        self,
        model: str,
        attention_method: Optional[Union[str, List[str]]] = None,
        tokenizer: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env_var: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        # Plugin-specific parameters
        threshold: Optional[float] = None,
        softmax_thresh: Optional[float] = None,
        stride: int = 16,
        chunk_size: int = 2048,
        skip_first_n: int = 0,
        skip_last_n: int = 0,
        skip_interval: int = 0,
        collect_outputs: bool = False,
        collect_patterns: bool = False,
        collect_layer_stride: int = 4,
        collect_head_stride: int = 8,
        collect_q_stride: Optional[int] = None,
        dtype: str = "bfloat16",
        # BaseModel parameters
        host: str = "127.0.0.1",
        port: str = "5000",
        ssh_server: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        enable_soft_fail: bool = False,
        context_limit_retry_strategy: Optional[str] = None,
        num_special_tokens_budget: int = 100,
        **kwargs
    ):
        """
        Initialize HuggingFace model with optional plugin support.
        
        Args:
            model: Model name or path
            attention_method: Plugin configuration (e.g., "xattn", "flash_2_4:prefill,softmax_skip:decode")
            tokenizer: Tokenizer name or path (defaults to model)
            Plugin parameters and BaseModel parameters as listed
        """
        # Handle both string and list formats for attention_method
        # Check for both regular list and OmegaConf ListConfig (from Hydra)
        from omegaconf import ListConfig
        if isinstance(attention_method, (list, ListConfig)):
            # Convert list format ['method1:phase1', 'method2:phase2'] to string 'method1:phase1,method2:phase2'
            attention_method = ','.join(attention_method)
            
        self.attention_method = attention_method
        self.plugin = None
        self._vanilla_model = None
        self._vanilla_tokenizer = None
        self._tunnel = None  # Initialize to avoid AttributeError in __del__
        
        if attention_method:
            # Initialize with plugin system
            self._init_with_plugin(
                model=model,
                attention_method=attention_method,
                tokenizer=tokenizer,
                threshold=threshold,
                softmax_thresh=softmax_thresh,
                stride=stride,
                chunk_size=chunk_size,
                skip_first_n=skip_first_n,
                skip_last_n=skip_last_n,
                skip_interval=skip_interval,
                collect_outputs=collect_outputs,
                collect_patterns=collect_patterns,
                collect_layer_stride=collect_layer_stride,
                collect_head_stride=collect_head_stride,
                collect_q_stride=collect_q_stride,
                dtype=dtype,
                **kwargs
            )
            # Set model_name_or_path for BaseModel compatibility
            self.model_name_or_path = model
            self.tokenizer = None  # Plugin handles tokenizer
        else:
            # Initialize with BaseModel for vanilla generation
            # Set base_url before calling super().__init__
            self.base_url = base_url or ""  # Empty string to bypass URL construction
            super().__init__(
                model=model,
                tokenizer=tokenizer,
                api_key=api_key,
                api_key_env_var=api_key_env_var,
                base_url=self.base_url,
                max_retries=max_retries,
                host=host,
                port=port,
                ssh_server=ssh_server,
                ssh_key_path=ssh_key_path,
                enable_soft_fail=enable_soft_fail,
                context_limit_retry_strategy=context_limit_retry_strategy,
                num_special_tokens_budget=num_special_tokens_budget,
            )
    
    def _init_with_plugin(
        self,
        model: str,
        attention_method: str,
        tokenizer: Optional[str],
        **kwargs
    ):
        """Initialize model with attention plugin system."""
        from evaluation.scripts.common.model_utils import (
            create_model_wrapper,
            ensure_model_loaded,
        )
        
        # Create args namespace matching call_api.py pattern
        args = Namespace(
            model_name_or_path=model,
            model=model,
            attention_method=attention_method,
            tokenizer=tokenizer or model,
            # Add all plugin-specific parameters
            threshold=kwargs.get('threshold'),
            softmax_thresh=kwargs.get('softmax_thresh'),
            stride=kwargs.get('stride', 16),
            chunk_size=kwargs.get('chunk_size', 2048),
            skip_first_n=kwargs.get('skip_first_n', 0),
            skip_last_n=kwargs.get('skip_last_n', 0),
            skip_interval=kwargs.get('skip_interval', 0),
            collect_outputs=kwargs.get('collect_outputs', False),
            collect_patterns=kwargs.get('collect_patterns', False),
            collect_layer_stride=kwargs.get('collect_layer_stride', 4),
            collect_head_stride=kwargs.get('collect_head_stride', 8),
            collect_q_stride=kwargs.get('collect_q_stride'),
            dtype=kwargs.get('dtype', 'bfloat16'),
            # Generation parameters
            temperature=kwargs.get('temperature', 0.0),
            top_p=kwargs.get('top_p', 0.95),
            top_k=kwargs.get('top_k', -1),
            stop=kwargs.get('stop_phrases'),
            stop_words=kwargs.get('stop_phrases'),  # Alias for compatibility
        )
        
        # Extract generation parameters
        max_new_tokens = kwargs.get('tokens_to_generate', 512)
        
        # Create model wrapper using plugin system
        LOG.info(f"Creating HuggingFace model with plugin: {args.attention_method}")
        self.plugin = create_model_wrapper(args, max_new_tokens, None)
        
        # Ensure model is loaded (for lazy-loading plugins)
        ensure_model_loaded(self.plugin)
        
        LOG.info(f"Successfully initialized plugin: {args.attention_method}")
    
    def _load_vanilla_model(self):
        """Load vanilla HuggingFace model and tokenizer."""
        if self._vanilla_model is not None:
            return
        
        LOG.info(f"Loading vanilla HuggingFace model: {self.model_name_or_path}")
        
        # Load tokenizer
        self._vanilla_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        if self._vanilla_tokenizer.pad_token is None:
            self._vanilla_tokenizer.padding_side = "left"
            self._vanilla_tokenizer.pad_token = self._vanilla_tokenizer.eos_token
            self._vanilla_tokenizer.pad_token_id = self._vanilla_tokenizer.eos_token_id
        
        # Load model
        self._vanilla_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        LOG.info("Vanilla model loaded successfully")
    
    def _build_completion_request_params(
        self,
        prompt: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        timeout: Optional[int] = None,
        stop_phrases: Optional[List[str]] = None,
        stream: bool = False,
        reasoning_effort: Optional[str] = None,
        extra_body: Optional[dict] = None,
        tools: Optional[List[dict]] = None,
    ) -> dict:
        """Build parameters for vanilla model generation."""
        if self._vanilla_model is None:
            self._load_vanilla_model()
        
        # Tokenize input
        inputs = self._vanilla_tokenizer(prompt, return_tensors="pt")
        
        # Build generation kwargs
        generation_kwargs = {
            "input_ids": inputs["input_ids"].to(self._vanilla_model.device),
            "attention_mask": inputs["attention_mask"].to(self._vanilla_model.device),
            "max_new_tokens": tokens_to_generate,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self._vanilla_tokenizer.pad_token_id,
            "eos_token_id": self._vanilla_tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            generation_kwargs["top_p"] = top_p
            if top_k > 0:
                generation_kwargs["top_k"] = top_k
        
        if repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        
        return generation_kwargs
    
    def _build_chat_request_params(
        self,
        messages: List[dict],
        stream: bool,
        **kwargs
    ) -> dict:
        """Build parameters for chat completion (converts to completion)."""
        if self._vanilla_model is None:
            self._load_vanilla_model()
        
        # Apply chat template (with thinking mode enabled by default for Qwen)
        prompt = self._vanilla_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Use completion params
        return self._build_completion_request_params(
            prompt=prompt,
            stream=stream,
            **kwargs
        )
    
    def _build_request_params(
        self,
        prompt: Union[str, List[dict]],
        stream: bool,
        **kwargs
    ) -> dict:
        """Build request parameters based on prompt type."""
        if isinstance(prompt, list):
            # It's a chat request
            return self._build_chat_request_params(prompt, stream, **kwargs)
        else:
            # It's a completion request
            return self._build_completion_request_params(prompt, stream=stream, **kwargs)
    
    async def generate_async(
        self,
        prompt: Union[str, List[dict]],
        tokens_to_generate: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 42,
        stop_phrases: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
        timeout: Optional[float] = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        include_response: bool = False,
        extra_body: Optional[dict] = None,
    ) -> dict:
        """Generate text using plugin or vanilla model."""
        if self.plugin:
            # Use plugin's process_batch method
            LOG.debug(f"Using plugin {self.attention_method} for generation")
            
            # Convert prompt to string if needed
            if isinstance(prompt, list):
                if hasattr(self.plugin, 'tokenizer'):
                    prompt_str = self.plugin.tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Fallback: just use the last user message
                    prompt_str = prompt[-1].get('content', '')
            else:
                prompt_str = prompt
            
            # Plugin process_batch expects kwargs matching its generation config
            plugin_kwargs = {}
            if tokens_to_generate:
                plugin_kwargs['max_new_tokens'] = tokens_to_generate
            if temperature > 0:
                plugin_kwargs['temperature'] = temperature
                plugin_kwargs['top_p'] = top_p
                plugin_kwargs['do_sample'] = True
                if top_k > 0:
                    plugin_kwargs['top_k'] = top_k
            else:
                plugin_kwargs['do_sample'] = False
            if stop_phrases:
                plugin_kwargs['stop'] = stop_phrases
            
            # Call plugin
            results = self.plugin.process_batch([prompt_str], **plugin_kwargs)
            result = results[0] if results else {"text": [""]}
            
            # Convert plugin output format to NeMo-Skills format
            generated_text = result.get("text", [""])[0] if isinstance(result.get("text"), list) else result.get("text", "")
            
            # Apply stop phrase removal if needed
            if remove_stop_phrases and stop_phrases:
                generated_text = trim_after_stop_phrases(generated_text, stop_phrases)
            
            output = {
                "generation": generated_text,
            }
            
            if include_response:
                output["response"] = result
            
            return output
        else:
            # Use vanilla model generation
            LOG.debug("Using vanilla HuggingFace generation")
            
            # Build generation parameters
            kwargs = {
                "tokens_to_generate": tokens_to_generate or 512,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "random_seed": random_seed,
                "stop_phrases": stop_phrases,
                "top_logprobs": top_logprobs,
                "timeout": timeout,
                # stream is passed separately, not in kwargs
                "reasoning_effort": reasoning_effort,
                "tools": tools,
                "extra_body": extra_body,
            }
            
            # Get generation kwargs
            generation_kwargs = self._build_request_params(prompt, stream, **kwargs)
            
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
            
            # Run generation in thread pool to make it async
            loop = asyncio.get_event_loop()
            generated_ids = await loop.run_in_executor(
                None,
                lambda: self._vanilla_model.generate(**generation_kwargs)
            )
            
            # Decode output
            input_length = generation_kwargs["input_ids"].shape[1]
            generated_tokens = generated_ids[:, input_length:]
            generated_text = self._vanilla_tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True
            )
            
            # Apply stop phrase removal
            if remove_stop_phrases and stop_phrases:
                generated_text = trim_after_stop_phrases(generated_text, stop_phrases)
            
            output = {
                "generation": generated_text,
            }
            
            if include_response:
                output["response"] = {
                    "model": self.model_name_or_path,
                    "choices": [{
                        "text": generated_text,
                        "finish_reason": "stop"
                    }]
                }
            
            return output
    
    def generate_sync(self, prompt: Union[str, List[dict]], **kwargs) -> dict:
        """Synchronous generation wrapper."""
        return asyncio.run(self.generate_async(prompt, **kwargs))
    
    def _get_tokenizer_endpoint(self) -> Optional[str]:
        """HuggingFace models don't have tokenizer endpoints."""
        return None
    
    def cleanup(self):
        """Clean up resources."""
        if self.plugin:
            if hasattr(self.plugin, 'cleanup'):
                self.plugin.cleanup()
        
        if self._vanilla_model is not None:
            self._vanilla_model.to("cpu")
            del self._vanilla_model
            self._vanilla_model = None
        
        if self._vanilla_tokenizer is not None:
            del self._vanilla_tokenizer
            self._vanilla_tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
