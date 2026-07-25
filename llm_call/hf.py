"""Model abstractions and logits processing utilities for TalkPlay agents.

Provides `LLM` for chat completions and tool calling, and a
`InsertToolTokenProcessor` to inject tool markers during generation.
"""
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import pydantic

class JudgeScore(pydantic.BaseModel):
    personalization_score: int = pydantic.Field(ge=1, le=5)
    personalization_reason: str
    explanation_score: int = pydantic.Field(ge=1, le=5)
    explanation_reason: str

class HF_LLM_Client:
    """
    A language model wrapper for music recommendation with tool calling capabilities.
    This class provides a unified interface for interacting with large language models
    to generate tool calls and responses for music recommendation tasks.
    Args:
        tools (list): List of available tools/functions that the model can call
        model_name (str, optional): HuggingFace model identifier. Defaults to "Qwen/Qwen3-4B-AWQ"
        device (str, optional): Device to run the model on. Defaults to "cuda"
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 8192
    """
    def __init__(self,
        model_name="Qwen/Qwen3-4B",
        device="cuda",
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.quantization_config = None
        self.model, self.tokenizer = self._load_model(device=self.device, quantization_config=self.quantization_config)
        if "gemma" in self.model_name.lower():
            self.sampling_parmas = {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "do_sample": True,
            }
        else:
            self.sampling_parmas = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "do_sample": True,
            }

    def _load_model(self, device: torch.device, quantization_config):
        """Load the model and tokenizer.
        Returns:
            tuple: `(model, tokenizer)` ready for inference.
        """
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model.to(device)
        model.eval()
        return model, tokenizer

    def chat_completion(self, messages, max_new_tokens=1024, schema_type="judge"):
        """Generate a natural-language response grounded in recommendation data.
        Args:
            messages (list[dict]): List of messages to steer answer generation.
            max_new_tokens (int): Maximum tokens to generate.
        Returns:
            tuple[str, str]: Model input text and final answer text.
        """
        if schema_type == "judge":
            response_schema = JudgeScore
        else:
            response_schema = None
        model_input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer(model_input_text,
            return_tensors="pt",
            # padding=True,
            # truncation=True,
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.sampling_parmas
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        model_output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return model_input_text, model_output_text


class HF_Judge_LLM_Client:
    def __init__(self,
        model_name="google/gemma-4-31B-it",
        device="cuda",
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # 최소 리소스 사용을 위해 float16 사용
        )
        self.model, self.tokenizer = self._load_model(device=self.device, quantization_config=self.quantization_config)
        self.sampling_parmas = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "do_sample": True,
        }

    def _load_model(self, device: torch.device, quantization_config):
        """Load the model and tokenizer.
        Returns:
            tuple: `(model, tokenizer)` ready for inference.
        """
        if device.type == "cuda":
            dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model.to(device)
        model.eval()
        return model, tokenizer

    def chat_completion(self, query, max_new_tokens=1024, schema_type="judge"):
        """Generate a natural-language response grounded in recommendation data.
        Args:
            messages (list[dict]): List of messages to steer answer generation.
            max_new_tokens (int): Maximum tokens to generate.
        Returns:
            tuple[str, str]: Model input text and final answer text.
        """
        if schema_type == "judge":
            text_schema = {
                "personalization_score": int,
                "personalization_reason": str,
                "explanation_score": int,
                "explanation_reason": str,
            }
        else:
            text_schema = None
        model_input = [
                {"role": "system","content": f"You are an expert at structured data extraction. Return ONLY valid JSON matching this schema: {text_schema}. Do not include markdown code blocks, backticks, or any extra text."},
                {"role": "user", "content": f"{query}"},
        ]
        model_input_text = self.tokenizer.apply_chat_template(
            model_input,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer(model_input_text,
            return_tensors="pt",
            # padding=True,
            # truncation=True,
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.sampling_parmas
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        model_output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        model_output_text = re.sub(r"^```(?:json)?\s*", "", model_output_text.strip())
        model_output_text = re.sub(r"\s*```$", "", model_output_text.strip())
        return model_output_text
