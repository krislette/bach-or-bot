from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from src.llm2vectrain.config import access_token
import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig
import os


def load_llm2vec_model():
    # Get cache directory from environment or use default
    cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/.cache/huggingface")

    model_id = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding=True, truncation=True, max_length=512, cache_dir=cache_dir
    )

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir
    )

    if torch.cuda.is_available():
        # GPU path: use bf16 for speed
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            token=access_token,
            cache_dir=cache_dir,
        )
    else:
        # CPU path: use float32 first, then quantize
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.float32,  # quantization requires fp32
            device_map="cpu",
            token=access_token,
            cache_dir=cache_dir,
        )

    try:
        from torchao.quantization import quantize_

        print("[INFO] Applying torchao quantization for CPU...")
        quant_config = Int8WeightOnlyConfig(group_size=None)
        print("[INFO] Applying torchao quantization with Int8WeightOnlyConfig...")
        quantize_(model, quant_config)
    except ImportError:
        print("[WARNING] torchao not installed. Run: pip install torchao")
        print("[WARNING] Falling back to non-quantized CPU model.")

    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
    return l2v
