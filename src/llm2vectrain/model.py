from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from src.llm2vectrain.access_token import access_token
import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig


def load_llm2vec_model():
    
    model_id = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding = True, truncation = True, max_length = 512)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    if torch.cuda.is_available():
    # GPU path: use bf16 for speed
        model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=access_token,
    )
    else:
    # CPU path: use float32 first, then quantize
        model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.float32,   # quantization requires fp32
        device_map="cpu",
        token=access_token,
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