from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from llm2vectrain.access_token import access_token
import torch


def load_llm2vec_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
    )
    config = AutoConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        token=access_token,
    )

    model = PeftModel.from_pretrained(
        model,
        "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
    )

    model = PeftModel.from_pretrained(
        model, "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
    )

    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
    return l2v