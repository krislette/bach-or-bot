from src.llm2vectrain.model import load_llm2vec_model
from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
import torch

# For Single Input
def l2vec_single_train(l2v, lyrics):
    vectors = l2v.encode([lyrics])
    return vectors

# For Batch Processing
def l2vec_train(l2v, lyrics_list):
    with torch.no_grad():
        vectors = l2v.encode(lyrics_list)  # lyrics_list: list of strings
    return vectors