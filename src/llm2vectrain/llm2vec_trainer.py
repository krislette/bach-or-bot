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