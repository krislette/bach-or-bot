import torch
import joblib
from sklearn.decomposition import IncrementalPCA

# For Single Input (Inference)
def l2vec_single_train(
    l2v, lyrics,
    pca_path="data/processed/pca_model.pkl" #Change directory
):
    # Encode single song
    vectors = l2v.encode([lyrics])  # shape: (1, hidden_dim)

    # Load trained IncrementalPCA
    pca = joblib.load(pca_path)

    # Apply transform (don't refit!)
    reduced_vectors = pca.transform(vectors)  # shape: (1, n_components)
    return reduced_vectors


# For Batch Processing (Training)
def l2vec_train(
    l2v, lyrics_batches,
    n_components=300,
    batch_size=512,
    save_pca=True,
    pca_path="data/processed/pca_model.pkl"
):
    # Initialize IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # First pass: fit PCA incrementally across all batches
    for lyrics_list in lyrics_batches:
        with torch.no_grad():
            vectors = l2v.encode(lyrics_list).detach().cpu().numpy()
        ipca.partial_fit(vectors)
        print(f"Fitted batch of size {len(lyrics_list)}")

    # Second pass: transform all batches consistently
    reduced_all = []
    for lyrics_list in lyrics_batches:
        with torch.no_grad():
            vectors = l2v.encode(lyrics_list).detach().cpu().numpy()
        reduced_vectors = ipca.transform(vectors)
        reduced_all.append(reduced_vectors)
        print(f"Processed batch of size {len(lyrics_list)}")

    # Save PCA object for later use in inference
    if save_pca:
        joblib.dump(ipca, pca_path)

    return reduced_all