import requests
from scripts.colab_url import colab_url

def request_llm2vect_single(lyrics):
    """Request a single vector from the LLM2Vec model. 
    Args:
        lyrics (str): The lyrics to be vectorized.
    Returns:
        None: Prints the vector or an error message.
    Raises:
        Exception: If the request fails or the response is not as expected.
    """

    task_url = colab_url + "/single"

    response = requests.post(task_url, json={"lyrics": lyrics})

    if response.status_code == 200:
        data = response.json()
        vectors = data.get("vectors")
        ###print("Received vector shape:", len(vectors[0]), "dimensions")
        ###print(vectors)
        print("Single request successful. Vector shape:", len(vectors[0]), "dimensions")
        return vectors
    else:
        print("Error:", response.text)

def request_llm2vect_batch(lyrics_list):
    """Request a batch of vectors from the LLM2Vec model.
    Args:
        lyrics_list (list): A list of lyrics to be vectorized.
    Returns:
        None: Prints the vectors or an error message.
    Raises:
        Exception: If the request fails or the response is not as expected.
    """ 
    task_url = colab_url + "/batch"

    response = requests.post(task_url, json={"lyrics_list": lyrics_list})

    if response.status_code == 200:
        data = response.json()
        vectors = data.get("vectors")
        ##print("Amount of Songs:", len(vectors))
        ##print("Received vector shape:", len(vectors[0]), "dimensions")
        ##print("First vector:", vectors[0])
        print("Batch request successful. Number of songs processed:", len(vectors))
        return vectors
    else:
        print("Error:", response.text)

def request_spectttra_single(audio_tensor):
    """Request a single vector from the SpecTTTra model. 
    Args:
        audio_tensor (torch.Tensor): The audio tensor to be vectorized.
    Returns:
        None: Prints the vector or an error message.
    Raises:
        Exception: If the request fails or the response is not as expected.
    """

    task_url = colab_url + "/spectttra/predict"

    # Convert tensor to list for JSON serialization
    audio_list = audio_tensor.squeeze(0).tolist()  # Assuming audio_tensor shape is (1, num_samples)

    response = requests.post(task_url, json={"audio": audio_list})

    if response.status_code == 200:
        data = response.json()
        vector = data.get("vector")
        ###print("Received vector shape:", len(vector), "dimensions")
        ###print(vector)
        print("Single request successful. Vector shape:", len(vector), "dimensions")
        return vector
    else:
        print("Error:", response.text)

def request_spectttra_batch(audio_tensors):
    """Request a batch of vectors from the SpecTTTra model.
    Args:
        audio_tensors (list): A list of audio tensors to be vectorized.
    Returns:
        None: Prints the vectors or an error message.
    Raises:
        Exception: If the request fails or the response is not as expected.
    """

    task_url = colab_url + "/spectttra/train"

    # Convert list of tensors to list of lists for JSON serialization
    audio_list = [tensor.squeeze(0).tolist() for tensor in audio_tensors]  # Assuming each tensor shape is (1, num_samples)

    response = requests.post(task_url, json={"audios": audio_list})

    if response.status_code == 200:
        data = response.json()
        vectors = data.get("vectors")
        ##print("Amount of Songs:", len(vectors))
        ##print("Received vector shape:", len(vectors[0]), "dimensions")
        ##print("First vector:", vectors[0])
        print("Batch request successful. Number of songs processed:", len(vectors))
        return vectors
    else:
        print("Error:", response.text)