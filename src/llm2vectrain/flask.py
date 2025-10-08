import requests
from src.llm2vectrain.colab_url import colab_url

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