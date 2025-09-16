
from src.preprocessing.preprocessor import dataset_read, bulk_preprocessing
from src.spectttra.spectttra_trainer import spectttra_train

def train_pipeline():

    # Instantiate X and Y vectors
    X, Y = None, None

    # Get batches from dataset and return full Y labels
    batches, Y = dataset_read()
    batch_count = 1

     # Instantiate LLM2Vec Model
      #llm2vec_model = LLM2Vec()

    for batch in batches:
        audio, lyrics = None, None  # Gets rid of previous values consuming current memory
        audio, lyrics = bulk_preprocessing(batch, batch_count)
        batch_count += 1

        # Call the train method for both models
        audio_features = spectttra_train(audio) # Extract embeddings with SpecTTTra
        #lyrics = llm2vec_train(llm2vec_model, lyrics) 

    # TODO: Concatenate the vectors of audio_features + lyrics_features
    # conc_feat = audio_features + lyrics_features

    # TODO: Final model training stage
    # X = np.array(conc_feat)
    # Y = np.array(labels)

    # model_train(X, Y)


if __name__ == "__main__":
    train_pipeline()
