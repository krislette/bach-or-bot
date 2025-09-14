
from src.preprocessing.preprocessor import dataset_read, bulk_preprocessing

def train_pipeline():

    # Instantiate X and Y vectors
    X, Y = None, None

    # Get batches from dataset and return full Y labels
    batches, Y = dataset_read()
    batch_count = 1

    # Instantiate SpecTTTra and LLM2Vec Models
    #spectttra_model = SpecTTTra()
    #llm2vec_model = LLM2Vec()

    for batch in batches:
        audio, lyrics = None, None      # Gets rid of previous values consuming current memory
        audio, lyrics = bulk_preprocessing(batch, batch_count)
        batch_count += 1

        # Call the train method for both models
        #audio = specttra_train(spectttra_model, audio) 
        #lyrics = llm2vec_train(llm2vec_model, lyrics) 

    # Concatenate the two vectors for each value
    #conc_feat = audio + lyrics

    #X = np.array(conc_feat)
    #Y = np.array(labels)

    #model_train(X, Y)


if __name__ == "__main__":
    train_pipeline()