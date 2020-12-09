import pickle


# NOTE: filepath should have its basename match a key in the trained dictionary

def save_checkpoint(filepath, **data):
    """
    Save trained matrix (i.e. cosine similarity) so we don't have to recompute
    each time we run the script along with other relavant data

    Pickle object will be a dictionary matching the keyword argument data
    for example, 
        save_checkpoint(
            "models/description_tfidf",
            cosine_sim=cosine_sim_matrix,
            indices=indices_df
        )
    """
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def get_checkpoint(filepath, train_function, *args, **kwargs):
    """
    Try loading checkpoint, if does not exist, train then save
    
    args, kwargs are anything that needs to be passed to the train_function
    """
    try:
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)

    except FileNotFoundError:
        checkpoint = train_function(*args, **kwargs)
        save_checkpoint(filepath, **checkpoint)
    
    return checkpoint
