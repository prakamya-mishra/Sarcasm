import time

def populate_embeddings_dict():
    starttime = time.time()
    with open('data/processed/glove.6B.300d.txt','r') as file:
        for line in file:
            values = line.split()
            word = values[0]
            word_embedding = np.asarray(values[1:])
            embeddings[word] = word_embedding
    endtime = time.time()
    print("Time taken to load embeddings:- ")
    print(endtime - starttime)

def embedding_lookup(x,embedding_dim=300):
    if(len(embeddings) == 0):
        populate_embeddings_dict()
    embedding = []
    for i in range(0,len(x)):
        if(x[i] in embeddings):
            embedding.append(embeddings[x[i]])
        else:
            zero_arr = np.zeros(embedding_dim).tolist()
            embedding.append(zero_arr)
    return np.array(embedding)