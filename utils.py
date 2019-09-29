import numpy as np
from numpy import nan
        
def get_rows(dataset, max_comment_length, max_parent_comment_length):
    comments = []
    parent_comments = []
    labels = []
    for index, row in dataset.iterrows():
        if(len(row[COMMENT_LABEL]) <= max_comment_length and len(row[PARENT_COMMENT_LABEL]) <= max_parent_comment_length 
            and row[COMMENT_LABEL] is not nan and row[PARENT_COMMENT_LABEL] is not nan 
            and len(row[COMMENT_LABEL]) > 0 and len(row[PARENT_COMMENT_LABEL]) > 0):
            comments.append(row[COMMENT_LABEL])
            parent_comments.append(row[PARENT_COMMENT_LABEL])
            labels.append(row['label'])
    return pd.DataFrame({COMMENT_LABEL: comments, PARENT_COMMENT_LABEL: parent_comments, 'label': labels})

def sample_training_data(processe, batch_id):
    mask = np.random.rand(dataset.shape[0]) < TRAIN_SIZE
    #dataset[~mask].to_csv('data/test/batch_' + str(batch_id) + ".csv")
    return dataset[mask]