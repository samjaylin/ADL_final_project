
import os
import gc
import pandas as pd
import numpy as np
from tokenizers import (
    normalizers,
    pre_tokenizers,
    SentencePieceBPETokenizer
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

import joblib

############ input training data #################
sub = pd.read_csv('dataset/sample_submission.csv')
org_test = pd.read_csv('dataset/test_essays.csv')
org_train = pd.read_csv('dataset/train_essays.csv')
daigt_train = pd.read_csv("dataset/train_v2_drcat_02.csv", sep=',')
aug_train = pd.read_csv('dataset/final_train.csv')
aug_test = pd.read_csv('dataset/final_test.csv')
train = daigt_train
test = org_test

train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)
y_train = train['label'].values

EMBEDDING_DIM = 200
MAX_NB_WORDS = 30522
LOWERCASE = False

from tokenizers import (
    normalizers,
    pre_tokenizers,
    SentencePieceBPETokenizer
)

###### Preparing pre-trained embedding book ######
embeddings_index = {}
f = open('models/glove.6B.200d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

########## preparing tokenizer ###################
# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = SentencePieceBPETokenizer()

# Adding normalization and pre_tokenizer
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# Adding special tokens and creating trainer instance
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

# Creating huggingface dataset object
dataset = Dataset.from_pandas(train[['text']])

def train_corp_iter():
    """
    A generator function for iterating over a dataset in chunks.
    """    
    for i in range(0, len(dataset), 300):
        yield dataset[i : i + 300]["text"]

# Training from iterator REMEMBER it's training on test set...
raw_tokenizer.train_from_iterator(train_corp_iter())
joblib.dump(raw_tokenizer, 'models/sequencepiece_tokenizer.pkl')
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

####### prepare for training ############

tokenized_texts_test = []

# Tokenize test set with new tokenizer
for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

# Tokenize train set
tokenized_texts_train = []

for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


tf_test = []
for sentence in tqdm(tokenized_texts_test):
    #     embedding_matrix = np.zeros((len(sentence) + 1, EMBEDDING_DIM))
    embedding_matrix = np.zeros((len(sentence), EMBEDDING_DIM))
    for i in range(len(sentence)):
        e = embeddings_index.get(sentence[i], embeddings_index["unk"])
        embedding_matrix[i] = e
    tf_test.append(sum(embedding_matrix)/len(embedding_matrix))
    
    
tf_train = []
for sentence in tqdm(tokenized_texts_train):
    embedding_matrix = np.zeros((len(sentence), EMBEDDING_DIM))
    for i in range(len(sentence)):
        e = embeddings_index.get(sentence[i], embeddings_index["unk"])
        embedding_matrix[i] = e
    tf_train.append(sum(embedding_matrix)/len(embedding_matrix))


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

tf_train = scaler.fit_transform(np.array(tf_train))
tf_test = scaler.fit_transform(np.array(tf_test))
print(tf_test.shape)
print(tf_train.shape)


def calculate_voting(tf_train, tf_test, y_train):
    clf = MultinomialNB(alpha=0.02)
    clf2 = MultinomialNB(alpha=0.01)
    
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
    p6={'n_iter': 1500,'verbose': -1,'objective': 'binary','metric': 'auc','learning_rate': 0.05073909898961407, 'colsample_bytree': 0.726023996436955, 'colsample_bynode': 0.5803681307354022, 'lambda_l1': 8.562963348932286, 'lambda_l2': 4.893256185259296, 'min_data_in_leaf': 115, 'max_depth': 23, 'max_bin': 898}
    lgb=LGBMClassifier(**p6)

    cat=CatBoostClassifier(
        iterations=1000,
        verbose=0,
        l2_leaf_reg=6.6591278779517808,
        learning_rate=0.005689066836106983,
        allow_const_label=True,
        task_type="CPU",
    )
    
    weights = [9.5, 43, 42, 42]
    # Creating the ensemble model
    ensemble = VotingClassifier(estimators=[
        ('mnb', clf),
        ('sgd', sgd_model),
        ('lgb', lgb), 
        ('cat', cat)],
        weights = [w/sum(weights) for w in weights],
        voting='soft',
        n_jobs=-1)

    # Fit the ensemble model
    ensemble.fit(tf_train, y_train)
    final_preds = ensemble.predict_proba(tf_test)[:,1]
    print(final_preds)
    joblib.dump(ensemble, 'sentencepiece_glove_model.pkl')
    # Garbage collection
    gc.collect()
    return(final_preds)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
final_preds_sentencePiece = calculate_voting(tf_train, tf_test, y_train)
_ = gc.collect()