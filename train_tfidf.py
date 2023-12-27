import gc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

sub = pd.read_csv('dataset/sample_submission.csv')
org_test = pd.read_csv('dataset/test_essays.csv')
org_train = pd.read_csv('dataset/train_essays.csv')
daigt_train = pd.read_csv("dataset/train_v2_drcat_02.csv", sep=',')
aug_train = pd.read_csv('dataset/final_train.csv')
aug_test = pd.read_csv('dataset/final_test.csv')
train = pd.read_csv('dataset/train_1226.csv')
train = train
test = org_test

train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)
# y_train = train['generated'].values
y_train = train['label'].values

LOWERCASE = False
VOCAB_SIZE = 30522

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
joblib.dump(raw_tokenizer, 'models/sentencepiece_tokenizer.pkl')

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

tokenized_texts_test = []

# Tokenize test set with new tokenizer
for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))


# Tokenize train set
tokenized_texts_train = []

for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))

def dummy(text):
    """
    A dummy function to use as tokenizer for TfidfVectorizer. It returns the text as it is since we already tokenized it.
    """
    return text

# Fitting TfidfVectoizer on train set
def fitting_vectorizer_on_train(a, b):
    print("Fitting vectorizer on train set...")
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
        tokenizer = dummy,
        preprocessor = dummy,
        token_pattern = None#, strip_accents='unicode'
                                )

    vectorizer.fit(a)

    # Getting vocab
    vocab = vectorizer.vocabulary_

#     print(vocab)


    # Here we fit our vectorizer on train set but this time we use vocabulary from test fit.
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                                analyzer = 'word',
                                tokenizer = dummy,
                                preprocessor = dummy,
                                token_pattern = None#, strip_accents='unicode'
                                )

    tf_test = vectorizer.fit_transform(b)
    tf_train = vectorizer.transform(a)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    del vectorizer
    gc.collect()
    return(tf_train, tf_test)  


# Fitting TfidfVectoizer on test set
def fitting_vectorizer_on_test(a, b):
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
        tokenizer = dummy,
        preprocessor = dummy,
        token_pattern = None#, strip_accents='unicode'
                                )

    vectorizer.fit(b)

    # Getting vocab
    vocab = vectorizer.vocabulary_
    # Here we fit our vectorizer on train set but this time we use vocabulary from test fit.
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                                analyzer = 'word',
                                tokenizer = dummy,
                                preprocessor = dummy,
                                token_pattern = None#, strip_accents='unicode'
                                )

    tf_train = vectorizer.fit_transform(a)
    tf_test = vectorizer.transform(b)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    del vectorizer
    gc.collect()
    return(tf_train, tf_test)  

def calculate_voting(tf_train, tf_test, y_train):
    print("Calculating voting...")
    clf = MultinomialNB(alpha=0.02)
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
    p6={'n_iter': 2500,'verbose': -1,'objective': 'cross_entropy','metric': 'auc','learning_rate': 0.05073909898961407, \
        'colsample_bytree': 0.726023996436955, 'colsample_bynode': 0.5803681307354022, 'lambda_l1': 8.562963348932286, \
        'lambda_l2': 4.893256185259296, 'min_data_in_leaf': 115, 'max_depth': 23, 'max_bin': 898}
    lgb=LGBMClassifier(**p6)

    cat=CatBoostClassifier(
        iterations=2000,
        verbose=0,
        l2_leaf_reg=6.6591278779517808,
        learning_rate=0.005689066836106983,
        allow_const_label=True,
        subsample=0.4,
        loss_function='CrossEntropy'
    )
    
    weights = [10, 40, 40, 40]
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
    joblib.dump(ensemble, 'models/sentencepiece_constructions_model.pkl')
    final_preds = ensemble.predict_proba(tf_test)[:,1]
    # Garbage collection
    gc.collect()
    return(final_preds)

tf_train, tf_test = fitting_vectorizer_on_train(tokenized_texts_train, tokenized_texts_test)  
# tf_train, tf_test = fitting_vectorizer_on_test(tokenized_texts_train, tokenized_texts_test)  
final_preds_sentencePiece = calculate_voting(tf_train, tf_test, y_train)
_ = gc.collect()
