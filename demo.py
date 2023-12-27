import gradio as gr
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer, LlamaForSequenceClassification, PreTrainedTokenizerFast, BitsAndBytesConfig
import torch
import numpy as np
from tqdm.auto import tqdm
from peft import PeftModel
from scipy.special import expit as sigmoid
import joblib
# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.5
# Tokenization and Encoding
model_name = "bert-base-uncased"

# 載入模型和分詞器
tokenizer_bert = BertTokenizer.from_pretrained(model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
bert_model.load_state_dict(torch.load("models/pretrained_model.pt"))

# 其他模型的設定和函數...
## TFIDF
def dummy(text):
    """
    A dummy function to use as tokenizer for TfidfVectorizer. It returns the text as it is since we already tokenized it.
    """
    return text

# 加載您的自定義 tokenizer
def load_tfidf_tokenizer():
    # 這裡假設您已經保存了 tokenizer
    raw_tokenizer = joblib.load('models/sentencepiece_tokenizer.pkl')
    # 加載保存的 TfidfVectorizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return tokenizer, vectorizer

def preprocess_text(text, tokenizer, vectorizer):
    # 使用 tokenizer 處理文本
    tokenized_text = tokenizer.tokenize(text)
    # 將 tokenized 文本轉換為 Tfidf 特徵
    tf_test = vectorizer.transform([' '.join(tokenized_text)])
    return tf_test

tfidf_tokenizer, tfidf_vectorizer = load_tfidf_tokenizer()
# 加載模型進行推論
print("Loading tfidf model...")
tfidf_model = joblib.load('models/sentencepiece_constructions_model.pkl')

# 其他模型的設定和函數...
## glove
def load_glove_embeddings(path):
    embeddings_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# 加載您的自定義 tokenizer
def load_glove_tokenizer():
    # 這裡假設您已經保存了 tokenizer
    raw_tokenizer = joblib.load('models/sequencepiece_tokenizer.pkl')
    # 加載保存的 TfidfVectorizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    return tokenizer
tokenizer_glove = load_glove_tokenizer()
embeddings_index = load_glove_embeddings('models/glove.6B.200d.txt')
glove_model = joblib.load('models/sequencepiece_glove_model.pkl')

# 其他模型的設定和函數...
## LLM
TARGET_MODEL = "mistralai/Mistral-7B-v0.1"
tokenizer_llm = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False)
tokenizer_llm.pad_token = tokenizer_llm.eos_token
PEFT_DIR = "models/mistral/"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


base_model = LlamaForSequenceClassification.from_pretrained(
    TARGET_MODEL,
    num_labels=2,
    quantization_config=bnb_config,
    device_map={"":0}
)
base_model.config.pretraining_tp = 1 
base_model.config.pad_token_id = tokenizer_llm.pad_token_id
score = torch.load(PEFT_DIR + "/score.pt")
base_model.score.weight = score
llm_model = PeftModel.from_pretrained(base_model, PEFT_DIR).to(device)

def classify_tfidf(text, tokenizer, vectorizer, model):
    # 預處理文本
    processed_text = preprocess_text(text, tokenizer, vectorizer)
    probs = model.predict_proba(processed_text)[:,1]
    prediction = (probs > threshold).astype(int)
    return "AI-generated" if prediction else "Student"


def classify_bert(text, tokenizer, model):
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().detach().numpy()
    probs = sigmoid(logits[:, 1])
    prediction = (probs > threshold).astype(int)
    return "AI-generated" if prediction else "Student"

def classify_glove(text, tokenizer, model):
    # Tokenize text
    tokenized_text = tokenizer.tokenize(text)
    # 將 tokenized 文本轉換為 GloVe 嵌入的平均值
    embedding_matrix = np.zeros((len(tokenized_text), 200))
    for i, word in enumerate(tokenized_text):
        embedding_matrix[i] = embeddings_index.get(word, embeddings_index["unk"])
    tf_text = np.mean(embedding_matrix, axis=0).reshape(1, -1)
    # 使用模型進行預測
    probs = model.predict_proba(tf_text)[:, 1]
    prediction = (probs > threshold).astype(int)
    return "AI-generated" if prediction else "Student"

def classify_llm(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = sigmoid(outputs.logits.cpu().detach().numpy()[:, 1])    
    probs_str = '%f' % probs
    prediction = (probs > threshold).astype(int)
    return "AI-generated" if prediction else "Student"

with gr.Blocks() as demo:
    with gr.Tab("TFIDF"):
        gr.Interface(
            fn=lambda text: classify_tfidf(text, tfidf_tokenizer, tfidf_vectorizer, tfidf_model),
            inputs=gr.Textbox(placeholder="請輸入一段文字..."),
            outputs="label"
        )
        
    with gr.Tab("DistilBERT"):
        gr.Interface(
            fn=lambda text: classify_bert(text, tokenizer_bert, bert_model),
            inputs=gr.Textbox(placeholder="請輸入一段文字..."),
            outputs="label"
        )

    with gr.Tab("GloVe"):
        gr.Interface(
            fn=lambda text: classify_glove(text, tokenizer_glove, glove_model),
            inputs=gr.Textbox(placeholder="請輸入一段文字..."),
            outputs="label"
        )
    with gr.Tab("Mistral 7B"):
        gr.Interface(
            fn=lambda text: classify_llm(text, tokenizer_llm, llm_model),
            inputs=gr.Textbox(placeholder="請輸入一段文字..."),
            outputs="label"
        )

demo.launch(share=True)
