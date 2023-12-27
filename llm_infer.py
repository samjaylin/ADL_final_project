from transformers import AutoTokenizer, LlamaForSequenceClassification
from datasets import Dataset

from transformers import DataCollatorWithPadding
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType # type: ignore
from transformers import BitsAndBytesConfig
import torch
from transformers import AutoTokenizer, LlamaForSequenceClassification

from transformers import TrainingArguments, Trainer

from scipy.special import expit as sigmoid
import numpy as np

#TARGET_MODEL = "/kaggle/input/mistral-7b-v0-1/Mistral-7B-v0.1"
#TARGET_MODEL = "/kaggle/input/llama-2/pytorch/7b-hf/1"

TARGET_MODEL = "mistralai/Mistral-7B-v0.1"


# %% Directory settings

# ====================================================
# Directory settings
# ====================================================
from pathlib import Path

OUTPUT_DIR = Path(f"./")
#OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PEFT_DIR = Path(f"./mistral/")

INPUT_DIR = Path("./data/")


import pandas as pd
test_df = pd.read_csv(INPUT_DIR / "test_essays.csv", sep=',')
print(f'test_df.shape: {test_df.shape}')


test_df.head(3)

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# from pandas
test_ds = Dataset.from_pandas(test_df)

def preprocess_function(examples, max_length=512):
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=True)

test_tokenized_ds = test_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

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
base_model.config.pretraining_tp = 1 # 1 is 7b
base_model.config.pad_token_id = tokenizer.pad_token_id
score = torch.load(PEFT_DIR / "score.pt")
base_model.score.weight = score


model = PeftModel.from_pretrained(base_model, str(PEFT_DIR))
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

pred_output = trainer.predict(test_tokenized_ds)
logits = pred_output.predictions

#print(logits)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  
probs = sigmoid(logits[:, 1])

#print(probs)

sub = pd.DataFrame()
sub['id'] = test_df['id']
sub['generated'] = probs
sub.to_csv(OUTPUT_DIR/'submission.csv', index=False)
#sub.head()