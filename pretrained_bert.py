import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, SequentialSampler, RandomSampler
from tqdm import tqdm

from transformers import AutoTokenizer,BertTokenizer#, AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification

# Set seed
seed = 42
torch.manual_seed(seed)


#load data
train_essay = pd.read_csv('dataset/train_essays.csv')
train_prompt = pd.read_csv('dataset/train_prompts.csv')
extra_dataset = pd.read_csv("dataset/concatenated.csv")
extra_dataset1 = pd.read_csv('dataset/Mistral7B_CME_v7.csv')
test_essay = pd.read_csv("dataset/test_essays.csv")


train_essay.rename(columns={'generated': 'label'}, inplace=True)
train_essay.reset_index(drop=True)
train_essay = train_essay[["text", "label"]]

extra_dataset.rename(columns={'generated': 'label'}, inplace=True)
extra_dataset.reset_index(drop=True)
extra_dataset = extra_dataset[["text", "label"]]

extra_dataset1.rename(columns={'generated': 'label'}, inplace=True)
extra_dataset1.reset_index(drop=True)
extra_dataset1 = extra_dataset1[["text", "label"]]

train_dataset = pd.concat([train_essay, extra_dataset, extra_dataset1])
train_dataset.value_counts('label')

#Tokenization and Encoding 
model_name = "bert-base-cased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def tokenize_text(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

encoded_data = train_dataset['text'].apply(tokenize_text)

input_ids = torch.cat([x['input_ids'] for x in encoded_data], dim=0)
attention_masks = torch.cat([x['attention_mask'] for x in encoded_data], dim=0)
labels = torch.tensor(train_dataset['label'].values)

#split train/test data
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids, labels, random_state=seed, test_size=0.2
)

train_masks, val_masks, _, _ = train_test_split(
    attention_masks, labels, random_state=seed, test_size=0.2
)
#create dataloaders
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


#define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# do train
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}', unit='batches'):
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}

        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='Validation', unit='batches'):
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[2].to(device)}

            outputs = model(**inputs)
            loss = outputs.loss
            val_loss += loss.item()
            logits = outputs.logits

    avg_val_loss = val_loss / len(val_dataloader)

    print(f'Epoch {epoch + 1}:')
    print(f'  Training Loss: {avg_train_loss}')
    print(f'  Validation Loss: {avg_val_loss}')

 #do predict
encoded_test = test_essay['text'].apply(tokenize_text)
test_inputs = torch.cat([x['input_ids'] for x in encoded_test], dim=0)
test_masks = torch.cat([x['attention_mask'] for x in encoded_test], dim=0)


test_data = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model.eval()
preds_test = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Test Prediction', unit='batches'):
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device)}

        outputs = model(**inputs)
        logits = outputs.logits
        preds_test.extend(logits[:, 1].cpu().numpy())

threshold = 0.5
test_essay['id'] = test_essay['id'].astype(str)
preds_test_np = np.array(preds_test)

submission_df = pd.DataFrame({
    'id': test_essay['id'],
    'generated': (preds_test_np > threshold).astype(int)
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
