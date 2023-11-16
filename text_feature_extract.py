import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import transformers
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, XLMRobertaModel


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TextEncoder:
    def __init__(self, model='mBERT', maxlen_title=50, maxlen_ovr=500):
        # self.maxlen_title = maxlen_title
        # self.maxlen_ovr = maxlen_ovr

        if model == 'mBERT':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        elif model == 'LaBSE':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
            self.model = AutoModel.from_pretrained('sentence-transformers/LaBSE')
        elif model == 'XLM-RoBERTa':
            self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
            self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.model.to(device)


    def encode(self, df):
        tokenized_title = self.tokenizer(df.title.values.tolist(), padding=True, truncation=True, return_tensors="pt")
        tokenized_ovr = self.tokenizer(df.overview.values.tolist(), padding=True, truncation=True, return_tensors="pt")
        # tokenized_title = self.tokenizer(df.title.values.tolist(), max_length=self.maxlen_title, padding='max_length', truncation=True, return_tensors="pt")
        # tokenized_ovr = self.tokenizer(df.overview.values.tolist(), max_length=self.maxlen_ovr, padding='max_length', truncation=True, return_tensors="pt")
        tokenized_title = {k:v.to(device) for k,v in tokenized_title.items()}
        tokenized_ovr = {k:v.to(device) for k,v in tokenized_ovr.items()}
        with torch.no_grad():
            hidden_title = self.model(**tokenized_title) 
            hidden_ovr = self.model(**tokenized_ovr)

        # Get only the [CLS] hidden states
        cls_title = hidden_title.last_hidden_state[:,0,:]
        cls_ovr = hidden_ovr.last_hidden_state[:,0,:]

        # Concatenate
        cls = torch.cat([cls_title, cls_ovr], dim=1)

        return cls

def main(params):
    df = pd.read_csv('movies.csv')
    encoder = TextEncoder(params.model)
    batch_size = 1000
    os.makedirs(params.save_path, exist_ok=True)
    for i in range(0, len(df), batch_size):
        df_batch = df.iloc[i:i+batch_size, :]
        Xfeatures = encoder.encode(df_batch).cpu().numpy()
        features_file_path = params.save_path + f'/{i}_features.npy'
        with open(features_file_path, 'wb') as features_file:
            np.save(features_file, Xfeatures)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--model", type=str, default='mBERT')
    parser.add_argument("--save_path", type=str, default='text_features_mBERT')

    params, unknown = parser.parse_known_args()
    main(params)