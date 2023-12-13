import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import transformers
from transformers import BertConfig, BertTokenizer, BertModel, AutoTokenizer, AutoModel, XLMRobertaModel


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
    def __init__(self, model='mBERT', way='merge'):
        self.way = way

        if model == 'mBERT':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        elif model == 'LaBSE':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
            self.model = AutoModel.from_pretrained('sentence-transformers/LaBSE')
        elif model == 'XLM-RoBERTa':
            self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
            self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.model.to(device)


    def encode(self, df):
        if self.way == 'merge':
            tokenized = self.tokenizer((df.clean_title + ' ' + df.clean_overview).values.tolist(), padding=True, truncation=True, return_tensors="pt")
            tokenized = {k:v.to(device) for k,v in tokenized.items()}
            with torch.no_grad():
                hidden = self.model(**tokenized)
            embedding = torch.mean(hidden.last_hidden_state, dim=1)

        elif self.way == 'overview':
            tokenized = self.tokenizer(df.clean_overview.values.tolist(), padding=True, truncation=True, return_tensors="pt")
            tokenized = {k:v.to(device) for k,v in tokenized.items()}
            with torch.no_grad():
                hidden = self.model(**tokenized)
            embedding = torch.mean(hidden.last_hidden_state, dim=1)

        elif self.way == 'separate':
            tokenized_title = self.tokenizer(df.clean_title.values.tolist(), padding=True, truncation=True, return_tensors="pt")
            tokenized_ovr = self.tokenizer(df.clean_overview.values.tolist(), padding=True, truncation=True, return_tensors="pt")
            tokenized_title = {k:v.to(device) for k,v in tokenized_title.items()}
            tokenized_ovr = {k:v.to(device) for k,v in tokenized_ovr.items()}
            with torch.no_grad():
                hidden_title = self.model(**tokenized_title) 
                hidden_ovr = self.model(**tokenized_ovr)
            embedding_title = torch.mean(hidden_title.last_hidden_state, dim=1)
            embedding_ovr = torch.mean(hidden_ovr.last_hidden_state, dim=1)
            embedding = torch.cat([embedding_title, embedding_ovr], dim=1)
    
        return embedding

def main(params):
    df = pd.read_csv(params.df)
    encoder = TextEncoder(params.model, params.way)
    features = []
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        df_batch = df.iloc[i:i+batch_size, :]
        features_batch = encoder.encode(df_batch).cpu().numpy()
        features.append(features_batch)
    Xfeatures = np.concatenate(features)
    np.save(params.save_path + '.npy', Xfeatures)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text Feature Extraction")

    parser.add_argument("--df", type=str, default='movies.csv')
    parser.add_argument("--model", type=str, default='mBERT')
    parser.add_argument("--way", type=str, default='overview')
    parser.add_argument("--save_path", type=str, default='text_features_mBERT')

    params, unknown = parser.parse_known_args()
    main(params)