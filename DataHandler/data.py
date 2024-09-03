import pandas as pd
# from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
# from torch.utils.data import Dataset, DataLoader
# import torch
from .preprocessing import *



def get_validation(val_df):
    test1 = pd.DataFrame()
    test2 = pd.DataFrame()
    test1['text'] = val_df['less_toxic'].copy()
    test2['text'] = val_df['more_toxic'].copy()
    test1['label'] = 0
    test2['label'] = 0
    return test1,test2


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, params):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = params['max_length']
        
    def __getitem__(self, index):
        title = str(self.data.text[index])
        title = " ".join(title.split())
        title = clean_text(title,remove_stopwords=False, stem_words=False, count_null_words=True, clean_wiki_tokens=True)
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids'] 
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.float32)
        } 
    
    def __len__(self):
        return self.len