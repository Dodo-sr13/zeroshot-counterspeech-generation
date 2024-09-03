import pandas as pd
from tqdm import tqdm
import numpy as np

def convert_kaggle(dataframe,dict_mapping):
    list_label=[]
    maximum=np.max(list(dict_mapping.values()))
    sum_total=np.sum(list(dict_mapping.values()))
    for index,row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        if(row[2:].sum()==0):
            list_label.append(0)
        else:
            final_label=0
            count=0
            for col in list(dataframe.columns)[2:]:
                if(row[col]==1):
                    final_label+=dict_mapping[col]
                    count+=1
            if(final_label==0):
                final_label=sum_total/len(dict_mapping)
            else:
                final_label=final_label/count
            list_label.append(final_label/maximum)
    return list_label



def convert_kaggle_unintended(dataframe,dict_mapping):
    list_label=[]
    for index,row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        list_label.append(row['toxicity'])
    return list_label


def convert_founta(dataframe):
    list_label=[]
    for index,row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        list_label.append(row['label']/2)
    return list_label


def convert_reddit(dataframe):
    list_label=[]
    for index,row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        list_label.append((row['offensiveness_score']+1)/(1+1))
    return list_label



def convert_davidson(dataframe):
    list_label=[]
    for index,row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        score=(row['hate_speech']*3+row['offensive_language']*2+row['neither']*1)/row['count']
        score=score/3
        list_label.append(score)
    return list_label