import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from DataHandler.data import  *
from DataHandler.mapping import  *

# import neptune.new as neptune
# from apiconfig import project_name,api_token

import optuna
import warnings

import scipy
from tqdm import tqdm
from tqdm import tqdm_notebook


optuna.logging.set_verbosity(1)
warnings.filterwarnings("ignore")

### write code to import datasets
params={
  'dataset':'toxic_comment',
  'model':'tfidf',
  'cache_path':'../../Saved_models/',
  'model_path':'tfidf',
  'random_seed':2021,
  'save_path':'Saved_Models/',
  'logging':'local',
  
  'tfidf_min_features':100,
  'tfidf_max_features':50000,
  'ridge_min_alpha':1e-2,
  'ridge_max_alpha':1e2,
  'min_df':1,
  'max_df':100,
  'n_trials':200,
  
}


def get_dictionary(params):
    
    if(params['dataset']=='toxic_comment'):
        dict_mapping={
            'severe_toxic':12,
            'obscene':5,
            'threat':8,
            'insult':6,
            'identity_hate':9,
            'toxic':4
        }
        
    elif(params['dataset']=='toxic_unintended'):
        dict_mapping={
            'severe_toxic':12,
            'obscene':5,
            'threat':8,
            'insult':6,
            'identity_hate':9
        }
        
    return dict_mapping


def objective(trial):
    alpha = trial.suggest_float(name="alpha", low=params['ridge_min_alpha'], high=params['ridge_max_alpha'], log=True)
    max_features = trial.suggest_int(name="max_features", low=params['tfidf_min_features'], high=params['tfidf_max_features'], step=100)

    min_df = trial.suggest_int(name="min_df", low=params['min_df'], high=params['max_df'])
    offset = trial.suggest_int(name="offset", low=params['min_df'], high=params['max_df'])    
        
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df= min_df, max_df=min_df+offset, analyzer = 'char_wb', ngram_range = (3,5), max_features = max_features)),
#          ("tfidf", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5), max_features = max_features)),
        ("clf", Ridge(alpha=alpha))]
                       ) 
    # Train the pipeline
    pipeline.fit(train_dataset['text'], train_dataset['label'])

    p1 = pipeline.predict(test1['text'])
    p2 = pipeline.predict(test2['text'])

    acc = np.round((p1 < p2).mean() * 100,2)
    return acc

# def optimize(params,run):
# if(run!=None):
#     run["sys/tags"].add('baseline model')

##### JIGSAW TOXIC COMMENT
if(params['dataset']=='toxic_comment'):
    df_train = pd.read_csv("Dataset/jigsaw-toxic-comment-classification-challenge/train.csv")
    df_test = pd.read_csv("Dataset/jigsaw-toxic-comment-classification-challenge/test.csv")
    df_test_labels = pd.read_csv("Dataset/jigsaw-toxic-comment-classification-challenge/test_labels.csv")
    df_test=df_test.merge(df_test_labels, how='inner', on='id')
    df_total=pd.concat([df_train,df_test]).reset_index(drop=True)
    df_total=df_total[df_total['toxic']!=-1]


    dict_mapping=get_dictionary(params)

    list_labels=convert_kaggle(df_total,dict_mapping)
    df_total['label']=list_labels

    df_total = df_total.rename(columns={"comment_text": "text"})

#     df_total = df_total.head(1000)

#         ###uncomment for using summed mapping
#         df_total['severe_toxic'] = df_total.severe_toxic * 2
#         df_total['label'] = df_total.iloc[:, 2:9].sum(axis = 1)
#         df_total['label']  = df_total['label']/df_total['label'].max()


#         # uncomment the line below if we need to balance the dataset
#         df_toxic=df_total[df_total['toxic']==1]
#         df_non_toxic=df_total[df_total['toxic']==0].sample(n = len(df_toxic))
#         df_total=pd.concat([df_toxic, df_non_toxic]).reset_index(drop=True)

if(params['dataset']=='toxic_unintended'):
    df_total=pd.read_csv('Dataset/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv')
    df_total=df_total[df_total['comment_text'].notna()]
    dict_mapping=get_dictionary(params)
    list_labels=convert_kaggle_unintended(df_total,dict_mapping)
    df_total['label']=list_labels
    df_total = df_total.rename(columns={"comment_text": "text"})
    # uncomment the line below if we need to balance the dataset
    df_toxic=df_total[df_total['toxicity']>0]
    df_non_toxic=df_total[df_total['toxicity']==0].sample(n = len(df_toxic))
    df_total=pd.concat([df_toxic, df_non_toxic]).reset_index(drop=True)
#     df_total = df_total.head(1000)
    
    
if(params['dataset']=='reddit'):
    df_total=pd.read_csv('Dataset/Ruddit_comments.csv')
    df_total=df_total[df_total['comments'].notna()]
    df_total=df_total[df_total['comments']!='[deleted]']
    list_labels=convert_reddit(df_total)
    df_total['label']=list_labels
    df_total = df_total.rename(columns={"comments": "text"})
if(params['dataset']=='davidson'):
    df_total=pd.read_csv('Dataset/Davidson.csv')
    list_labels=convert_davidson(df_total)
    df_total['label']=list_labels
    df_total = df_total.rename(columns={"tweet": "text"})
if(params['dataset']=='founta'):
    df_total=pd.read_csv('Dataset/founta.csv')
    list_labels=convert_founta(df_total)
    df_total['label']=list_labels
if(params['dataset']=='ensemble'):
    df_total=pd.read_csv('Dataset/ensemble_val_preds.csv')
    df_total = df_total.rename(columns={"score": "label"})


# params['dict_mapping']=dict_mapping
# if(params['logging']=='neptune'):    
#     run["parameters"] = params

## validation dataset
val_df = pd.read_csv("Dataset/jigsaw-toxic-severity-rating/validation_data.csv")
val_df.drop_duplicates(subset=['less_toxic', 'more_toxic'], keep='first', inplace=True)
val_df.reset_index(inplace=True)

# val_df = val_df.head(1000)

### Creating the datasets
train_dataset = df_total.reset_index(drop=True)
test1,test2 = get_validation(val_df)
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VAL Dataset: {}".format(val_df.shape))
# test1 = [clean_text(text) for text in test1]
# test2 = [clean_text(text) for text in test2]



if __name__ == "__main__":

       
    # fix_the_random(seed_val = params['random_seed'])
    params['logging']='local'
    run=None
    if(params['logging']=='neptune'):
        run = neptune.init(project=project_name,api_token=api_token)
#         run = neptune.init(project=project_name, api_token=api_token, mode="offline")
        run["parameters"] = params
    else:
        pass
    
#     optimize(params,run)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=params['n_trials'], show_progress_bar=True)


    print(f"Best Value: {study.best_trial.value}")
    print(f"Best Params: {study.best_params}")



