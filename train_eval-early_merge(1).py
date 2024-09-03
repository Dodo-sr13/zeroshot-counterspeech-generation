import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from DataHandler.data import  *
from DataHandler.mapping import  *
from ModelCode.modelcopy2 import *
from apiconfig import project_name,api_token
import neptune.new as neptune
import GPUtil
import argparse
import json
import random
import os
import time
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import scipy

model_memory=5
total_memory=16



def get_gpu(gpu_id):
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 2, maxLoad = 1.0, maxMemory = (1-(model_memory/total_memory)), includeNan=False, excludeID=[], excludeUUID=[])
        for i in range(len(tempID)):
            if len(tempID) > 0 and (tempID[i]==gpu_id):
                print("Found a gpu")
                print('We will use the GPU:',tempID[i],torch.cuda.get_device_name(tempID[i]))
                deviceID=[tempID[i]]
                return deviceID
            else:
                time.sleep(5)
                
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    

def fix_the_random(seed_val = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def get_label(n):
    label_dict = {0.000000:0, 0.777777:1, 0.611111:2, 0.666666:3, 
                  0.555555:4, 0.740740:5, 0.750000:6, 0.722222:7,
                  0.703703:8, 0.888888:9, 1.000000:10, 0.833333:11,
                  0.814814:12, 0.944444:13, 0.851851:14}
    return label_dict[n]    
# dict_mapping_kaggle={
#     'severe_toxic':5,
#     'obscene':1,
#     'threat':4,
#     'insult':2,
#     'identity_hate':3
# }
def get_dictionary(params):
    
    if(params['dataset']=='toxic_comment'):
        dict_mapping={
            'severe_toxic':9,
            'obscene':5,
            'threat':8,
            'insult':6,
            'identity_hate':7
        }
        
    elif(params['dataset']=='toxic_unintended'):
        dict_mapping={
            'severe_toxic':9,
            'obscene':5,
            'threat':8,
            'insult':6,
            'identity_hate':7
        }
    
    
    
    return dict_mapping

def save_detection_model(model,tokenizer,params):
    if len(params['model_path'].split('/'))>1:
        params['model_path']=params['model_path'].split('/')[1]
    
    output_dir = params['save_path']+params['model_path']+'_'+params['dataset']
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(params, os.path.join(output_dir, "training_args.bin"))

def save_predictions(output1,output2,val_df):
    if len(params['model_path'].split('/'))>1:
        params['model_path']=params['model_path'].split('/')[1]
    
    output_dir = 'Predicitions/'+params['model_path']+'_'+params['dataset']+'/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving predictions to %s" % output_dir)
    
    out1 =[]
    for out in output1:
        out1.append(out.cpu().detach().numpy())

    predict1 = np.array(out1).reshape(len(out1))

    out2 =[]
    for out in output2:
        out2.append(out.cpu().detach().numpy())

    predict2 = np.array(out2).reshape(len(out2))
    
    validation_score = pd.DataFrame({'worker_id': val_df['worker'], 'less_toxic':val_df['less_toxic'],'less_toxic_score':predict1, 'more_toxic':val_df['more_toxic'],'more_toxic_score':predict2})
    validation_score.to_csv(output_dir+'validation_score.csv',index=False)

    
def evaluate_accuracy(output1,output2):
    out1 =[]
    for out in output1:
        out1.append(out.cpu().detach().numpy())

    predict1 = np.array(out1).reshape(len(out1))

    out2 =[]
    for out in output2:
        out2.append(out.cpu().detach().numpy())

    predict2 = np.array(out2).reshape(len(out2))
    accuracy=np.round((predict1 < predict2).mean() * 100,2)
    return accuracy

def predict(model, dataloader, device):
    tokenizer = AutoTokenizer.from_pretrained(params['model_path'],use_fast=False, cache_dir=params['cache_path'])
    
    predicted_label = []
    actual_label = []
    model.eval()
    with torch.no_grad():
        for step,data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
            input_ids = data['ids'].to(device, dtype = torch.long)
            attention_mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float32)
            targets = targets.unsqueeze(1)
            
            ###################################################
            
            tf_vect = tokenizer.batch_decode(input_ids, skip_special_tokens = True)    
            tf_vect = torch.from_numpy((tfidf.transform(tf_vect)).T.todense())
            tf_vect = tf_vect.to(device, dtype = torch.float32)

#             input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask,  tf_vect=tf_vect)
                        
            predicted_label += output[0]
            actual_label += targets
            
    return predicted_label

    
def train(params,run, device):
    if(run!=None):
        run["sys/tags"].add('baseline model')
    tokenizer = AutoTokenizer.from_pretrained(params['model_path'],use_fast=False, cache_dir=params['cache_path'])
    dict_mapping={}
    
    ##### JIGSAW TOXIC COMMENT
    if(params['dataset']=='toxic_comment-early_merge'):
        df_train = pd.read_csv("Dataset/jigsaw-toxic-comment-classification-challenge/train.csv")
        df_test = pd.read_csv("Dataset/jigsaw-toxic-comment-classification-challenge/test.csv")
        df_test_labels = pd.read_csv("Dataset/jigsaw-toxic-comment-classification-challenge/test_labels.csv")
        df_test=df_test.merge(df_test_labels, how='inner', on='id')
        df_total=pd.concat([df_train,df_test]).reset_index(drop=True)
        df_total=df_total[df_total['toxic']!=-1]

        df_total = df_total.rename(columns={"comment_text": "text"})
        
#         dict_mapping=get_dictionary(params)
             
#         list_labels=convert_kaggle(df_total,dict_mapping)
#         df_total['label']=list_labels
        
    
#       ### uncomment for using summed mapping
        df_total['severe_toxic'] = df_total.severe_toxic * 2
        df_total['label'] = df_total.iloc[:, 2:9].sum(axis = 1)
        df_total['label']  = df_total['label']/df_total['label'].max()
        
        
#         df_total = df_total.head(1000)
        
        
        
        
#         ### uncomment the line below if we need to balance the dataset
#         df_toxic=df_total[df_total['toxic']==1]
#         df_non_toxic=df_total[df_total['toxic']==0].sample(n = len(df_toxic))
#         df_total=pd.concat([df_toxic, df_non_toxic]).reset_index(drop=True)

#         ### uncomment to add Weighted Random Sampler
#         df_total['n'] = [get_label(np.trunc(1000000*(x))/1000000) for i, x in enumerate(df_total['label'])]
#         class_sample_count = np.array(df_total['label'].value_counts())
#         weight = 1. / class_sample_count
#         samples_weight = np.array([weight[t] for t in df_total['n']])
#         samples_weight = torch.from_numpy(samples_weight)
#         sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        
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

        
        
        
    params['dict_mapping']=dict_mapping
    if(params['logging']=='neptune'):    
        run["parameters"] = params
        
    ## validation dataset
    val_df = pd.read_csv("Dataset/jigsaw-toxic-severity-rating/validation_data.csv")
    val_df.drop_duplicates(subset=['less_toxic', 'more_toxic'], keep='first', inplace=True)
    val_df.reset_index(inplace=True)
    
#     val_df = val_df.head(1000)


    ### Creating the datasets
    train_dataset = df_total.reset_index(drop=True)
    test1,test2 = get_validation(val_df)
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    training_set = Triage(train_dataset, tokenizer, params)
    print("VAL Dataset: {}".format(val_df.shape))
    testing_set1 = Triage(test1, tokenizer, params)
    testing_set2 = Triage(test2, tokenizer, params)

    
    #### parameters
    train_params = {'batch_size': params['train_batch_size'],
                    'shuffle': True,
                    'num_workers': 4
                   }
    test_params = {'batch_size': params['val_batch_size'],
                   'shuffle': False,
                   'num_workers': 4
                  }

#     training_loader = DataLoader(training_set, **train_params, sampler=sampler)
    training_loader = DataLoader(training_set, **train_params)
    test_loader1 = DataLoader(testing_set1, **test_params)
    test_loader2 = DataLoader(testing_set2, **test_params)
    
    if('roberta' in params['model_path']):
        model = RobertaForRegression.from_pretrained(
                params['model_path'], # Use the 12-layer BERT model, with an uncased vocab.
                cache_dir=params['cache_path'],
                params=params).to(device)
        
    elif('Hate-speech-CNERG/dehatebert-mono-english' in params['model_path']):
        model = HateAlert.from_pretrained(
                params['model_path'], 
                cache_dir=params['cache_path'],
                params=params).to(device)

    else:
        model = BertForRegression.from_pretrained(
                params['model_path'], # Use the 12-layer BERT model, with an uncased vocab.
                cache_dir=params['cache_path'],
                params=params).to(device)

    
    optimizer = AdamW(model.parameters(),
                  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = params['epsilon'], # args.adam_epsilon  - default is 1e-8.
                  weight_decay=params['weight_decay']
                )


    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(training_loader) * params['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps/10),num_training_steps = total_steps)
    
    best_acc=0   
    
    # Train the pipeline
    tfidf.fit_transform(df_total['text'], df_total['label'])
    
    for epoch_i in range(0, params['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        print('Training...')
        total_loss = 0
        model.train()

        for step,data in tqdm(enumerate(training_loader, 0), total=len(training_loader)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float32)
            targets = targets.unsqueeze(1)

            #####################################################################
            tf_vect = tokenizer.batch_decode(ids, skip_special_tokens = True)    
            tf_vect = torch.from_numpy((tfidf.transform(tf_vect)).T.todense())
            tf_vect = tf_vect.to(device, dtype = torch.float32)
            
            model.zero_grad()    
            ##################3
            outputs = model(input_ids=ids, attention_mask=mask,labels=targets, tf_vect=tf_vect)
            loss = outputs[0]
            
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            
                
            total_loss += loss.item()
            
            if(params['logging']=='neptune'):
                run['train/loss'].log(loss.item())
          
          
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
            
            
        
        output1 = predict(model, test_loader1, device)
        output2 = predict(model, test_loader2, device)
        acc=evaluate_accuracy(output1,output2)
        
        if(params['logging']=='neptune'):
            run['val/accuracy'].log(acc)
        else:
            print(f"Validation accuracy: {acc}")
        
        
        avg_train_loss = total_loss / len(training_loader)
        print('avg_train_loss',avg_train_loss)
        
        if(acc>=best_acc):
            best_acc=acc
            if(params['logging']=='neptune'):
                run['val/best_val_acc']=best_acc
            else:
                print(f"Validation best accuracy: {acc}")
            ### only save the model when finalising don't save for every run.
            save_detection_model(model,tokenizer,params)
            ### add the code to generate the validation.csv
            save_predictions(output1,output2,val_df)
            
            
            
    del model
    torch.cuda.empty_cache()
    return 1


##### Models ######
# cardiffnlp/twitter-roberta-base-hate
# GroNLP/hateBERT


### Datasets ###
# worst 
# implicit hate 

params={
  'dataset':'toxic_comment-early_merge',
  'model':'cardiffnlp/twitter-roberta-base-hate',
  'features':'tfidf',
  'cache_path':'../../Saved_models/',
  'model_path':'cardiffnlp/twitter-roberta-base-hate',
  'train_batch_size':32,
  'val_batch_size':32,
  'max_length':256,
  'learning_rate':5e-5,  ### learning rate 2e-5 for bert 0.001 for gru
  'weight_decay':1e-5,
  'epsilon':1e-8,
  'epochs':3,
  'dropout':0.2,
  'random_seed':2021,
  'device':'cuda',
  'save_path':'Saved_Models/',
  'logging':'local'
}

    

if __name__ == "__main__":
#     my_parser = argparse.ArgumentParser()
#     my_parser.add_argument('path',
#                            metavar='--p',
#                            type=str,
#                            help='The path to json containining the parameters')
    
#     my_parser.add_argument('index',
#                            metavar='--i',
#                            type=int,
#                            help='list id to be used')
    
#     my_parser.add_argument('gpuid',
#                            metavar='--i',
#                            type=int,
#                            help='gpu id to be used')
    
    
    
#     args = my_parser.parse_args()
    
#     with open(args.path,mode='r') as f:
#             params_list = json.load(f)

#     params=params_list[args.index]

    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        ##### You can set the device manually if you have only one gpu
        ##### comment this line if you don't want to manually set the gpu
        #deviceID = get_gpu(args.gpuid)
        deviceID = get_gpu(1)
        torch.cuda.set_device(deviceID[0])
        #### comment this line if you want to manually set the gpu
        #### required parameter is the gpu id
        #torch.cuda.set_device(args.gpuid)

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
        
    fix_the_random(seed_val = params['random_seed'])
    params['logging']='neptune'
    params['epochs']=3
    run=None
    if(params['logging']=='neptune'):
        run = neptune.init(project=project_name,api_token=api_token)
#         run = neptune.init(project=project_name, api_token=api_token, mode="offline")
        run["parameters"] = params
    else:
        pass
    
    tfidf = TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5), max_features = 5000)
    
    train(params,run, device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    