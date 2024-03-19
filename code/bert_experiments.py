import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification,RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import re
from hunspell import Hunspell
from bert_experiment_setup import eval_metrics

import pandas as pd
import numpy as np

from tqdm import trange
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def experiment_run(train_path,test_path,file_name,best_file_name,seeds,model_type,model_name,epochs,clean = True, spelling = False,  val_path = None, dict = False):
    
    if dict:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        if clean:
            train['Comment'] = train['Comment'].map(lambda x: clean_text(x, for_embedding=True) if isinstance(x, str) else x)
            test['Comment'] = test['Comment'].map(lambda x: clean_text(x, for_embedding=True) if isinstance(x, str) else x)

        train_df = train
    else:
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        if clean:
            train['Comment'] = train['Comment'].map(lambda x: clean_text(x, for_embedding=True) if isinstance(x, str) else x)
            val['Comment'] = train['Comment'].map(lambda x: clean_text(x, for_embedding=True) if isinstance(x, str) else x)
            test['Comment'] = test['Comment'].map(lambda x: clean_text(x, for_embedding=True) if isinstance(x, str) else x)

        train_df = pd.concat([train,val])

    train_text = train_df.Comment.values
    labels = train_df.Label.values

    test_text = test.Comment.values
    test_labels = test.Label.values

    if spelling:
        train_text = spell_correct(train_text)
        test_text = spell_correct(test_text)

    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    count_unk(train_text,tokenizer)
    count_unk(test_text,tokenizer)

    token_id = []
    attention_masks = []

    for sample in train_text:
        encoding_dict = preprocessing(sample, tokenizer)
        token_id.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])

    test_ids = []
    test_attention_mask = []
    for sample in test_text:
        encoding_dict_test = preprocessing(sample, tokenizer)
        test_ids.append(encoding_dict_test['input_ids']) 
        test_attention_mask.append(encoding_dict_test['attention_mask'])

    token_id = torch.cat(token_id, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    labels = torch.tensor(labels)

    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)
    test_labels = torch.tensor(test_labels)
    test_posts = torch.tensor(test['ID_Posting'])
    test_set = TensorDataset(test_ids,test_attention_mask,test_labels, test_posts)
    test_loader = DataLoader(test_set, sampler = RandomSampler(test_set), batch_size = 16)

    if dict:
        val_ratio = 0.2
        train_idx, val_idx = train_test_split(np.arange(len(labels)),test_size = val_ratio,shuffle = True,stratify = labels, random_state=1337)

        # Train and validation sets
        train_set = TensorDataset(token_id[train_idx], attention_masks[train_idx], labels[train_idx])

        val_set = TensorDataset(token_id[val_idx], attention_masks[val_idx], labels[val_idx])

    else:
        # Train and validation sets
        train_set = TensorDataset(token_id[:len(train)], attention_masks[:len(train)], labels[:len(train)])
        val_set = TensorDataset(token_id[len(train):], attention_masks[len(train):], labels[len(train):])
    # Prepare DataLoader
    train_dataloader = DataLoader(train_set,sampler = RandomSampler(train_set),batch_size = 16)

    validation_dataloader = DataLoader(val_set,sampler = SequentialSampler(val_set),batch_size = 16)
    global_best_score = -1
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_F1s = []
    for seed_val in seeds:
        best_score = -1
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        print(f'Run for seed {seed_val}')
        if model_type == 'bert':

            #load the BertForSequenceClassification model
            model = BertForSequenceClassification.from_pretrained(model_name,num_labels = 2,output_attentions = False,output_hidden_states = False)

        else:
            model = RobertaForSequenceClassification.from_pretrained(model_name,num_labels = 2,output_attentions = False,output_hidden_states = False)


        #recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
        optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5,eps = 1e-08)

        model.to(device)
       

        for _ in trange(epochs, desc = 'Epoch'):
            
            #set model to training mode
            model.train()
            
            #tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()
                #forward pass
                train_output = model(b_input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = b_input_mask, 
                                    labels = b_labels)
                #backward pass
                train_output.loss.backward()
                optimizer.step()
                #update tracking variables
                tr_loss += train_output.loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            #set model to evaluation mode
            model.eval()

            #tracking variables 
            val_accuracy = []
            val_precision = []
            val_recall = []
            val_f1 = []

            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                #forward pass
                    eval_output = model(b_input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                #calculate validation metrics
                b_accuracy, b_precision, b_recall, b_f1 = b_metrics(logits, label_ids)
                val_accuracy.append(b_accuracy)
                #update precision only when (tp + fp) !=0; ignore nan
                if b_precision != 'nan': val_precision.append(b_precision)
                #update recall only when (tp + fn) !=0; ignore nan
                if b_recall != 'nan': val_recall.append(b_recall)
                #update specificity only when (tn + fp) !=0; ignore nan
                if b_f1 != 'nan': val_f1.append(b_f1)

            print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
            print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
            print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
            print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
            print('\t - Validation F1: {:.4f}\n'.format(sum(val_f1)/len(val_f1)) if len(val_f1)>0 else '\t - Validation F1: NaN')
            if sum(val_f1)/len(val_f1) > best_score:
                torch.save(model.state_dict(), file_name)
                best_score = sum(val_f1)/len(val_f1)
                print('New best model')
        
        if model_type == 'bert':
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        else:
            model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.load_state_dict(torch.load(file_name))
        model.to(device)
        
        
        preds, posts, recall, precision, f1_score, accuracy = predict(model, test_loader)
        if f1_score > global_best_score:
                torch.save(model.state_dict(), best_file_name)
                global_best_score = f1_score
                print('New best global model')
        pred_df = pd.DataFrame(list(zip(posts,preds)),columns = ['ID_Posting', 'Score'])
        eval_test = pred_df.merge(test, on = 'ID_Posting')
        TP = len(eval_test.loc[(eval_test['Score']==eval_test['Label'])&(eval_test['Label']==1)])
        FP = len(eval_test.loc[(eval_test['Score']==1)&(eval_test['Label']==0)])
        TN  = len(eval_test.loc[(eval_test['Score']==eval_test['Label'])&(eval_test['Label']==0)])
        FN = len(eval_test.loc[(eval_test['Score']==0)&(eval_test['Label']==1)])

        print(f'The best performing model of this run found {TP} true positives, {FP} false positives, {TN} true negatives and {FN} false negatives')
        test_accuracies.append(accuracy)
        test_precisions.append(precision)
        test_recalls.append(recall)
        test_F1s.append(f1_score)

    print(f'The mean Accuracy across all runs for this experiments is: {sum(test_accuracies)/len(test_accuracies)}')
    print(f'The mean Precision across all runs for this experiments is: {sum(test_precisions)/len(test_precisions)}')
    print(f'The mean Recall across all runs for this experiments is: {sum(test_recalls)/len(test_recalls)}')
    print(f'The mean F1 Score across all runs for this experiments is: {sum(test_F1s)/len(test_F1s)}')
    
    if model_type == 'bert':
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(torch.load(file_name))
    return model


#remove multiple white spaces, html tags, non ascii cahracters and single character "words"
def clean_text(text):
    white_space = re.compile(r"\s+", re.IGNORECASE)
    tags = re.compile(r"<[^>]+>")
    ascii = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
    single_char = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

    text = re.sub(tags, " ", text)
    text = re.sub(ascii, " ", text)
    text = re.sub(single_char, " ", text)
    text = re.sub(white_space, " ", text)

    word_tokens = word_tokenize(text)
    words_filtered = word_tokens

    text_clean = " ".join(words_filtered)
    return text_clean

#use an Austrian German dictionary for spelling correction
def spell_correct(text):
    dict_path = 'de_AT_frami'
    h = Hunspell("de_AT_frami", hunspell_data_dir=dict_path)
    text_spell =[]
    #try spell correction
    for x in text:
        corrected = []
        for y in x.split():
            if is_latin1(y):    
                if not h.spell(y):
                    if len(h.suggest(y)) >0:
                        y = h.suggest(y)[0]
            corrected.append(y)
        text_spell.append(' '.join(corrected))
    return text_spell

def is_latin1(string):
    for char in string:
        if ord(char) > 255:
            return False
    return True

def count_unk(text,tokenizer):
    #count unknown and split tokens
    count = 0
    for x in text:
        if '[UNK]' in tokenizer.tokenize(x):
            count = count +1
    print(f'Amount of unknown tokens: {count}')
    count2 = 0
    for x in text:
        for y in tokenizer.tokenize(x):
            count2 = count2 + y.count('##')
    print(f'Amount of separated tokens: {count2}')

def preprocessing(input_text, tokenizer):

  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 260,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

#calculate true positives, false positives, true negatives and false negatives
def b_tp(preds, labels):
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

#calculate precision, recall, f1 and accuracy for a set of predictions
def b_metrics(preds, labels):

  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

  b_f1 = 2*((b_precision*b_recall)/(b_precision+b_recall)) if (b_precision+b_recall)> 0 else 0
  return b_accuracy, b_precision, b_recall, b_f1

def predict(model, test_loader):
    model.eval()

    #store predictions and ground truth to calculate evalutation metrics
    predictions , true_labels, posts = [], [], []

    #loop for the prediction on our test set
    for batch in test_loader:
        #send batch to the device and unpack its content
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels,b_posting = batch
        
        #calculate our prediction logits
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]
        #send predictions back to the CPU
        logits = logits.detach().to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()
        preds = np.argmax(logits, axis=1).flatten()
        #add predictions to our lists
        for i in preds:
            predictions.append(i)
        for i in label_ids:
            true_labels.append(i)
        for i in b_posting.detach().to('cpu').numpy():
            posts.append(i)
    recall, precision, f1_score, accuracy = eval_metrics(true_labels, predictions)

    return predictions ,posts, recall, precision, f1_score, accuracy
