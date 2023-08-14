import json
import os
import torch
import random
import sys

import datasets
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torch.utils.tensorboard import SummaryWriter

path = './vilt_output.txt'
sys.stdout = open(path, 'w')


def collate_fn(batch):
  
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
        
    # create padded pixel values and corresponding pixel mask
    
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    
    batch = {}
    
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels, dim = 0 )

    return batch


# This function returns the Validation Loss and accuracy on the Validation Set


# This function returns the Validation Loss and accuracy on the Validation Set

def calculateAccuracyVal():
    
    matchScore, loopCounter = 0,0

    for index in range(0,1500):
        
        loopCounter += 1
        
        val_example = val_dataset_object[index]
        val_example = {k: v.unsqueeze(0).to(device) for k,v in val_example.items()}
        val_outputs = model(**val_example)
        
        validationLoss = val_outputs.loss

        val_logits = val_outputs.logits
        val_predicted_classes = torch.sigmoid(val_logits)
        val_ans = reverse_mapping[torch.argmax(val_predicted_classes).item()]
        
        #print(f'T: {answerList_val[index]} <-> P: {val_ans}' )

        # accuracy score
        
        if answerList_val[index] == val_ans:
            matchScore += 1
                
    #print(matchScore, loopCounter)
    accuracyVal = (matchScore/loopCounter)*100
    return ( accuracyVal,validationLoss.item() )


# This function returns accuracy on the Test Set
# However, accuracy needs to be found out on the whole Test Set using the saved model chkpts

def calculateAccuracyTest():
    
    matchScore, loopCounter = 0,0

    for index in range(0,1800):
        
        loopCounter += 1
        
        test_example = test_dataset_object[index]
        test_example = {k: v.unsqueeze(0).to(device) for k,v in test_example.items()}
        test_outputs = model(**test_example)

        test_logits = test_outputs.logits
        test_predicted_classes = torch.sigmoid(test_logits)
        test_ans = reverse_mapping[torch.argmax(test_predicted_classes).item()]
        
        #print(f'T: {answerList_val[index]} <-> P: {test_ans}' )

        # accuracy score
        
        if answerList_test[index] == test_ans:
            matchScore += 1
                
    #print(matchScore, loopCounter)
    return ((matchScore/loopCounter)*100)

#---------------------------------------------------------------------------------------------------------------------------------------------

print('\nLoading JSON Data')

train_file_path = '/home/aritra/cric/train_questions.json'
val_file_path = '/home/aritra/cric/val_questions.json'
test_file_path = '/home/aritra/cric/test_v1_questions.json'

# Training Set

with open(train_file_path, "r") as file:
     train_json = json.load(file)


# Validation Set

with open(val_file_path, "r") as file:
     val_json = json.load(file)


# Test Set

with open(val_file_path, "r") as file:
     test_json = json.load(file)

#---------------------------------------------------------------------------------------------------------------------------------------------

print('\nExtracting Data of Training Set')

questionList = []
answerList = []
imgList = []

print('\nExcluding Erroneous Indices')

# verifying
indexToExclude = []

with open('error1.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)
        
with open('error2.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)
        
with open('error3.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)


for i in tqdm(range(len(train_json))):
    
    if i in indexToExclude:
        continue
        
    pointer = train_json[i]
    
    questionList.append(pointer['question'])
    answerList.append(pointer['answer'])
    imgList.append(pointer['image_id'])

print('\nSize of the training set is: ',len(questionList))

questionList = questionList[0:100000]
answerList = answerList[0:100000]
imgList = imgList[0:100000]

print('\nSize of the training set reduced to: ',len(questionList))

#---------------------------------------------------------------------------------------------------------------------------------------------

print('\nCreating word to number mapping')

mapping = {}
counter = 0

uniqueAnsList = list(set(answerList))

for word in uniqueAnsList:
    
    if word not in mapping:
        
        mapping[word] = counter
        counter += 1


numOfClasses = max(mapping.values())

print('\nCreating number to word mapping')

reverse_mapping = dict([(value, key) for key, value in mapping.items()])

#---------------------------------------------------------------------------------------------------------------------------------------------

print('\nProcessing Training Set')

labels = []

for i in range(len(answerList)):
    labels.append( mapping[ answerList[i] ] )


scores = []

for i in tqdm(range(len(answerList))):
    
    s = [0] * (numOfClasses+1)
    s[ mapping[ answerList[i]] ] = 1
    
    scores.append(s)


imgPathList = []
filepath = '/home/aritra/cric/images/img/'

for i in tqdm(range(len(imgList))):
    
    imgName = str(imgList[i]) + '.jpg'
    concatedPath = os.path.join(filepath,imgName)
    
    imgPathList.append(concatedPath)


listToDictionary = {'questions':questionList, 'labels': labels, 'scores': scores, 'images':imgPathList}
modified_train_set = Dataset.from_dict(listToDictionary)


# mapping each filepath to images in the directory

modified_train_set = modified_train_set.cast_column("images", datasets.Image())

#-----------------------------------------------------------------------------------------------------------------------------------------

print('\nExtracting Validation Set into a Combined Set to split into val and test set')

questionList_combined = []
answerList_combined = []
imgList_combined = []

# collecting the index containing errorneous images

print('\nExcluding Erroneous Indices')

indexToExcludeVal = []
with open('error_validation.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExcludeVal.append(number)

with open('error_validation2.txt', 'r') as file:
    for line in file:
        number = int(line.strip())  # Convert the read line to an integer
        indexToExcludeVal.append(number)


# excluding the index containing errorneous images

for i in tqdm(range(len(val_json))):
    
    if (i in indexToExcludeVal):
        continue
        
    pointer = val_json[i]
    
    questionList_combined.append(pointer['question'])
    answerList_combined.append(pointer['answer'])
    imgList_combined.append(pointer['image_id'])

print('The size of the Combined Set is: ',len(questionList_combined))

'''
In the CRIC dataset, the validation and the test set are same, so splitting the 33170 val set into two parts that is 

1. 16585 data points -> Val Set
2. 16585 data points -> Test Set

'''

# Creating Splitted Val Set from Validation Set

questionList_val = questionList_combined[0:16585]
answerList_val = answerList_combined[0:16585]
imgList_val = imgList_combined[0:16585]

print('The size of the Val Set is: ',len(questionList_val))


# Creating Splitted Test Set from Validation Set

questionList_test = questionList_combined[16586:]
answerList_test = answerList_combined[16586:]
imgList_test = imgList_combined[16586:]

print('The size of the test Set reduced is: ',len(questionList_test))

uniqueAnswerListVal = list(set(answerList_val))

#-----------------------------------------------------------------------------------------------------------------------------------------

print('\nProcessing Validation Data')

labels_val = []

for i in range(len(answerList_val)):
    labels_val.append( mapping[ answerList_val[i] ] )


scores_val = []

for i in tqdm(range(len(answerList_val))):
    
    s = [0] * (numOfClasses+1)
    s[ mapping[ answerList_val[i]] ] = 1
    
    scores_val.append(s)


imgPathList_val = []
filepath = '/home/aritra/cric/images/img/'

for i in tqdm(range(len(imgList_val))):
    
    imgName = str(imgList_val[i]) + '.jpg'
    concatedPath = os.path.join(filepath,imgName)
    
    imgPathList_val.append(concatedPath)


# creating HF dataset to map images fast of Val_set

listToDictionary = {'questions':questionList_val, 'labels':labels_val, 'scores':scores_val, 'images':imgPathList_val}
modified_val_set = Dataset.from_dict(listToDictionary)

# mapping each filepath of Val Set to images in the directory

modified_val_set = modified_val_set.cast_column("images", datasets.Image())


#---------------------------------------------------------------------------------------------------------------------------------------


print('\nProcessing Test Data Set')

labels_test = []

for i in range(len(answerList_test)):
    labels_test.append( mapping[ answerList_test[i] ] )

scores_test = []

for i in tqdm(range(len(answerList_test))):
    
    s = [0] * (numOfClasses+1)
    s[ mapping[ answerList_test[i]] ] = 1
    
    scores_test.append(s)


imgPathList_test = []
filepath = '/home/aritra/cric/images/img/'

for i in tqdm(range(len(imgList_test))):
    
    imgName = str(imgList_test[i]) + '.jpg'
    concatedPath = os.path.join(filepath,imgName)
    
    imgPathList_test.append(concatedPath)


# creating HF dataset to map images fast of test_set

listToDictionary = {'questions':questionList_test, 'labels':labels_test, 'scores':scores_test, 'images':imgPathList_test}
modified_test_set = Dataset.from_dict(listToDictionary)

# mapping each filepath of test Set to images in the directory

modified_test_set = modified_test_set.cast_column("images", datasets.Image())


print('\nEnd of all pre Processing')

#-----------------------------------------------------------------------------------------------------------------------------------------

print('\nImporting Transformer')

from transformers import ViltProcessor, ViltForQuestionAnswering

from transformers import ViltConfig
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", id2label = reverse_mapping, label2id = mapping).to(device)

#-----------------------------------------------------------------------------------------------------------------------------------------

print('\nClass Declaration')

class cric_dataset(Dataset):
    
    def __init__(self, dataset, processor):
        self.processor = processor
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        
        #print(idx)
        item = self.dataset[idx]

        #print(item)
        
        encodings = self.processor(images = item["images"], text = item["questions"], padding="max_length", truncation=True, return_tensors = "pt")
        encodings = {k:v.squeeze() for k,v in encodings.items()}
                                
        encodings['labels'] = torch.tensor(item['scores'], dtype = torch.float32)
        
        return encodings

#-----------------------------------------------------------------------------------------------------------------------------------------

train_dataset_object = cric_dataset(modified_train_set, processor)
val_dataset_object = cric_dataset(modified_val_set, processor)
test_dataset_object = cric_dataset(modified_test_set, processor)

train_dataloader = DataLoader(train_dataset_object, collate_fn = collate_fn, shuffle = True, batch_size = 16)

tot_number_of_steps = len(train_dataloader)

print('\nTotal number of steps: ',tot_number_of_steps)

#----------------------------------------------------------------------------------------------------------------------------------------

print('\nFinetuning Begins')

scaler = GradScaler()
writer = SummaryWriter()
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)

model.train()

for epoch in tqdm(range(10)):  

    print(f"Epoch: {epoch}")

    for idx, batch in enumerate(train_dataloader):

        batch = {k:v.to(device) for k,v in batch.items()}

        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            outputs = model(**batch)
            loss = outputs.loss

        print(idx,"-> Loss:", loss.item())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        scaler.update()
        
        # Plots in tensorboard
    
        if (idx % 400 == 0):
            
            acc_score_val, validationLoss = calculateAccuracyVal()
            acc_score_test = calculateAccuracyTest()
            
            print(f'\nValidation Accuracy: {acc_score_val}, Test Accuracy: {acc_score_test} \n')
            
            writer.add_scalar('Training Loss', loss.item(), epoch * tot_number_of_steps + idx)
            writer.add_scalar('Validation Loss', validationLoss, epoch * tot_number_of_steps + idx)

            writer.add_scalar('Accuracy Score On Val Set', acc_score_val, epoch * tot_number_of_steps + idx)
            writer.add_scalar('Accuracy Score On Test Set', acc_score_test, epoch * tot_number_of_steps + idx)
            model.train()
            
    # Save model checkpoint
    
    save_path = os.path.join('./model_chkpts/vilt-mlm-classification-model/', 'vilt_mlm_mod_e' + str(epoch+1) + '_cric_trained')
    model.save_pretrained(save_path)

writer.close()

