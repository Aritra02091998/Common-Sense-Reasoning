import json
import os
import torch
import random

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


# Performance metrics Code

def calculateAccuracy():

    model.eval()
    matchScore, loopCounter = 0,0

    for index in range(0,500):
        
        loopCounter += 1
        
        example = val_dataset_object[index]
        example = {k: v.unsqueeze(0).to(device) for k,v in example.items()}
        outputs = model(**example)

        logits = outputs.logits
        predicted_classes = torch.sigmoid(logits)
        ans = reverse_mapping[torch.argmax(predicted_classes).item()]
        
        #print(f'T: {answerList_val[index]} <-> P: {ans}' )

        # accuracy score
        
        if answerList_val[index] == ans:
            matchScore += 1
                
    # print(matchScore, loopCounter)
    return ((matchScore/loopCounter)*100)

#---------------------------------------------------------------------------------------------------------------------------------------------

print('\nLoading JSON Data')
train_file_path = '/home/aritra/cric/train_questions.json'
val_file_path = '/home/aritra/cric/val_questions.json'


# Training Set

with open(train_file_path, "r") as file:
     train_json = json.load(file)


# Validation Set

with open(val_file_path, "r") as file:
     val_json = json.load(file)

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

print('\nExtracting Validation Data')

questionList_val = []
answerList_val = []
imgList_val = []

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
    
    questionList_val.append(pointer['question'])
    answerList_val.append(pointer['answer'])
    imgList_val.append(pointer['image_id'])

print('The size of the Val Set is: ',len(questionList_val))

questionList_val = questionList_val[0:2000]
answerList_val = answerList_val[0:2000]
imgList_val = imgList_val[0:2000]

print('The size of the Val Set reduced to: ',len(questionList_val))

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

train_dataloader = DataLoader(train_dataset_object, collate_fn = collate_fn, shuffle = True, batch_size = 16)

tot_number_of_steps = len(train_dataloader)

print('\nTotal number of steps: ',tot_number_of_steps)

#----------------------------------------------------------------------------------------------------------------------------------------

print('\nFinetuning Begins')

scaler = GradScaler()
writer = SummaryWriter()
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)

model.train()

for epoch in tqdm(range(8)):  

    print(f"\nEpoch: {epoch}\n")

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
        
        if (idx % 100 == 0):
            
            acc_score = calculateAccuracy()
            writer.add_scalar('Training Loss', loss.item(), epoch * tot_number_of_steps + idx)
            writer.add_scalar('Accuracy Score On Validation', acc_score, epoch * tot_number_of_steps + idx)
            model.train()

writer.close()


