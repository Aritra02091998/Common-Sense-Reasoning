import json
import os
import torch
import sys

import datasets
from tqdm.auto import tqdm

from datasets import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader


def generateCaptionUsingVitGpt(image):

  pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = captioningModel.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds[0]


# this function creates a uniques list of all the labels(answers) which was using a randomised set for this purpose

def findUnique(targetList):
    
    uniqueList = []
    
    for word in targetList:
        if word not in uniqueList:
            uniqueList.append(word)
    
    return uniqueList

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

with open(test_file_path, "r") as file:
     test_json = json.load(file)

#---------------------------------------------------------------------------------------------------------------------------------------------

print('\nExtracting Data of Training Set')

questionList = []
answerList = []
imgList = []

print('\nExcluding Erroneous Indices\n')

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

#---------------------------------------------------------------------------------------------------------------------------------------------

print('\nCreating word to number mapping')

mapping = {}
counter = 0

uniqueAnsList = findUnique(answerList)

for word in uniqueAnsList:
    
    if word not in mapping:
        
        mapping[word] = counter
        counter += 1


numOfClasses = max(mapping.values())

print('\nCreating number to word mapping')
print('\nLength of the unique labels is: ',len(uniqueAnsList))
print('\nLength of the mapping created is: ',len(mapping))

reverse_mapping = dict([(value, key) for key, value in mapping.items()])

#---------------------------------------------------------------------------------------------------------------------------------------------

print('\nProcessing Training Set\n')

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

print('\nExtracting Validation Set')

questionList_val = []
answerList_val = []
imgList_val = []

print('\nExcluding Erroneous Indices')

# collecting the index containing errorneous images

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

print('The size of the val Set is: ',len(questionList_val))

uniqueAnswerListVal = list(set(answerList_val))

#-----------------------------------------------------------------------------------------------------------------------------------------

print('\nProcessing Validation Data\n')

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

print('\nExtracting Test Data Set')

questionList_test = []
answerList_test = []
imgList_test = []


print('\nExcluding Erroneous Indices')

# collecting the index containing errorneous images


indexToExcludeTest = []

with open('error_testSet1.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExcludeTest.append(number)
        
with open('errorTestSet2.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExcludeTest.append(number)


# excluding the index containing errorneous images

for i in tqdm(range(len(test_json))):
    
    if i in indexToExcludeTest:
        continue
        
    pointer = test_json[i]
    
    questionList_test.append(pointer['question'])
    answerList_test.append(pointer['answer'])
    imgList_test.append(pointer['image_id'])

print('\nSize of the Test set is ',len(questionList_test))

#---------------------------------------------------------------------------------------------------------------------------------------

print('\nProcessing Test Data Set\n')

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

# Importing Image Captioning Model

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

captioningModel = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = "cuda" if torch.cuda.is_available() else "cpu"

captioningModel = captioningModel.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


# Write the generated captions in a text files

print('Image Captioning Starts')

with open('./cricImageCaptions/cricVitGpt2CaptionsTrainSet.txt', 'w') as file:
    
    for imgIndex in tqdm(range(364921)):
        image = modified_train_set[imgIndex]['images']
        caption = generateCaptionUsingVitGpt(image)
        file.write(str(imgIndex) + '_' + str(caption) + '\n')

print('End of Writing')
