from PIL import Image
import datasets
import numpy as np
import torch
import requests
import os

from datasets import load_dataset
from datasets import Dataset
from transformers import BlipProcessor, BlipForQuestionAnswering

from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
bleurt_model.eval()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device Selected:', device)


# loading BLIP VQA model

print('Loading BLIP VQA Model In: ', device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)


# prediction and groundTruth variable are expected to contain strings.

def bleurt_score(prediction, groundTruth):
    references = prediction
    candidates = groundTruth

    with torch.no_grad():
      scores = bleurt_model(**tokenizer(references, candidates, return_tensors='pt'))[0].squeeze()

    return(scores)


def evaluate_validation_loss_on_val_set(small_val_set):
    
    groundTruthList = small_val_set['answers']
    predictedList = []
    bleurt_scores = []
    
    # set the model to inference mode
    
    model.eval()
    
    for i in range(len(small_val_set)):
        
        question = small_val_set[i]['questions']
        image = small_val_set[i]['images']
                
        inputs = processor(image, question, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        ans = processor.decode(outputs[0], skip_special_tokens=True)
        
        # print(f'{question}? -> {ans}')

        predictedList.append(ans)
        
    for i in range(len(predictedList)):
        
        scores = bleurt_score(predictedList[i], groundTruthList[i])
        bleurt_scores.append(scores)
    
    model.train()
    
    return (sum(bleurt_scores)/len(bleurt_scores))

def main():
    
    from datasets import Dataset

    # loading dataset

    print('Loading Dataset')
    ds_visdial = datasets.load_dataset('jxu124/visdial', split="train")


    # removing unnecessary columns

    print('Removing Column')
    ds_visdial = ds_visdial.remove_columns('global_image_id')
    ds_visdial = ds_visdial.remove_columns('anns_id')


    img_list = ds_visdial['image_path']
    dialog = ds_visdial['dialog']
    capList = ds_visdial['caption']


    # Training Set Creation

    print('Training Set Creation')
    questionList = []
    ansList = []
    img_path_list = []
    captionList = []

    for i in range(123287):
        
        dialogSet = dialog[i]
        img = img_list[i]
        caption = capList[i]
        
        for j in range(10):
            question = dialogSet[j][0]
            answer = dialogSet[j][1]
            
            questionList.append(question)
            ansList.append(answer)
            img_path_list.append(img)
            captionList.append(caption)

    listToDictionary = {'caption':captionList, 'questions':questionList, 'answers':ansList, 'images':img_path_list}
    modified_train_set = Dataset.from_dict(listToDictionary)


    # For fast mapping the path to PIL Images

    path_map = {"coco/val2014": "coco/train2014"}

    from dataclasses import dataclass

    @dataclass
    class ReplaceImagePath():
        path_map: {}
        def __call__(self, features):
            for k, v in self.path_map.items():
                features['images'] = features['images'].replace(k, v)
            return features
        
    modified_train_set = modified_train_set.map(ReplaceImagePath(path_map=path_map)).cast_column("images", datasets.Image())
    modified_train_set = modified_train_set.shuffle(seed = 42)


    # Creation of Validation set

    print('Validation Set Creation')
    ds_validation_visdial = datasets.load_dataset('jxu124/visdial', split="validation")


    # Removing Unneessary columns

    ds_validation_visdial = ds_validation_visdial.remove_columns('global_image_id')
    ds_validation_visdial = ds_validation_visdial.remove_columns('anns_id')

    dialog = ds_validation_visdial['dialog']
    img_list = ds_validation_visdial['image_path']
    capList = ds_validation_visdial['caption']


    questionList_val = []
    ansList_val = []
    img_path_list_val = []
    captionList_val = []

    for i in range(2064):
        
        dialogSet = dialog[i]
        img = img_list[i]
        caption = capList[i]
        
        for j in range(10):
            question = dialogSet[j][0]
            answer = dialogSet[j][1]
            
            questionList_val.append(question)
            ansList_val.append(answer)
            img_path_list_val.append(img)
            captionList_val.append(caption)


    listToDictionary = {'caption':captionList_val, 'questions':questionList_val, 'answers':ansList_val, 'images':img_path_list_val}
    modified_val_set = Dataset.from_dict(listToDictionary)

    modified_val_set = modified_val_set.cast_column("images", datasets.Image())


    # Taking small portion of Validation set to evaluate validation loss

    shuffled_val_set = modified_val_set.shuffle(seed=42)
    temp = shuffled_val_set[0:200]
    small_val_set = Dataset.from_dict(temp)

    # ---------------------end of all preprocessing---------------------------------


    # Finetuning

    print('Finetuning Code')

    from torch.utils.data import DataLoader
    from datasets import Dataset

    class visdial_dataset(Dataset):
        
        def __init__(self, dataset, processor):
            self.processor = processor
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self,idx):
            item = self.dataset[idx]
            
            encodings = self.processor(images = item["images"], text = item["questions"], padding = "max_length", return_tensors = "pt")
            labels = self.processor(text = item['answers'], padding = "max_length", return_tensors = "pt").input_ids

            encodings['labels'] = labels
            encodings = {k:v.squeeze() for k,v in encodings.items()}

            return encodings

    train_dataset_object = visdial_dataset(modified_train_set, processor)
    train_dataloader = DataLoader(train_dataset_object, shuffle=True, batch_size=16)


    # it means 12,32,870 records has been grouped in batch of size 16.
    # so total number of batch formed is 1232870 / 16 = 77055
    # that means in each epoch, total the loop will repeat for 77,055 times

    tot_number_of_steps = len(train_dataloader)
    number_of_epoch = 10


    # Tensorboard Display

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()


    # --------------------------Main Training Loop--------------------------------------------------

    running_loss = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
    model.train()

    for epoch in range(number_of_epoch):

      # print("Epoch:", epoch)
      
      for idx, batch in enumerate(train_dataloader):
        
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        labels = batch.pop("labels").to(device)
        attention_mask = batch.pop('attention_mask').to(device)
        
        outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=labels,
                        attention_mask = attention_mask
                       )
        
        loss = outputs.loss

        # print("Loss:", loss.item())
        print(idx+1 ,end = ' ')
        running_loss = running_loss + loss.item()
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        # tensorboard display
        # here (epoch * tot_number_of_steps + i) denotes the current step out of the total (77054 * num_of_epoch) steps.
        # And, running_loss/100 is the mean loss of 100 iterations.
        
        if (idx+1) % 100 == 0:
            
            print(f'\n Epoch:{epoch+1}, Step:{(idx+1)/tot_number_of_steps}, Output Loss:{loss.item()} \n')
            
            val_loss = evaluate_validation_loss_on_val_set(small_val_set)
            writer.add_scalar('Output Loss', running_loss/100, epoch * tot_number_of_steps + idx)
            writer.add_scalar('Validation Loss', val_loss, epoch * tot_number_of_steps + idx)
            
            running_loss = 0

        # save model at last few epoch
        
        if epoch in (0,6,7,8,9):
            save_path = os.path.join("./model_chkpts/", "mod" + str(epoch))
            model.save_pretrained(save_path)

    writer.close()
    sys.exit()

if __name__ == "__main__":
    main()
