from PIL import Image
import datasets
import numpy as np
import torch
import requests
import os
import warnings
    
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

from datasets import load_dataset
from datasets import Dataset
from transformers import BlipProcessor, BlipForQuestionAnswering

# for fp16 precision

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# for metrics

from nltk.translate.bleu_score import sentence_bleu
import evaluate

# time

import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('\n Device Selected:', device)


# loading BLIP VQA model

print('\n Loading BLIP VQA Model In: ', device)
processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("ybelkada/blip-vqa-base").to("cuda")


# By default the sentence_bleu() function calculates the cumulative 4-gram BLEU score, also called BLEU-4

# Cumulative and individual 1-gram BLEU Score
def calculate_cumulative_bleu_scores(pred,ground):
    
    bleu1scores = []
    bleu2scores = []
    bleu3scores = []

    for i in range(len(pred)):
        
        reference = [ground[i].split()]
        candidate = pred[i].split()
        
        score_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        score_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        score_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))

        bleu1scores.append(score_1)
        bleu2scores.append(score_2)
        bleu3scores.append(score_3)            

    bleu1 = sum(bleu1scores)/len(bleu1scores)
    bleu2 = sum(bleu2scores)/len(bleu2scores)
    bleu3 = sum(bleu3scores)/len(bleu3scores)

    return (bleu1,bleu2,bleu3)


def calculate_rogueL_score(pred,ground):
    
    rouge = evaluate.load('rouge')
    predictions = pred
    references = ground
    results = rouge.compute(predictions=predictions,references=references)
    
    # print(results)
    return results['rougeL']


def evaluate_validation_metrics_and_loss_on_val_set(val_set_questions,val_set_answers,val_set_images):
    
    groundTruthList = val_set_answers
    predictedList = []
    running_val_loss = 0.0
    
    # set the model to inference mode
    model.eval()
    
    for i in range(150):
        
        question = val_set_questions[i]
        image = val_set_images[i]
        label = val_set_answers[i]
                
        inputs = processor(images = image, text = question, return_tensors = "pt").to(device)
        labels = processor(text = label, return_tensors = "pt").input_ids
        inputs['labels'] = labels
        
        val_model = model(**inputs)
        val_loss = val_model.loss

        running_val_loss = running_val_loss + val_loss.item()
        
        answer = model.generate(**inputs)
        ans = processor.decode(answer[0], skip_special_tokens=True)

        predictedList.append(ans)
        
    bScore = calculate_cumulative_bleu_scores(predictedList, groundTruthList)
    rScore = calculate_rogueL_score(predictedList, groundTruthList)
    
    model.train()
    
    return (bScore, rScore, (running_val_loss/150))


def main():
    
    from datasets import Dataset

    # loading dataset

    print('\n Loading Dataset')
    ds_visdial = datasets.load_dataset('jxu124/visdial', split="train")


    # removing unnecessary columns

    print('\n Removing Column')
    ds_visdial = ds_visdial.remove_columns('global_image_id')
    ds_visdial = ds_visdial.remove_columns('anns_id')


    img_list = ds_visdial['image_path']
    dialog = ds_visdial['dialog']
    capList = ds_visdial['caption']


    # Training Set Creation

    print('\n Training Set Creation')
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

    print('\n Training Set Created')

    #------------------------end of creation of training set-------------------------------------------


    #------------------------Creation of Validation set------------------------------------------------

    print('\n Validation Set Creation Started')
    ds_validation_visdial = datasets.load_dataset('jxu124/visdial', split="validation")


    # Removing Unneessary columns

    ds_validation_visdial = ds_validation_visdial.remove_columns('global_image_id')
    ds_validation_visdial = ds_validation_visdial.remove_columns('anns_id')

    dialog = ds_validation_visdial['dialog']
    img_list = ds_validation_visdial['image_path']
    capList = ds_validation_visdial['caption']

    # Validation Set Creation

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
    temp = shuffled_val_set[0:150]
    small_val_set = Dataset.from_dict(temp)


    # Validation set separated into 3 list for faster retrieval of data instead of the HF dataset

    val_set_questions = small_val_set['questions']
    val_set_answers = small_val_set['answers']
    val_set_images = small_val_set['images']

    print('\n Validation Set Created')

    # ---------------------end of all preprocessing of Training & Validation Set---------------------



    #--------------------------------------Finetuning Code---------------------------------------------

    print('\n Finetuning Code Begins')

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
    Scaler = GradScaler()

    # Tensorboard Display

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()


    # -------------------------Training Loop With Grad Accumulation---------------------------------------
    
    current_datetime_object = datetime.datetime.now()
    start_time = current_datetime_object.hour
    print(start_time)

    running_loss = 0.0
    gradient_accumulation_steps = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
    model.train()

    for epoch in range(number_of_epoch):

        print('\n Loop Begins')

        for idx, batch in enumerate(train_dataloader):
        
            input_ids = batch.pop("input_ids").to("cuda")
            pixel_values = batch.pop("pixel_values").to("cuda")
            labels = batch.pop("labels").to("cuda")
            attention_mask = batch.pop('attention_mask').to("cuda")
            
            with autocast():
                outputs = model(
                                input_ids=input_ids,
                                pixel_values=pixel_values,
                                labels=labels,
                                attention_mask = attention_mask
                               )

                loss = outputs.loss
            
            
            print(idx+1 ,end = ' ')
            running_loss = running_loss + loss.item()

            Scaler.scale(loss).backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                Scaler.step(optimizer)
                optimizer.zero_grad()
                Scaler.update()
            
            # tensorboard display
            # here (epoch * tot_number_of_steps + i) denotes the current step out of the total (77054 * num_of_epoch) steps.
            # And, running_loss/100 is the mean loss of 100 iterations.
            
            if (idx+1) % 300 == 0:
                
                print(f'\n Epoch:{epoch+1}, Step:{(idx+1)/tot_number_of_steps}, Output Loss:{loss}')
                
                validation_results = evaluate_validation_metrics_and_loss_on_val_set(val_set_questions,val_set_answers,val_set_images)
                returned_bScore_1,returned_bScore_2,returned_bScore_3 = validation_results[0]
                returned_rScore = validation_results[1]
                returned_valLoss = validation_results[2]
                
                print(f'rScore: {returned_rScore}, Val Loss: {returned_valLoss}\n')
                print(f'bScore1:{returned_bScore_1}, bScore2:{returned_bScore_2}, bScore3:{returned_bScore_3}' )

                writer.add_scalar('Training Loss', running_loss/100, epoch * tot_number_of_steps + idx)
                writer.add_scalar('Validation Loss', returned_valLoss, epoch * tot_number_of_steps + idx)

                writer.add_scalar('Cumm_BLEU_1 Score', returned_bScore_1, epoch * tot_number_of_steps + idx)
                writer.add_scalar('Cumm_BLEU_2 Score', returned_bScore_2, epoch * tot_number_of_steps + idx)
                writer.add_scalar('Cumm_BLEU_3 Score', returned_bScore_3, epoch * tot_number_of_steps + idx)

                writer.add_scalar('RogueL Score', returned_rScore, epoch * tot_number_of_steps + idx)

                
                running_loss = 0
            
            # save model at last few epoch and after fixed time interval of 6 hrs
            
            current_datetime_object_in_loop = datetime.datetime.now()
            curr_time = current_datetime_object_in_loop.hour
            current_date = datetime.date.today()

            # print('Curr TIme:',curr_time)
            
        if epoch in (0,1,2,3,4,5,6,7,8,9):
            print('\n ..........Save Triggered.................... \n')
            save_path = os.path.join("./model_chkpts/", "mod" + "_" + str(epoch) + "_" + str(current_datetime_object_in_loop.hour) + "-" + str(curr_time) + "_" + str(current_date))
            model.save_pretrained(save_path)
                
    writer.close()
    sys.exit()

if __name__ == "__main__":
    main()
