# SDSK2BERT


## Background
This work use the specific knowledge to explore the specific depth when distilling the BERT model. The specific knowledge refers to the weights learned in distillation, which may already have the ability to classify. Initializing a smaller student with such knowledge will reduce the possibility of trapping in bad local saddels. 
<img src="https://github.com/LifangD/SDSK2BERT/blob/master/imgs/clustering.png" width="50%">


## Architecture 
<img src="https://github.com/LifangD/SDSK2BERT/blob/master/imgs/arc.png" width="50%">

## Prepare  
  1. Dataset: DNLI、MNLI
  2. Download Pretrained BERT (English, base&large)
      
       

## Train & Test
Please check if the resourses are prepared and the paths/arguments are specified. Example is shown in scripts/*.sh

#### step1: train the teacher

```
sh scripts/train_tea.sh
```

#### step2: save the logits from the teacher
Loading the large teacher is avoided, which accelerates the training process
```
sh scripts/get_logit.sh
```

#### step3: train the student
```
sh scripts/train_stu.sh
```

#### step4: eval the student 
```python
python eval.py
```

## Result 
<img src="https://github.com/LifangD/SDSK2BERT/blob/master/imgs/result.png" width="50%">




