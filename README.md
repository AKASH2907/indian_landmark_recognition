# monument-classification
indian monumnet classification


## To-dos
1) Image-Net models check - 

ResNet-101/151
ResNeXt

3)Grid Search Method

4) Model Ensemble methods

Evaluation Metric Finalisation
DELF Features use 
Features Generate

## 5) Model Architecture Finalize

## Done

d) IV3

e) Inception ResNet V2 

2) Dataset Partition (After Cleansing Corrupted Images)

a) Buddhist - 809 - 647/81/81

b) Dravidian - 822 - 657/83/82

c) kalinga - 1102 - 881/111/110

d) Mughal  - 781 - 624/79/78

Total Number of Training Images Originally (after cleansing) - 3514 - 2809/354/351

Train/Val/Test Split - 80/10/10 - For Now

3) Saliency Detection - Objectness Trained Model

## Test Results
Results on the test data :

Model Architecture| Epochs | Train | Validation | Test
------------- | -------- | ---------  | ---------- | ----------
Inception V3  | 7| 90 | 80|77.2 
Inception ResNet V2  | 7| 91 |80  |80.9
Saliency | 7 | |79|78.91
Images + Saliency| 4|||80.79
Saliency + Images |4 |||79

Test Image-> Saliency -> Batch Formation -> ImageNet Weights
