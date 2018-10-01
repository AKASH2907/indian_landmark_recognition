# monument-classification
indian monumnet classification


## To-dos
1) Image-Net models check - 

a) Alexnet

b) VGG-16

c) ReSNet-50

(Could recheck with 60/20/20)

3)Grid Search Method

4) Model Ensemble methods

5) Dataset Augmentation - imgaug library

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

## Test Results
Results on the test data :

Model Architecture| Data Subset | Train | Validation | Test
------------- | -------- | ---------  | ---------- | ----------
Inception V3  | Images| 90 | 80|77.2 
Inception ResNet V2  | Images| 97.29 |29.17  |47.96
