# monument-recognition

**Updates: Paper accepted at WDH Workshop, ICVGIP'18 :grimacing:**

This is an implementation of Indian Architectural Classification implemented on Python 3 and Keras with TensorFlow backend.The architecture consists of average ensemble of Graph-based Visual Saliency Network and supervised classification algorithms such as kNN and Random Forest. ImageNet model used for feature generation is Inception ResNet V2.

![collage](https://user-images.githubusercontent.com/22872200/48219234-fc839b00-e3b1-11e8-8efb-dea1392663a3.jpg)

The repository includes:

* Load Training batches for the model
* Salient Region Detection
* Finetuning on ImageNet models - Inception V3 and Inception ResNet V2
* Multi-stage Training
* Graph-ased Visual Saliency and ImageNet model end-to-end
* DELF Landmark retrieval
* Evaluation File and Metrics

## Getting Started

* Install the required dependencies:
 ```javascript
 pip install -r requirements.txt
```
* 
* 

## Step by Step Classification 

## Dataset

The dataset used in this repo is made by our team. We did scrapping from several websites and then filtered out corrupt images to genrate a datset of 3514 images. The dataset is divided in the ratio of 80/10/10 (2809/354/351) that is train/val/test respectively.

Classes| Total |  Training | Validation | Test
-------------| --------- | ---------  | ---------- | ----------
Buddhist  | 809  | 647 | 81 | 81 
Dravidian | 822  | 657 | 83 | 82
Kalinga   | 1102 | 881 | 111| 110
Mughal    | 781  | 624 | 79 | 78

## Graph-based Visual Saliency

![gbvs](https://user-images.githubusercontent.com/22872200/49278828-e4cc9d80-f4ab-11e8-950c-ded899cbd559.png)



1) Image Net models & Evaluation Metric Finalisation

d) IV3

e) Inception ResNet V2 

3) Saliency Detection - Objectness Trained Model

4) sklearn feature classification - kNN, SVM, Random Forest

5) Model Architecture Finalize

6) Model Ensemble methods

## Test Results
Results on the test data :

Model Architecture| Epochs | Train | Validation | Test
------------- | -------- | ---------  | ---------- | ----------
Inception V3  | 7| 90 | 83|83.47
Inception ResNet V2  | 7| 91 |77 |76.35
Images + Saliency(IRV2)|5||81|80
Images + Saliency(IV3)|5||80|78.91

Test Images prediction - 

1) First Network Architecture - 

Test Image-> Saliency -> Batch Formation -> ImageNet Weights

GBVS + IRV2(IRV2 + Saliency wts) - 5 images - 81.05 &&& 10 images - 85.18

GBVS + IV3(IV3 wts only) - 10 images - 80.626

GBVS + IV3(IV3 + Saliency wts) - 10 images - 



2) 2nd Network architecture: - 
IV3 - kNN - 87% 

IRV2 - kNN - 88%

Ensemble Difreent Classifiers - 91% approximately

Parameters: n_neighbours = 20

## To-dos
Dataset Visualization
DELF Image Feature Retrieval 

## References

[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, "[
Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)" arXiv preprint arXiv:1512.00567.

[2] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi, "[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)" arXiv preprint arXiv:1602.07261. 

[3] TRIANTAFYLLIDIS, Georgios; KALLIATAKIS, Gregory. "[Image based Monument Recognition using Graph based Visual Saliency](https://elcvia.cvc.uab.es/article/view/v12-n2-triantafyllidis-kalliatakis)", ELCVIA Electronic Letters on Computer Vision and Image Analysis.
