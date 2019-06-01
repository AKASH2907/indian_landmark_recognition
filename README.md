# monument-recognition

**Updates: Paper accepted at WDH Workshop, 11th ICVGIP'18 :grimacing:**

This is an implementation of Indian Architectural Classification implemented on Python 3 and Keras with TensorFlow backend.The architecture consists of average ensemble of Graph-based Visual Saliency Network and supervised classification algorithms such as kNN and Random Forest. ImageNet model used for feature generation is Inception ResNet V2.

![collage](https://user-images.githubusercontent.com/22872200/48219234-fc839b00-e3b1-11e8-8efb-dea1392663a3.jpg)

The repository includes:

* Load Training batches for the model
* Salient Region Detection
* Finetuning on ImageNet models - Inception V3 and Inception ResNet V2
* Multi-stage Training
* Graph-based Visual Saliency and ImageNet model end-to-end
* DELF Landmark retrieval
* Evaluation File and Metrics

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below).

## Getting Started

* Install the required dependencies:
 ```javascript
 pip install -r requirements.txt
```
* [inception_v3_finetuning.py](https://github.com/AKASH2907/indian_landmark_recognition/blob/master/inception_v3_finetuning.py): Transfer Learning on Original Images and Salient Images.
* [inception_resnet_v2_finetuning.py](https://github.com/AKASH2907/indian_landmark_recognition/blob/master/inception_resnet_v2_finetuning.py): Transfer Learning  on Original Images and Salient Images.
* [saliency.py](https://github.com/AKASH2907/indian_landmark_recognition/blob/master/saliency.py): Detect salient regions in an image. 

## Step by Step Classification 

## Citation
If you use this repistory, please cite the paper as follows:
```
@article{DBLP:journals/corr/abs-1811-12748,
  author    = {Akash Kumar and
               Sagnik Bhowmick and
               N. Jayanthi and
               S. Indu},
  title     = {Improving Landmark Recognition using Saliency detection and Feature
               classification},
  journal   = {CoRR},
  volume    = {abs/1811.12748},
  year      = {2018}
}
```
## Dataset

The dataset used in this repo is made by our team. We did scrapping from several websites and then filtered out corrupt images to genrate a datset of 3514 images. The dataset is divided in the ratio of 80/10/10 (2809/354/351) that is train/val/test respectively.

Classes| Total |  Training | Validation | Test
-------------| --------- | ---------  | ---------- | ----------
Buddhist  | 809  | 647 | 81 | 81 
Dravidian | 822  | 657 | 83 | 82
Kalinga   | 1102 | 881 | 111| 110
Mughal    | 781  | 624 | 79 | 78

## Graph-based Visual Saliency

![gbvs](https://user-images.githubusercontent.com/22872200/49820966-4d1a5980-fd9f-11e8-9d65-c385fd12592d.png)

Image Saliency is what stands out and how fast you are able to quickly focus on the most relevant parts of what you see. Now, in the case of landmarks the less salient region is common backgrounds, that’s of blue sky. The architectural de-
sign of the monuments is what differentiates between the classes. 


## Test Results
Accuracy during Multi-Stage Training on Inception V3 and Inception ResNet V2 models :

Model Architecture| Data Subset | Train | Validation | Test
------------- | -------- | ---------  | ---------- | ----------
Inception V3  | Original Images| 90 | 77.23|75.42
Inception V3  | Original + Salient| 91.81 |80.3 |78.91
Inception ResNet V2|Original Images|91.76|77|76.35
Inception ResNet V2|Original + Salient|92.29|81|80

Evaluation comparison (in %) of different models:

Model Architecture|  Train | Validation | Test
------------- | ---------  | ---------- | -------
GBVS + InceptionResNetV2| 92.61 | 89.65 |  86.18
Inception ResNetV2 + kNN| 93.62 | 90.72 | 86.94
Inception ResNetV2 + Random Forest| 91.58 | 89.8
Average Ensembling |94.58|93.8|90.08

Comparison of our best model with competing methods[4]:

Framework|Test
---------|----
SIFT + BoW| 51%
Gabor Transform + Radon Barcode | 70%
Radon Barcode| 75%
CNN | 82%
Our Method | 91%

Test Images prediction - 

1) 1st Network Architecture - 

Test Image-> Saliency -> Batch Formation -> Pretrained ImageNet Weights
 
2) 2nd Network architecture: - 
Pretrained Inception V3 -> kNN - 87% 

IRV2 - kNN - 88%

Ensemble Diffrent Classifiers - 91% approximately

Parameters: n_neighbours = 20

## References

[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, "[
Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)" arXiv preprint arXiv:1512.00567.

[2] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi, "[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)" arXiv preprint arXiv:1602.07261. 

[3] TRIANTAFYLLIDIS, Georgios; KALLIATAKIS, Gregory. "[Image based Monument Recognition using Graph based Visual Saliency](https://elcvia.cvc.uab.es/article/view/v12-n2-triantafyllidis-kalliatakis)", ELCVIA Electronic Letters on Computer Vision and Image Analysis.
[4] Sharma S., Aggarwal P., Bhattacharyya A.N., Indu S. (2018) Classification of Indian Monuments into Architectural Styles. NCVPRIPG 2017. Communications in Computer and Information Science, vol 841. Springer, Singapore.
