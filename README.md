# awesome-video-anomaly-detection  [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
Papers for Video Anomaly Detection, released codes collections.
## Datasets
0. UMN
1. UCSD
2. Subway Entrance/Exit
3. CUHK Avenue
    - HD-Avenue
4. ShanghaiTech
    - HD-ShanghaiTech
5. UCF-Crime (Weakly Supervised)
    - UCFCrime2Local (subset of UCF-Crime but with spatial annotations.) [download_link](http://imagelab.ing.unimore.it/UCFCrime2Local)
    - Spatial Temporal Annotations [download_link](https://github.com/xuzero/UCFCrime_BoundingBox_Annotation)
6. Traffic-Train
7. Belleview
8. Street Scene (WACV 2020)
9. IITB-Corridor (WACV 2020)
-----
## Unsupervised
### 2017
1. Joint Detection and Recounting of Abnormal Events by Learning Deep Generic Knowledge, ICCV 2017. (Explainable VAD)
2. A revisit of sparse coding based anomaly detection in stacked rnn framework, ICCV 2017. [code](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection)
### 2018
1. Future Frame Prediction for Anomaly Detection -- A New Baseline, CVPR 2018. [code](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
2. Adversarially Learned One-Class Classifier for Novelty Detection, CVPR 2018. [code](https://github.com/khalooei/ALOCC-CVPR2018)

### 2019
1. Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection, ICCV 2019.[code](https://github.com/donggong1/memae-anomaly-detection)
2. Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos, CVPR 2019.[code](https://github.com/RomeroBarata/skeleton_based_anomaly_detection)
3. Object-Centric Auto-Encoders and Dummy Anomalies for Abnormal Event Detection, CVPR 2019.
4. Anomaly Detection in Video Sequence with Appearance-Motion Correspondence, ICCV 2019.
5. AnoPCN: Video Anomaly Detection via Deep Predictive Coding Network, ACM MM 2019.
### 2020
1. Street Scene: A new dataset and evaluation protocol for video anomaly detection, WACV 2020.
2. Multi-timescale Trajectory Prediction for Abnormal Human Activity Detection, WACV 2020.
3. Continual Learning for Anomaly Detection in Surveillance Videos, CVPR 2020.[paper](https://arxiv.org/pdf/2004.07941.pdf)
4. Self-trained Deep Ordinal Regression for End-to-End Video Anomaly Detection, CVPR 2020. [paper](https://arxiv.org/pdf/2003.06780.pdf)
## Weakly-Supervised
### 2018
1. Real-world Anomaly Detection in Surveillance Videos, CVPR 2018 [code](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)
### 2019
1. Graph Convolutional Label Noise Cleaner:Train a Plug-and-play Action Classifier for Anomaly Detection, CVPR 2019, 
[code](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection)
2. Margin Learning Embedded Prediction for Video Anomaly Detection with A Few Anomalies, IJCAI 2019.
3. Temporal Convolutional Network with Complementary Inner Bg Loss For Weakly Supervised Anomaly Detection. ICIP 19.
4. Motion-Aware Feature for Improved Video Anomaly Detection. BMVC 19.
### 2020
1. Learning a distance function with a Siamese network to localize anomalies in videos, WACV 2020.

## Supervised
### 2019
1. Exploring Background-bias for Anomaly Detection in Surveillance Videos, ACM MM 19.
2. Anomaly locality in video suveillance, ICIP 19.

------
## Reviews / Surveys
1. An Overview of Deep Learning Based Methods for Unsupervised and Semi-Supervised Anomaly Detection in Videos, J. Image, 2018.[page](https://beedotkiran.github.io/VideoAnomaly.html)
2. DEEP LEARNING FOR ANOMALY DETECTION: A SURVEY, [paper](https://arxiv.org/pdf/1901.03407.pdf)
3. Video Anomaly Detection for Smart Surveillance [paper](https://arxiv.org/pdf/2004.00222.pdf)

## Books
1. Outlier Analysis. Charu C. Aggarwal
------
Generally, anomaly detection in recent researchs are based on the datasets get from pedestrian (likes UCSD, Avenue, ShanghaiTech, etc.)， or UCF-Crime (real-wrold anomaly).
However some focus on specefic scene as follows.

## Specific Scene
### Traffic
CVPR 2018 workshop, CVPR 2019 workshop, AICity Challenge series.
#### Driving
When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos. [github](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)

### Old-man Fall Down

### Fighting/Violence
1. Localization Guided Fight Action Detection in Survellance Videos. ICME 2019.
2. 

### Social/ Group Anomaly
1. Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks, Neurips 2019.

## Related Topics:
1. Video Representation (Unsupervised Video Representation, reconstruction, prediction etc.)
2. Object Detection
3. Pedestrian Detection
4. Skeleton Detection
5. Graph Neural Networks
6. GAN
7. Action Recongnition / Temporal Action Localization
8. Metric Learning
9. Label Noise Learning
10. Cross-Modal/ Multi-Modal
11. Dictionary Learning
12. One-Class Classification / Novelty Detection / Out-of-Disturibution Detection
13. Action Recognition.
    - Human in Events: A Large-Scale Benchmark for Human-centric Video Analysis in Complex Events. ACM MM 2020 workshop.

## Performance Evaluation Methods
1. AUC
2. PR-AUC
3. Score Gap
4. False Alarm Rate on Normal with 0.5 as threshold (Weakly supervised, proposed in CVPR 18)

## Performance Comparision on UCF-Crime 
|Model| Convference/Journal |Supervised| Feature | End2End| 32 Segments | AUC (%) | FAR@0.5 on Normal (%)| 
|----|----|----|----|-----|----|----|----|
|Deep MIL Ranking | CVPR 18 | Weakly | C3D RGB | X | √ | 75.41 | 1.9|
|MIL_IBL |  ICIP 19 | Weakly | C3D RGB | X | √ |  78.66 | -|
|MA_MIL| BMVC 19 | Weakly | PWC Flow| X | √ |  79.0 | -|
|Graph Label Noise Cleaner | CVPR 19 | Weakly | TSN RGB | √ | X | 82.12 | 0.1|
|Background Bias | ACM MM 19 | Fully | NLN RGB | √ | X | 82.0 | - |
