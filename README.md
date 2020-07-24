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
### 2016
1. <span id = "01601">[[Conv-AE]]</span>[Learning Temporal Regularity in Video Sequences](https://openaccess.thecvf.com/content_cvpr_2016/papers/Hasan_Learning_Temporal_Regularity_CVPR_2016_paper.pdf), CVPR 16. [Code](https://github.com/iwyoo/TemporalRegularityDetector-tensorflow/blob/master/model.py)
### 2017
1. <span id = "01701">[[Hinami.etl]]</span>[Joint Detection and Recounting of Abnormal Events by Learning Deep Generic Knowledge](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hinami_Joint_Detection_and_ICCV_2017_paper.pdf), ICCV 2017. (Explainable VAD)
2. <span id = "01702">[[Stacked-RNN]]</span>[A revisit of sparse coding based anomaly detection in stacked rnn framework](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_A_Revisit_of_ICCV_2017_paper.pdf), ICCV 2017. [code](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection)
3. <span id = "01703">[[ConvLSTM-AE]]</span>[Remembering history with convolutional LSTM for anomaly detection](https://ieeexplore.ieee.org/abstract/document/8019325), ICME 2017.[Code](https://github.com/zachluo/convlstm_anomaly_detection)
4. <span id = "01704">[[Conv3D-AE]]</span>[Spatio-Temporal AutoEncoder for Video Anomaly Detection](https://dl.acm.org/doi/abs/10.1145/3123266.3123451)，ACM MM 17.
### 2018
1. <span id = "01801">[[FramePred]]</span>[Future Frame Prediction for Anomaly Detection -- A New Baseline](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Future_Frame_Prediction_CVPR_2018_paper.pdf), CVPR 2018. [code](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
2. <span id = "01802">[[ALOOC]]</span>[Adversarially Learned One-Class Classifier for Novelty Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.pdf), CVPR 2018. [code](https://github.com/khalooei/ALOCC-CVPR2018)

### 2019
1. <span id = "01901">[[Mem-AE]]</span>[Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf), ICCV 2019.[code](https://github.com/donggong1/memae-anomaly-detection)
2. <span id = "01902">[[Skeleton-based]]</span>[Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos](http://openaccess.thecvf.com/content_CVPR_2019/papers/Morais_Learning_Regularity_in_Skeleton_Trajectories_for_Anomaly_Detection_in_Videos_CVPR_2019_paper.pdf), CVPR 2019.[code](https://github.com/RomeroBarata/skeleton_based_anomaly_detection)
3. <span id = "01903">[[Object-Centric]]</span>[Object-Centric Auto-Encoders and Dummy Anomalies for Abnormal Event Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf), CVPR 2019.
4. <span id = "01904">[[Appearance-Motion Correspondence]]</span>[Anomaly Detection in Video Sequence with Appearance-Motion Correspondence](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.pdf), ICCV 2019.
5. <span id = "01905">[[AnoPCN]]</span>[AnoPCN: Video Anomaly Detection via Deep Predictive Coding Network](https://people.cs.clemson.edu/~jzwang/20018630/mm2019/p1805-ye.pdf), ACM MM 2019.
### 2020
1. <span id = "02001">[[Street-Scene]]</span>[Street Scene: A new dataset and evaluation protocol for video anomaly detection](http://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf), WACV 2020.
2. <span id = "02002">[[Rodrigurs.etl]])</span>[Multi-timescale Trajectory Prediction for Abnormal Human Activity Detection](http://openaccess.thecvf.com/content_WACV_2020/papers/Rodrigues_Multi-timescale_Trajectory_Prediction_for_Abnormal_Human_Activity_Detection_WACV_2020_paper.pdf), WACV 2020.
3. <span id = "02003">[[GEPC]]</span>[Graph Embedded Pose Clustering for Anomaly Detection](https://arxiv.org/pdf/1912.11850.pdf), CVPR 2020.[code](https://github.com/amirmk89/gepc)
4. <span id = "02004">[[Self-trained]]</span>[Self-trained Deep Ordinal Regression for End-to-End Video Anomaly Detection](https://arxiv.org/pdf/2003.06780.pdf), CVPR 2020. 
5. <span id = "02005">[[MNAD]]</span>[Learning Memory-guided Normality for Anomaly Detection](https://arxiv.org/pdf/2003.13228.pdf), CVPR 2020. [code](https://cvlab.yonsei.ac.kr/projects/MNAD)
6. <span id = "02006">[[MNAD]]</span>[Continual Learning for Anomaly Detection in Surveillance Videos] (https://arxiv.org/pdf/2004.07941),CVPR 2020 Worksop.
7. <span id = "02007">[[Old is Gold]]</span>[Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.pdf), CVPR 2020.
8. <span id = "02008">[[Any-Shot]]</span>[Any-Shot Sequential Anomaly Detection in Surveillance Videos]((http://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Doshi_Any-Shot_Sequential_Anomaly_Detection_in_Surveillance_Videos_CVPRW_2020_paper.pdf)), CVPR 2020 workshop.

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
2. [Weakly Supervised Video Anomaly Detection via Center-Guided Discrimative Learning](https://ieeexplore.ieee.org/document/9102722), ICME 2020.


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
#### First-Person Traffic
1. Unsupervised Traffic Accident Detection in First-Person Videos, IROS 2019.

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
|MIL | CVPR 18 | Weakly | C3D RGB | X | √ | 75.41 | 1.9|
|IBL |  ICIP 19 | Weakly | C3D RGB | X | √ |  78.66 | -|
|MA_MIL| BMVC 19 | Weakly | PWC Flow| X | √ |  79.0 | -|
|Graph Label Noise Cleaner | CVPR 19 | Weakly | TSN RGB | √ | X | 82.12 | 0.1|
|Background Bias | ACM MM 19 | Fully | NLN RGB | √ | X | 82.0 | - |

## Perfromace Comparision on ShanghaiTech
| Model | Reported on Conference/Journal | Supervision | Feature | End2Emd |  AUC(%) | FAR@0.5 (%) |
|----|----|----|----|-----|----|----|
| <span id = "31601">[[Conv-AE]](#01601)</span> | CVPR 16 | Un | - | √ | 60.85 | - |
| <span id = "31702">[[stacked-RNN]](#01702)</span> | ICCV 17 | Un | - | √ | 68.0 | - |
| future pred | CVPR 18 | Un | - | √ | 72.8 | - |
| future pred * | IJCAI 19 | Un | - | √ | 73.4 | - |
| Mem-AE | ICCV 19 | Un | - | √ | 71.2 | - |
| Mem-Norm | CVPR 20 | Un | - |  √ | 70.5 | - |
| MLEP |IJCAI 19 | 10% test vids with Video Anno | - | √ | 75.6 | - |
| MLEP |IJCAI 19 | 10% test vids with Frame Anno | - | √ | 76.8 | - |
| MIL | ICME 2020 | Weakly (Re-Organized Dataset) | I3D-RGB | X | 86.3 | 0.15 |
| IBL | ICME 2020 | Weakly (Re-Organized Dataset) | I3D-RGB | X | 82.5 | 0.10 |
| GCN label cleaner | CVPR 19 | Weakly (Re-Organized Dataset) | C3D-RGB | √ | 76.44 |  - |
| GCN label cleaner | CVPR 19 | Weakly (Re-Organized Dataset) | TSN-Flow | √ | 84.13 |  - |
| GCN label cleaner | CVPR 19 | Weakly (Re-Organized Dataset) | TSN-RGB | √ | 84.44| - | 
| AR-Net | ICME 20 | Weakly (Re-Organized Dataset) | I3D-RGB & I3D Flow | X | 91.24| 0.10 |
## Performance Comparision on Avenue 
| Model | Conference/Journal | Supervision | Feature | End2End |  AUC(%) |
|----|----|----|----|-----|----|
| Conv-AE | CVPR 16 | Un | - | √ | 80.0 |
| ConvLSTM-AE | ICME 17 | Un | - | √ | 77.0 | 
| DeepAppearance | ICAIP 17 | Un | - | √ | 84.6 |
| Unmasking | ICCV 17 | Un | 3D gradients+VGG conv5 | X | 80.6 |
| sRNN | ICCV 17 | Un | - | √ |  81.7 |
| future pred | CVPR 18 | Un | - | √ | 85.1 |
| Mem-AE | ICCV 19 | Un | - | √ | 83.3 |
| ACT | ICCV 19 | Un | - | √ | 86.9 |
| future pred * | IJCAI 19 | Un | - | √ | 89.2 |
| Mem-Norm | CVPR 20 | Un | - |  √ | 88.5 |
| MLEP |IJCAI 19 | 10% test vids with Video Anno | - | √ | 91.3 |
| MLEP |IJCAI 19 | 10% test vids with Frame Anno | - | √ | 92.8 |
