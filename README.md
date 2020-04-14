# awesome-video-anomaly-detection  [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
Papers for Video Anomaly Detection, released codes collections.
## Datasets
0. UMN
1. UCSD
2. Subway Entrance/Exit
3. CUHK Avenue
4. ShanghaiTech
5. UCF-Crime (Weakly Supervised)
6. Traffic-Train
7. Belleview
8. Street Scene (WACV 2020)
9. IITB-Corridor (WACV 2020)
-----
## Unsupervised
1. Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection, ICCV 2019.[code](https://github.com/donggong1/memae-anomaly-detection)
2. Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos, CVPR 2019.[code](https://github.com/RomeroBarata/skeleton_based_anomaly_detection)
3. Object-Centric Auto-Encoders and Dummy Anomalies for Abnormal Event Detection, CVPR 2019.
4. Anomaly Detection in Video Sequence with Appearance-Motion Correspondence, ICCV 2019.
5. Future Frame Prediction for Anomaly Detection -- A New Baseline, CVPR 2018. [code](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
6. Adversarially Learned One-Class Classifier for Novelty Detection, CVPR 2018. [code](https://github.com/khalooei/ALOCC-CVPR2018)
7. Joint Detection and Recounting of Abnormal Events by Learning Deep Generic Knowledge, ICCV 2017. (Explainable VAD)
8. A revisit of sparse coding based anomaly detection in stacked rnn framework, ICCV 2017. [code](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection)
9. Street Scene: A new dataset and evaluation protocol for video anomaly detection, WACV 2020.
10. Multi-timescale Trajectory Prediction for Abnormal Human Activity Detection, WACV 2020.

## Weakly-Supervised
1. Real-world Anomaly Detection in Surveillance Videos, CVPR 2018 [code](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)
2. Graph Convolutional Label Noise Cleaner:Train a Plug-and-play Action Classifier for Anomaly Detection, CVPR 2019, [code](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection)
3. Margin Learning Embedded Prediction for Video Anomaly Detection with A Few Anomalies, IJCAI 2019.
4. Temporal Convolutional Network with Complementary Inner Bg Loss For Weakly Supervised Anomaly Detection. ICIP 19.

------
## Reviews / Surveys
1. An Overview of Deep Learning Based Methods for Unsupervised and Semi-Supervised Anomaly Detection in Videos, J. Image, 2018.[page](https://beedotkiran.github.io/VideoAnomaly.html)
2. 

------
Generally, anomaly detection in recent researchs are based on the datasets get from pedestrian (likes UCSD, Avenue, ShanghaiTech, etc.)， or UCF-Crime (real-wrold anomaly).
However some focus on specefic scene as follows.

## Specific Scene
### Traffic
CVPR 2018 workshop, CVPR 2019 workshop, AICity Challenge series.

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

## Performance Evaluation Methods
1. AUC
2. PR-AUC
3. Score Gap
4. False Alarm Rate on Normal with 0.5 as threshold (Weakly supervised, proposed in CVPR 18)
