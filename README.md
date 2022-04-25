# awesome-video-anomaly-detection  [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
Papers for Video Anomaly Detection, released codes collections.

Any addition or bug please open an issue, pull requests or e-mail me by `fjchange@hotmail.com ` 

## Recent Updated
- AAAI 2022
- CVPR 2022

## Datasets
0. UMN [`Download link`](http://mha.cs.umn.edu/)
1. UCSD [`Download link`](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)
2. Subway Entrance/Exit [`Download link`](http://vision.eecs.yorku.ca/research/anomalous-behaviour-data/)
3. CUHK Avenue [`Download link`](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
    - HD-Avenue <span id = "05">[Skeleton-based](#01902)</span>
4. ShanghaiTech [`Download link`](https://svip-lab.github.io/dataset/campus_dataset.html)
    - HD-ShanghaiTech <span id = "00">[Skeleton-based](#01902)</span>
5. UCF-Crime (Weakly Supervised)
    - UCFCrime2Local (subset of UCF-Crime but with spatial annotations.) [`Download_link`](http://imagelab.ing.unimore.it/UCFCrime2Local), <span id = "01">[Ano-Locality](#21902)</span>
    - Spatial Temporal Annotations [`Download_link`](https://github.com/xuzero/UCFCrime_BoundingBox_Annotation) <span id = "02">[Background-Bias](#21901)</span>
6. Traffic-Train
7. Belleview
8. Street Scene (WACV 2020) <span id = "03">[Street Scenes](#02001)</span>, [`Download link`](https://www.merl.com/demos/video-anomaly-detection)
9. IITB-Corridor (WACV 2020) <span id = "04">[Rodrigurs.etl](#02002)</span>
10. XD-Violence (ECCV 2020) <span id ='05'>[XD-Violence](#12003)</span>[`Download link`](https://roc-ng.github.io/XD-Violence/)
11. ADOC (ACCV 2020) <span id ='06'>[ADOC](#02012)</span>[`Download_link`](http://qil.uh.edu/main/datasets/)
12. UBnormal (CVPR 2022) <span id='07'>[UBnormal] [`Project Link`](https://github.com/lilygeorgescu/UBnormal) `Open-Set`

__The Datasets belowed are about Traffic Accidents Anticipating in Dashcam videos or Surveillance videos__

1. CADP [(CarCrash Accidents Detection and Prediction)](https://github.com/ankitshah009/CarCrash_forecasting_and_detection)
2. DAD  [paper](https://yuxng.github.io/chan_accv16.pdf), [`Download link`](https://aliensunmin.github.io/project/dashcam/)
3. A3D  [paper](https://arxiv.org/abs/1903.00618?), [`Download link`](https://github.com/MoonBlvd/tad-IROS2019)
4. DADA  [`Download link`](https://github.com/JWFangit/LOTVS-DADA)
5. DoTA   [`Download_link`](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
6. Iowa DOT [`Download_link`](https://www.aicitychallenge.org/2018-ai-city-challenge/)


1. Driver_Anomaly [Project_link](https://github.com/okankop/Driver-Anomaly-Detection)
-----
## Unsupervised
### 2016
1. <span id = "01601">[Conv-AE]</span> [Learning Temporal Regularity in Video Sequences](https://openaccess.thecvf.com/content_cvpr_2016/papers/Hasan_Learning_Temporal_Regularity_CVPR_2016_paper.pdf), `CVPR 16`. [Code](https://github.com/iwyoo/TemporalRegularityDetector-tensorflow/blob/master/model.py)
### 2017
1. <span id = "01701">[Hinami.etl]</span> [Joint Detection and Recounting of Abnormal Events by Learning Deep Generic Knowledge](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hinami_Joint_Detection_and_ICCV_2017_paper.pdf), `ICCV 2017`. (Explainable VAD)
2. <span id = "01702">[Stacked-RNN]</span> [A revisit of sparse coding based anomaly detection in stacked rnn framework](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_A_Revisit_of_ICCV_2017_paper.pdf), `ICCV 2017`. [code](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection)
3. <span id = "01703">[ConvLSTM-AE]</span> [Remembering history with convolutional LSTM for anomaly detection](https://ieeexplore.ieee.org/abstract/document/8019325), `ICME 2017`.[Code](https://github.com/zachluo/convlstm_anomaly_detection)
4. <span id = "01704">[Conv3D-AE]</span> [Spatio-Temporal AutoEncoder for Video Anomaly Detection](https://dl.acm.org/doi/abs/10.1145/3123266.3123451),`ACM MM 17`.
5. <span id = "01705">[Unmasking]</span> [Unmasking the abnormal events in video](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ionescu_Unmasking_the_Abnormal_ICCV_2017_paper.pdf), `ICCV 17`.
6. <span id = "01706">[DeepAppearance]</span> [Deep appearance features for abnormal behavior detection in video](https://www.researchgate.net/profile/Radu_Tudor_Ionescu/publication/320361315_Deep_Appearance_Features_for_Abnormal_Behavior_Detection_in_Video/links/5a469e9fa6fdcce1971b7258/Deep-Appearance-Features-for-Abnormal-Behavior-Detection-in-Video.pdf)
### 2018
1. <span id = "01801">[FramePred]</span> [Future Frame Prediction for Anomaly Detection -- A New Baseline](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Future_Frame_Prediction_CVPR_2018_paper.pdf), `CVPR 2018`. [code](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
2. <span id = "01802">[ALOOC]</span> [Adversarially Learned One-Class Classifier for Novelty Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.pdf), `CVPR 2018`. [code](https://github.com/khalooei/ALOCC-CVPR2018)
3. [Detecting Abnormality Without Knowing Normality: A Two-stage Approach for Unsupervised Video Abnormal Event Detection](https://dl.acm.org/doi/10.1145/3240508.3240615), `ACM MM 18`.

### 2019
1. <span id = "01901">[Mem-AE]</span> [Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf), `ICCV 2019`.[code](https://github.com/donggong1/memae-anomaly-detection)
2. <span id = "01902">[Skeleton-based]</span> [Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos](http://openaccess.thecvf.com/content_CVPR_2019/papers/Morais_Learning_Regularity_in_Skeleton_Trajectories_for_Anomaly_Detection_in_Videos_CVPR_2019_paper.pdf), `CVPR 2019`.[code](https://github.com/RomeroBarata/skeleton_based_anomaly_detection)
3. <span id = "01903">[Object-Centric]</span> [Object-Centric Auto-Encoders and Dummy Anomalies for Abnormal Event Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf), `CVPR 2019`.
4. <span id = "01904">[Appearance-Motion Correspondence]</span> [Anomaly Detection in Video Sequence with Appearance-Motion Correspondence](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.pdf), `ICCV 2019`.[code](https://github.com/nguyetn89/Anomaly_detection_ICCV2019)
5. <span id = "01905">[AnoPCN]</span>[AnoPCN: Video Anomaly Detection via Deep Predictive Coding Network](https://people.cs.clemson.edu/~jzwang/20018630/mm2019/p1805-ye.pdf), ACM MM 2019.
### 2020
1. <span id = "02001">[Street-Scene]</span> [Street Scene: A new dataset and evaluation protocol for video anomaly detection](http://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf), `WACV 2020`.
2. <span id = "02002">[Rodrigurs.etl])</span> [Multi-timescale Trajectory Prediction for Abnormal Human Activity Detection](http://openaccess.thecvf.com/content_WACV_2020/papers/Rodrigues_Multi-timescale_Trajectory_Prediction_for_Abnormal_Human_Activity_Detection_WACV_2020_paper.pdf), `WACV 2020`.
3. <span id = "02003">[GEPC]</span> [Graph Embedded Pose Clustering for Anomaly Detection](https://arxiv.org/pdf/1912.11850.pdf), `CVPR 2020`.[code](https://github.com/amirmk89/gepc)
4. <span id = "02004">[Self-trained]</span> [Self-trained Deep Ordinal Regression for End-to-End Video Anomaly Detection](https://arxiv.org/pdf/2003.06780.pdf), `CVPR 2020`. 
5. <span id = "02005">[MNAD]</span> [Learning Memory-guided Normality for Anomaly Detection](https://arxiv.org/pdf/2003.13228.pdf), `CVPR 2020`. [code](https://cvlab.yonsei.ac.kr/projects/MNAD)
6. <span id = "02006">[Continual-AD]]</span> [Continual Learning for Anomaly Detection in Surveillance Videos](https://arxiv.org/pdf/2004.07941),`CVPR 2020 Worksop.`
7. <span id = "02007">[OGNet]</span> [Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.pdf), `CVPR 2020`. [code](https://github.com/xaggi/OGNet)
8. <span id = "02008">[Any-Shot]</span> [Any-Shot Sequential Anomaly Detection in Surveillance Videos](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Doshi_Any-Shot_Sequential_Anomaly_Detection_in_Surveillance_Videos_CVPRW_2020_paper.pdf),`CVPR 2020 workshop`.
9. <span id = "02009">[Few-Shot]</span>[Few-Shot Scene-Adaptive Anomaly Detection](https://arxiv.org/pdf/2007.07843.pdf)`ECCV 2020 Spotlight` [code](https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection)
10. <span id = "02010">[CDAE]</span>[Clustering-driven Deep Autoencoder for Video Anomaly Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600324.pdf)`ECCV 2020`
11. <span id = "02011">[VEC]</span>[Cloze Test Helps: Effective Video Anomaly Detection via Learning to Complete Video Events](https://arxiv.org/abs/2008.11988)`ACM MM 2020 Oral` [code](https://github.com/yuguangnudt/VEC_VAD)
12. <span id ='02012'>[ADOC]</span>[A Day on Campus - An Anomaly Detection Dataset for Events in a Single Camera] `ACCV 2020`
13. <span id ='02013'>[CAC]</span>[Cluster Attention Contrast for Video Anomaly Detection](http://web.pkusz.edu.cn/adsp/files/2020/08/Cluster_Attention_Contrast_for_Video_Anomaly_Detection.pdf) `ACM MM 2020`
14. <span id ='02014'>[STC-Graph]</span>[Scene-Aware Context Reasoning for Unsupervised Abnormal Event Detection in Videos](https://dl.acm.org/doi/pdf/10.1145/3394171.3413887) `ACM MM 2020`

### 2021
1. <span id ='02101'>[AMCM]</span>[Appearance-Motion Memory Consistency Network for Video Anomaly Detection](https://www.aaai.org/AAAI21Papers/AAAI-4120.CaiR.pdf) `AAAI 2021`
2. <span id='02102'>[SSMT,Self-Supervised-Multi-Task]</span>[Anomaly Detection in Video via Self-Supervised and Multi-Task Learning](https://arxiv.org/pdf/2011.07491.pdf) `CVPR 2021`
3. <span id='02103'>[HF2-VAD]</span>[A Hybrid Video Anomaly Detection Framework via Memory-Augmented Flow Reconstruction and Flow-Guided Frame Prediction](https://arxiv.org/pdf/2108.06852.pdf)`ICCV 2021 Oral`
4. <span id='02104'>[ROADMAP]</span>[Robust Unsupervised Video Anomaly Detection by Multipath Frame Prediction](https://arxiv.org/pdf/2011.02763)`TNNLS 2021`
5. <span id='02105'>[AEP]</span>[Abnormal Event Detection and Localization via Adversarial Event Prediction](https://ieeexplore.ieee.org/abstract/document/9346050/) `TNNLS 2021`

### 2022
1. <span id='02201'>[Casual]</span>[A Causal Inference Look At Unsupervised Video Anomaly Detection](https://www.aaai.org/AAAI22Papers/AAAI-37.LinX.pdf)`AAAI 2022`
2. <span id='02202'>[BDPN]</span>[Comprehensive Regularization in a Bi-directional Predictive Network for Video Anomaly Detection](https://www.aaai.org/AAAI22Papers/AAAI-470.ChenC.pdf)`AAAI 2022`
3. <span id='02203'>[GCL]</span>[Generative Cooperative Learning for Unsupervised Video Anomaly Detection](https://arxiv.org/pdf/2203.03962.pdf)`CVPR 2022`

## Weakly-Supervised
### 2018
1. <span id = "11801">[Sultani.etl]</span> [Real-world Anomaly Detection in Surveillance Videos](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf), `CVPR 2018` [code](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)
### 2019
1. <span id = "11901">[GCN-Anomaly]</span> [Graph Convolutional Label Noise Cleaner:Train a Plug-and-play Action Classifier for Anomaly Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf),` CVPR 2019`, 
[code](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection)
2. <span id = "11902">[MLEP]</span> [Margin Learning Embedded Prediction for Video Anomaly Detection with A Few Anomalies](https://pdfs.semanticscholar.org/e878/6acbfabaf4938c9c8e2d3a15e0f110a1ec7f.pdf), `IJCAI 2019`[code](https://github.com/svip-lab/MLEP).
3. <span id = "11903">[IBL]</span> [Temporal Convolutional Network with Complementary Inner Bag Loss For Weakly Supervised Anomaly Detection](https://ieeexplore.ieee.org/abstract/document/8803657/). `ICIP 19`.
4. <span id = "11904">[Motion-Aware]</span> [Motion-Aware Feature for Improved Video Anomaly Detection](https://arxiv.org/pdf/1907.10211). `BMVC 19`.
### 2020
1. <span id = "12001">[Siamese]</span> [Learning a distance function with a Siamese network to localize anomalies in videos](https://arxiv.org/abs/2001.09189), `WACV 2020`.
2. <span id = "12002">[AR-Net]</span> [Weakly Supervised Video Anomaly Detection via Center-Guided Discrimative Learning](https://ieeexplore.ieee.org/document/9102722),` ICME 2020`.[code](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020)
3. <span id ='12003'>['XD-Violence']</span> [Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision](https://arxiv.org/pdf/2007.04687.pdf) `ECCV 2020`
4. <span id ='12004'>[CLAWS]</span> [CLAWS: Clustering Assisted Weakly Supervised Learning with Normalcy Suppression for Anomalous Event Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670358.pdf) `ECCV 2020`
### 2021
1. <span id="12101">[MIST]</span> [MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection](https://arxiv.org/abs/2104.01633) `CVPR 2021` [Project Page](https://kiwi-fung.win/2021/04/28/MIST/)
2. <span id='12102'>[RTFM]</span> [Weakly-supervised Video Anomaly Detection with Contrastive Learning of
Long and Short-range Temporal Features](https://arxiv.org/pdf/2101.10030.pdf) `ICCV 2021`[Code](https://github.com/tianyu0207/RTFM)
3. <spa id='12103'>[STAD]</span>[Weakly-Supervised Spatio-Temporal Anomaly Detection in Surveillance Video](https://arxiv.org/pdf/2108.03825) `IJCAI 2021`
4. <span id='12104'>[WSAL]</span>[Localizing Anomalies From Weakly-Labeled Videos](https://arxiv.org/pdf/2008.08944)`TIP 2021` [Code](https://github.com/ktr-hubrt/WSAL)
5. <span id='12105'>[CRFD]</span>[Learning Causal Temporal Relation and Feature Discrimination for Anomaly Detection](https://ieeexplore.ieee.org/abstract/document/9369126/)`TIP 2021`
### 2022
1. <span id='12201'>[MSL]</span>[Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection](https://www.aaai.org/AAAI22Papers/AAAI-6637.LiS.pdf)`AAAI 2022`

## Supervised
### 2019
1. <span id = "21901">[Background-Bias]</span>[Exploring Background-bias for Anomaly Detection in Surveillance Videos](https://dl.acm.org/doi/abs/10.1145/3343031.3350998), `ACM MM 19`.
2. <span id = "21902">[Ano-Locality]</span>[Anomaly locality in video suveillance](https://arxiv.org/pdf/1901.10364).

## Others
### 2020
1. <span id ="62001">[Few-Shot]</span>[Few-Shot Scene-Adaptive Anomaly Detection](https://arxiv.org/pdf/2007.07843) `ECCV 2020`[code](https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection)
------
## Reviews / Surveys
1. An Overview of Deep Learning Based Methods for Unsupervised and Semi-Supervised Anomaly Detection in Videos, J. Image, 2018.[page](https://beedotkiran.github.io/VideoAnomaly.html)
2. DEEP LEARNING FOR ANOMALY DETECTION: A SURVEY, [paper](https://arxiv.org/pdf/1901.03407.pdf)
3. Video Anomaly Detection for Smart Surveillance [paper](https://arxiv.org/pdf/2004.00222.pdf)
4.  A survey of single-scene video anomaly detection, `TPAMI 2020` [paper](https://arxiv.org/pdf/2004.05993.pdf).


## Books
1. Outlier Analysis. Charu C. Aggarwal
## Specific Scene

------

Generally, anomaly detection in recent researches are based on the datasets from pedestrian (likes UCSD, Avenue, ShanghaiTech, etc.)， or UCF-Crime (real-world anomaly).
However some focus on specific scene as follows.

### Traffic
CVPR  workshop, AI City Challenge series.
#### 	First-Person Traffic
​		Unsupervised Traffic Accident Detection in First-Person Videos, IROS 2019.

#### 	Driving

​		When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos. [github](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)

### Old-man Fall Down

### Fighting/Violence
1. Localization Guided Fight Action Detection in Surveillance Videos. ICME 2019.
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
7. Action Recognition / Temporal Action Localization
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

## Performance Comparison on UCF-Crime 
| Model                                               | Reported on Convference/Journal | Supervised | Feature  | Encoder-based | 32 Segments | AUC (%) | FAR@0.5 on Normal (%) |
| --------------------------------------------------- | ------------------------------- | ---------- | -------- | ------- | ----------- | ------- | --------------------- |
| <span id = "31801">[Sultani.etl](#11801)</span>     | CVPR 18                         | Weakly     | C3D RGB  | X       | √           | 75.41   | 1.9                   |
| <span id = "31903">[IBL](#11903)</span>             | ICIP 19                         | Weakly     | C3D RGB  | X       | √           | 78.66   | -                     |
| <span id = "31904">[Motion-Aware](#11904)</span>    | BMVC 19                         | Weakly     | PWC Flow | X       | √           | 79.0    | -                     |
| <span id = "31901">[GCN-Anomaly](#11901)</span>     | CVPR 19                         | Weakly     | TSN RGB  | √       | X           | 82.12   | 0.1                   |
| <span id = '32013'>[ST-Graph](#02014)</span>        | ACM MM 20                       | Un         | -        | √       | X           | 72.7    |                       |
| <span id = "31902">[Background-Bias](#21901)</span> | ACM MM 19                       | Fully      | NLN RGB  | √       | X           | 82.0    | -                     |
| <span id = "31905">[CLAWS](#12004)</span>           | ECCV 20                         | Weakly     | C3D RGB  | √       | X           | 83.03   | -                     |
| <span id = "32101">[MIST](#12101)</span>            | CVPR 21                         | Weakly     | I3D RGB  | √       | X           | 82.30   | 0.13                  |
| <span id = '32102'>[RTFM](#12102)</span>            | ICCV 21                         | Weakly     | I3D RGB  | X       | √           | 84.03   | -                     |
| <span id = '32104'>[WSAL](#12104)</span>            | TIP 21                          | Weakly     | I3D RGB  | X       | √           | 85.38   | -                     |
| <span id = '32104'>[CRFD](#12105)</span>            | TIP 21                          | Weakly     | I3D RGB  | X       | √           | 84.89   | -                     |
| <span id = '32201_1'>[MSL](#12201)</span>            | AAAI 22                          | Weakly     | C3D RGB  | √        | X           | 82.85   | -                     |
| <span id = '32201_2'>[MSL](#12202)</span>            | AAAI 22                          | Weakly     | I3D RGB  | √        | X           | 85.30   | -                     |
| <span id = '32201_3'>[MSL](#12201)</span>            | AAAI 22                          | Weakly     | VideoSwin-RGB  | √        | X           | 85.62   | -                     |
| <span id = '32203_1'>[GCL](#12203)</span>            | CVPR 22                          | Weakly     | ResNext  | √        | X           | 79.84   | -                     |  
| <span id = '32203_2'>[GCL](#12203)</span>            | CVPR 22                          | Un     | ResNext  | √        | X           | 71.04   | -                     |  
## Performance Comparison on ShanghaiTech
| Model                                             | Reported on Conference/Journal | Supervision                   | Feature            | Encoder-based | AUC(%) | FAR@0.5 (%) |
| ------------------------------------------------- | ------------------------------ | ----------------------------- | ------------------ | ------- | ------ | ----------- |
| <span id = "41601">[Conv-AE](#01601)</span>       | CVPR 16                        | Un                            | -                  | √       | 60.85  | -           |
| <span id = "41702">[stacked-RNN](#01702)</span>   | ICCV 17                        | Un                            | -                  | √       | 68.0   | -           |
| <span id = "41801">[FramePred](#01801)</span>     | CVPR 18                        | Un                            | -                  | √       | 72.8   | -           |
| <span id = "41902">[FramePred*](#11902)</span>    | IJCAI 19                       | Un                            | -                  | √       | 73.4   | -           |
| <span id = "41901-1">[Mem-AE](#01901)</span>      | ICCV 19                        | Un                            | -                  | √       | 71.2   | -           |
| <span id = "42005">[MNAD](#02005)</span>          | CVPR 20                        | Un                            | -                  | √       | 70.5   | -           |
| <span id = "42011">[VEC](#02011)</span>           | ACM MM 20                      | Un                            | -                  | √       | 74.8   | -           |
| <span id ='42014'>[ST-Graph](#02014)</span>       | ACM MM 20                      | Un                            | -                  | √       | 74.7   | -           |
| <span id = '42013'>[CAC](#02013)</span>           | ACM MM 20                      | Un                            | -                  | √       | 79.3   |             |
| <span id='42101'>[AMMC](#02101)</span>            | AAAI 21                        | Un                            | -                  | √       | 73.7   | -           |
| <span id='42102'>[SSMT](#02102)</span>            | CVPR 21                        | Un                            | -                  | √       | 90.2   | -           |
| <span id='42103'>[HF2-VAD](#02103)</span>         | ICCV 21                        | Un                            | -                  | √       | 76.2   | -           |
| <span id='42104'>[ROADMAP](#02104)</span>         | TNNLS 21                       | Un                            | -                  | √       | 76.6   | -           |
| <span id='42202'>[BDPN](#02202)</span>         | AAAI 22                       | Un                            | -                  | √       | 78.1   | -           |
| <span id = "41902-1">[MLEP](#11902)</span>        | IJCAI 19                       | 10% test vids with Video Anno | -                  | √       | 75.6   | -           |
| <span id = "41902-2">[MLEP](#11902)</span>        | IJCAI 19                       | 10% test vids with Frame Anno | -                  | √       | 76.8   | -           |
| <span id = "42002-1">[Sultani.etl](#12002)</span> | ICME 2020                      | Weakly (Re-Organized Dataset) | C3D-RGB            | X       | 86.3   | 0.15        |
| <span id = "42002-2">[IBL](#12002)</span>         | ICME 2020                      | Weakly (Re-Organized Dataset) | I3D-RGB            | X       | 82.5   | 0.10        |
| <span id = "41901-2">[GCN-Anomaly](#11901)</span> | CVPR 19                        | Weakly (Re-Organized Dataset) | C3D-RGB            | √       | 76.44  | -           |
| <span id = "41901-3">[GCN-Anomaly](#11901)</span> | CVPR 19                        | Weakly (Re-Organized Dataset) | TSN-Flow           | √       | 84.13  | -           |
| <span id = "41901-4">[GCN-Anomaly](#11901)</span> | CVPR 19                        | Weakly (Re-Organized Dataset) | TSN-RGB            | √       | 84.44  | -           |
| <span id = "42002">[AR-Net](#12002)</span>        | ICME 20                        | Weakly (Re-Organized Dataset) | I3D-RGB & I3D Flow | X       | 91.24  | 0.10        |
| <span id = "42002">[CLAWS](#12004)</span>         | ECCV 20                        | Weakly (Re-Organized Dataset) | C3D-RGB            | √       | 89.67  |             |
| <span id='42101'>[MIST](#12101)</span>            | CVPR 21                        | Weakly (Re-Organized Dataset) | I3D-RGB            | √       | 94.83  | 0.05        |
| <span id='42102'>[RTFM](#12102)</span>            | ICCV 21                        | Weakly (Re-Organized Dataset) | I3D-RGB            | X       | 97.21  | -           |
| <span id='42102'>[CRFD](#12105)</span>            | TIP 21                         | Weakly (Re-Organized Dataset) | I3D-RGB            | X       | 97.48  | -           |
| <span id='42201_0'>[MSL](#12201)</span>            | AAAI 22                        | Weakly (Re-Organized Dataset) | C3D-RGB            | X       | 94.81  | -      |
| <span id='42201_1'>[MSL](#12201)</span>            | AAAI 22                        | Weakly (Re-Organized Dataset) | I3D-RGB            | X       | 96.08  | -      |
| <span id='42201_1'>[MSL](#12201)</span>            | AAAI 22                        | Weakly (Re-Organized Dataset) | VideoSwin-RGB            | X       | 97.32  | -      |
| <span id='42203_1'>[GCL](#12203)</span>            | CVPR 22                        | Weakly (Re-Organized Dataset) | ResNext           | X       | 86.21  | -      |
| <span id='42203_2'>[GCL](#12203)</span>            | CVPR 22                        | Un | ResNext           | X       | 78.93  | -      |

## Performance Comparison on Avenue 
| Model                                                        | Reported on Conference/Journal | Supervision                   | Feature                | End2End | AUC(%) |
| ------------------------------------------------------------ | ------------------------------ | ----------------------------- | ---------------------- | ------- | ------ |
| <span id = "51601">[Conv-AE](#01601)</span>                  | CVPR 16                        | Un                            | -                      | √       | 70.2   |
| <span id = "51601-2">[Conv-AE*](#01801)</span>               | CVPR 18                        | Un                            | -                      | √       | 80.0   |
| <span id = "51703">[ConvLSTM-AE](#01703)</span>              | ICME 17                        | Un                            | -                      | √       | 77.0   |
| <span id = "51706">[DeepAppearance](#01706)</span>           | ICAIP 17                       | Un                            | -                      | √       | 84.6   |
| <span id = "51705">[Unmasking](#01705)</span>                | ICCV 17                        | Un                            | 3D gradients+VGG conv5 | X       | 80.6   |
| <span id = "51702">[stacked-RNN](#01702)</span>              | ICCV 17                        | Un                            | -                      | √       | 81.7   |
| <span id = "51801">[FramePred](#01801)</span>                | CVPR 18                        | Un                            | -                      | √       | 85.1   |
| <span id = "51901-1">[Mem-AE](#01901)</span>                 | ICCV 19                        | Un                            | -                      | √       | 83.3   |
| <span id = "51904">[Appearance-Motion Correspondence](#01904) </span> | ICCV 19               | Un                            | -                      | √       | 86.9   |
| <span id = "51902">[FramePred*](#11902)</span>               | IJCAI 19                       | Un                            | -                      | √       | 89.2   |
| <span id = "52005">[MNAD](#02005)</span>                     | CVPR 20                        | Un                            | -                      | √       | 88.5   |
| <span id = "52011">[VEC](#02011)</span>                      | ACM MM 20                      | Un                            | -                      | √       | 90.2   |
| <span id = '52014'>[ST-Graph](#02014)</span>                 | ACM MM 20                      | Un                            | -                      | √       | 89.6   |
| <span id = '52013'>[CAC](#02013)</span>                      | ACM MM 20                      | Un                            | -                      | √       | 87.0   |
| <span id='52101'>[AMMC](#02101)</span>                       | AAAI 21                        | Un                            | -                      | √       | 86.6   |
| <span id='52102'>[SSMT](#02102)</span>                       | CVPR 21                        | Un                            | -                      | √       | 92.8   |
| <span id='52103'>[HF2-VAD](#02103)</span>                    | ICCV 21                        | Un                            | -                      | √       | 91.1   |
| <span id='52104'>[ROADMAP](#02104)</span>                    | TNNLS 21                       | Un                            | -                      | √       | 88.3   |
| <span id='52105'>[AEP](#02105)</span>                        | TNNLS 21                       | Un                            | -                      | √       | 90.2   |
| <span id='52201'>[Causal](#02201)</span>                        | AAAI 22                       | Un                            | I3D-RGB                     | X       | 90.3   |
| <span id='52202'>[BDPN](#02202)</span>                        | AAAI 22                       | Un                            | -                    |  √     | 90.3   |
| <span id = "51801-1">[MLEP](#11902)</span>                   | IJCAI 19                       | 10% test vids with Video Anno | -                      | √       | 91.3   |
| <span id = "51801-2">[MLEP](#11902)</span>                   | IJCAI 19                       | 10% test vids with Frame Anno | -                      | √       | 92.8   |

## Performance Comparison on XD-Violence 
| Model                                                 | Reported on Conference/Journal | Supervision              | Feature             | Encoder-based | 32 Segments | AP(%)  |
| ----------------------------------------------------- | ------------------------------ | ------------------------ | ------------------- | ------- |-------------| ------ |
| <span id='61801'>[Sultani et al.](#11801)</span>      | ECCV 2020 (reported by Wu)     | Weakly                   | I3D-RGB             | X       |   √         | 73.20  |     
| <span id='62003'>[Wu et al.](#12003)</span>           | ECCV 2020                      | Weakly                   | C3D-RGB             | X       |   X         | 67.19  |
| <span id='62003-1'>[Wu et al.](#12003)</span>         | ECCV 2020                      | Weakly                   | I3D-RGB+Audio       | X       |   X         | 78.64  |
| <span id = "62102">[RTFM](#12102)</span>              | ICCV 2021                      | Weakly                   | I3D-RGB             | X       |   √         | 77.81  |
| <span id = "62105">[CRFD](#12105)</span>              | TIP 2021                       | Weakly                   | I3D-RGB             | X       |   √         | 75.90  |
| <span id = "62201_0">[MSL](#12201)</span>              | AAAI 2022                       | Weakly                   | C3D-RGB             | X       |    X         | 75.53  |
| <span id = "62201_1">[MSL](#12201)</span>              | AAAI 2022                       | Weakly                   | I3D-RGB             | X       |    X         | 78.28  |
| <span id = "62201_2">[MSL](#12201)</span>              | AAAI 2022                       | Weakly                   | VideoSwin-RGB             | X       |    X         | 78.59  |

