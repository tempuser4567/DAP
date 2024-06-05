# Deepfake Detector Assessment Platform (DAP)

We have set up a project for evaluating deepfake detection. Users can upload detection algorithms, and the project will assess the performance of these algorithms in various aspects.

The main evaluation features of the project are shown in the following figure.

![image1](/png/1.png)


## Dataset Introduction:

The dataset is primarily composed of three parts.: public datasets, self-constructed datasets and specially processed dataset. The public datasets include **11 open-source datasets**. 

The self-constructed dataset is generated using deepfake algorithms and includes 3 face-swapping algorithms, 2 facial reenactment algorithms, 2 facial attribute manipulation algorithms, 4 entire face synthesis algorithms, 4 text-to-image algorithms, and 1 image-to-video algorithm, totaling **16 deepfake algorithms**. 

The specially processed dataset is primarily generated using various **data augmentation and adversarial evasion techniques**. It is used to evaluate the robustness and adversarial attack resilience of detection algorithms.


### Public Dataset Introduction:

We downloaded 11 datasets from open-source datasets to be used as public datasets. These public datasets include data from different deepfake algorithms, genders, ethnicities, expressions, lighting conditions, and camera angles, which can be used to evaluate the performance of detection algorithms in these specific areas.

| Index | Name                |
| :---- | :------------------ |
| 1     | FaceForensics++     |
| 2     | Celeb-DF v1         |
| 3     | Celeb-DF v2         |
| 4     | FakeAVCeleb         |
| 5     | DeeperForensics-1.0 |
| 6     | DFFD                |
| 7     | CelebA              |
| 8     | VGGface2            |
| 9     | VidTIMIT            |
| 10    | DFDC                |
| 11    | Lav-DF              |

The different attribute images included in the public dataset are as follows:

![image2](/png/2.png)

### Self-constructed Dataset Introduction:

The self-constructed datasets include data generated by diverse deepfake algorithms, which can be used to assess the generalization capability of detection algorithms.
The number of images generated by the forgery algorithm is shown in the table below.

| Index | Category                      | Algorithm              | Number    |
| ----- | ----------------------------- | ---------------------- | --------- |
| 1     | Face Swapping                 | FaceShifter            | 596,416   |
| 2     | Face Swapping                 | MobileFaceSwap         | 670,682   |
| 3     | Face Swapping                 | FaceDancer             | 606,953   |
| 4     | Facial Reenactment            | HyperReenact           | 941,082   |
| 5     | Facial Reenactment            | DGFR                   | 576,142   |
| 6     | Facial Attribute Manipulation | StarGAN2               | 420,000   |
| 7     | Facial Attribute Manipulation | STGAN                  | 1,215,821 |
| 8     | Entire Face Synthesis         | ProGAN                 | 100,000   |
| 9     | Entire Face Synthesis         | StyleGAN2              | 100,000   |
| 10    | Entire Face Synthesis         | StyleGAN3              | 100,000   |
| 11    | Entire Face Synthesis         | StyleGANV              | 240,000   |
| 12    | Text-to-Image                 | Stable_Diffusion       | 120,552   |
| 13    | Text-to-Image                 | Mini-DALLE             | 82,982    |
| 14    | Text-to-Image                 | LDM                    | 4,445     |
| 15    | Text-to-Image                 | Mini-DALLE3            | 100,000   |
| 16    | Image-to-Video                | Stable Video Diffusion | 101,070   |


### Specially Processed Dataset Introduction:

To test the robustness of the detection algorithm on augmented data, We have implemented several data augmentation algorithms, such as adjusting image brightness, contrast, hue, sharpness, rotation, blurring, and occlusion, to evaluate the robustness of detection algorithms. 
</br>
</br>
Additionally, we have configured adversarial (adding adversarial noise) and evasion (image reconstruction) algorithms to assess the security of detection algorithms when faced with malicious adversarial and evasion techniques.

![image3](/png/3.png)

### Data and Label Organization Structure: 

Our data consists of three levels: **video-level, frame-level, and face-level.** The latter two are obtained by extracting frames and faces from the original videos, respectively, and are used to evaluate detection algorithms that target different inputs. The overall data structure is as follows:

```text
datasets
├── FaceForensics++
│ ├── original_sequences
│ │ ├── youtube
│ │ │ ├── c23
│ │ │ │ ├── videos
│ │ │ │ │ └── *.mp4
│ │ │ │ └── frames
│ │ │ │ │ └── *.png
│ │ │ │ └── faces
│ │ │ │ │ └── *.png
│ │ │ └── c40
│ │ │ │ ├── ...
│ │ ├── actors
│ │ │ ├── ...
│ ├── manipulated_sequences
│ │ ├── Deepfakes
│ │ │ ├── c23
│ │ │ │ └── videos
│ │ │ │ │ └── *.mp4
│ │ │ │ └── frames
│ │ │ │ │ └── *.png
│ │ │ └── c40
│ │ │ │ ├── ...
│ │ ├── Face2Face
│ │ │ ├── ...
│ │ ├── FaceSwap
│ │ │ ├── ...
│ │ ├── NeuralTextures
│ │ │ ├── ...
│ │ ├── FaceShifter
│ │ │ ├── ...
│ │ └── DeepFakeDetection
│ │ ├── ...
```
Other datasets are similar to the above structure



Our label structures are as follows:

```
datasets
├── FaceForensics++
│ ├── videos
│ │ ├── attribute
│ │ │ └── FaceForensics++_fake_DeepFakeDetection_c23_videos.txt
│ │ │ └── FaceForensics++_real_actors_c23_videos.txt
│ │ │ └── ...
│ │ └── FaceForensics++_real_videos.txt
│ │ └── FaceForensics++_fake_videos.txt
│ ├── frames
│ │ ├── ...
│ ├── faces
│ │ ├── ...
```

The attributes folder contains data labels that provide detailed categorization based on specific attributes such as forgery methods, ethnicity, gender, and other characteristics.


### The Dataset and Related Test Content:

For evaluating detection algorithms, we select 10,000 real images and 10,000 forged images from the corresponding datasets for result testing.

The specific test items and the corresponding datasets are shown in the table below:

| Test Content                                | Dataset                                   |
| ------------------------------------------- | ----------------------------------------- |
| Benchmark Performance Evaluation            | All public datasets                       |
| Attribute Bias Assessment                   | Datasets with corresponding label files   |
| Forgery Algorithm Generalization Assessment | All self-constructed datasets             |
| Image Distortion Robustness Assessment      | Robustness processed dataset              |
| Adversarial Attack Resilience Evaluation    | Adversarial and evasion processed dataset |
| Forgery Localization Accuracy Evaluation    | Datasets with corresponding label files   |

The test content and metrics are as follows:

**1、Benchmark Performance Evaluation:** The test evaluates the basic performance metrics of detection algorithms on a general dataset, including AUC, ACC, EER, F1/F2-score, confidence, and other indicators.

**2、Attribute Bias Assessment:** The test assesses the detection capability on datasets with specific attributes.   (??? test metrics)

**3、Forgery Algorithm Generalization Assessment:** The test evaluates the generalization capability of detection algorithms.

**4、Image Distortion Robustness Assessment:** The test assesses the robustness of detection algorithms when faced with traditional data modifications and enhancements.

**5、Adversarial Attack Resilience Evaluation:** The test evaluates the detection capability of algorithms when facing malicious evasion and attack forgery data.

**6、Forgery Localization Accuracy Evaluation:** The test evaluates the special functionalities of detection algorithms, if they exist, such as forged region localization, fragment forgery detection, etc.



## Algorithms Introduction

We have configured some forgery algorithms and detection algorithms. Forgery algorithms are used to generate data (see the introduction in the self-constructed dataset section), while detection algorithms are used to test metrics and provide baselines for existing detection algorithms. Currently, we have configured 11 detection algorithms, listed as follows.

| ID   | name               | function                             |
| ---- | ------------------ | ------------------------------------ |
| 1    | Xception           | Detection of forged face-level data  |
| 2    | SRM                | Detection of forged face-level data  |
| 3    | SBI                | Detection of forged face-level data  |
| 4    | DSP_FWA            | Detection of forged face-level data  |
| 5    | Multiple-attention | Detection of forged face-level data  |
| 6    | SeqDeepFake        | Detection of forged face-level data  |
| 7    | SLADD              | Detection of forged face-level data  |
| 8    | CADDM              | Detection of forged frame-level data |
| 9    | Multiple-attention | Detection of forged video-level data |
| 10   | ClassNSeg          | Detecting forged regions             |
| 11   | BA-TFD             | Detecting forged segments            |


## How to Start

**1、Installation**

Run the following script to install necessary environment:

```
conda create -f env.yaml -n DAP
```

or create your own conda virtual environment and run:

```
pip install -r requirements.txt
 ```
 
**2、Evaluation**

First you need to put your deepfake detection model in folder './user'.

Then, a base Docker image is needed to create the Docker container required for running the model. You can apply for the image's tar file by filling out this form( https://forms.gle/c3HnnpvQWstYzrdc8 ). Once you have received the tar file, you can import the base image using the following command:

```
docker load -i base_docker.tar
```
 
Finally, run the following command:

```
python backend_api_video.py
```

Inside the backend_api_video.py file, the following key processes will be executed:

Debugging the Algorithm: This step involves running a debug process to ensure that scripts of the model are functioning correctly.

Running the Evaluation: This step initiates the evaluation process for your deepfake detection model.

Computing the Metrics: This step calculates the evaluation metrics to assess the performance of your model.

Each of these steps includes breakpoints for detailed inspection and debugging, ensuring the evaluation process is thorough and accurate.

## Presentation of Results

Here we present some test results.

![image6](/png/6.png)

<div align='center'>Benchmark_Performance_Evaluation</div>
<br/>
<br/>
<br/>

![image4](/png/4.png)

<div align='center'>Lighting Condition of Attribute Bias</div>
<br/>
<br/>
<br/>

![image5](/png/5.png)

<div align='center'>Ethnic Group of Attribute Bias</div>
<br/>
<br/>
<br/>

![image7](/png/7.png)

<div align='center'>Forgery_Algorithm_Generalization_Evaluation</div>
<br/>
<br/>
<br/>

![image8](/png/8.png)

<div align='center'>Image_Distortion_Robustness_Evaluation_(Compression)</div>
<br/>
<br/>
<br/>

![image10](/png/10.png)

<div align='center'>Adversarial_Attack_Resilience_Evaluation_(Adversarial_Perturbation)</div>
<br/>
<br/>
<br/>

![image11](/png/11.png)

<div align='center'>Forgery Localization Accuracy Evaluation</div>
<br/>
<br/>
<br/>

## Instructions for Use

