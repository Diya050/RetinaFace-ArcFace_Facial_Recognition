# Real-Time Facial Recognition using RetinaFace + ArcFace (InsightFace)

This project implements a real-time face recognition system designed to evaluate and benchmark modern face detection and recognition pipelines. It implements a complete Proof-of-Concept (POC) using **RetinaFace for face detection** and **ArcFace for face recognition**, powered by the **InsightFace framework**.

## Features

* High-accuracy face detection using RetinaFace
* Discriminative face embeddings using ArcFace
* Real-time webcam-based recognition
* Evaluation on test images with detailed metrics
* ROC, AUC, FAR, FRR, EER and confusion matrix analysis
* FPS and latency measurement
* Modular architecture suitable for research and production prototyping

## System Pipeline

```
Input Image / Webcam
↓
RetinaFace
↓
Face Alignment
↓
ArcFace (via InsightFace)
↓
Embedding Matching
↓
Identity Prediction + Metrics
```

## Project Structure

```
RetinaFace-ArcFace_Facial_Recognition/
│
├── images_dataset/
│     │
│     ├── train_images/         # Known identities (training images)
│     │      ├── Person1/
│     │      │   ├── 1.jpg
│     │      │   ├── 2.jpg
│     │      │
│     │      ├── Person2/
│     │            ├── 1.jpg
│     │            ├── 2.jpg
│     │
│     ├── test_images/          # Testing dataset
│             ├── Person1/
│             │   ├── 1.jpg
│             │   ├── 2.jpg
│             │
│             ├── Person2/
│                  ├── 1.jpg
│                  ├── 2.jpg
│
│
├── main.py               # Complete pipeline + evaluation
├── requirements.txt
├── README.md
└── .gitignore

```

## Dataset Description

The dataset used in this project is a **custom face dataset** organized by identity folders.

### Characteristics

* Frontal and side faces
* Multiple facial expressions
* Different lighting conditions
* Medium occlusion (glasses, partial face visibility)
* Realistic variations similar to real-world scenarios

### Dataset Split

* Training: **5 images per person**
* Testing: **15 images per person**

This setup evaluates the system’s ability to generalize across pose and illumination changes.

## Evaluation Metrics

| Metric          | Description                          |
| --------------- | ------------------------------------ |
| Accuracy        | Overall recognition correctness      |
| Precision       | Correct positive predictions         |
| Recall          | True positive detection rate         |
| F1 Score        | Balance between precision and recall |
| FAR             | False Acceptance Rate                |
| FRR             | False Rejection Rate                 |
| ROC Curve       | Threshold vs performance             |
| AUC             | Area under ROC                       |
| EER             | Equal Error Rate                     |
| Rank-1 Accuracy | Identification accuracy              |
| FPS             | Frames processed per second          |
| Latency         | Total per-frame processing time      |


## Getting Started

### Prerequisites

Ensure the following are installed on your system:

* Windows 10 / 11 (64-bit)
* Python **3.11.x**
* Microsoft C++ Build Tools (required for some dependencies)


### Step 1: Install Python 3.11

Download Python 3.11 from the official site:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

During installation:

* Check **“Add Python to PATH”**
* Select **Customize installation → Install for all users**

Verify installation:

```powershell
py -3.11 --version
```

### Step 2: Create Virtual Environment

Navigate to your project folder:

```powershell
cd path\to\your\project
```

Create virtual environment:

```powershell
py -3.11 -m venv facetrack_env
```

Activate:

```powershell
facetrack_env\Scripts\activate
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```


### Step 3: Install Dependencies

Install required packages:

```powershell
pip install -r requirements.txt
```

### Step 4: Running the Pipeline

Activate environment:

```powershell
facetrack_env\Scripts\activate
```

Run:

```powershell
python main.py
```



## Outputs and Visualizations

<p align="center">
<img width="600" height="450" alt="Screenshot 2026-02-14 201655" src="https://github.com/user-attachments/assets/de51ea13-859a-4914-9499-da6107415b2d" />

<img width="600" height="400" alt="Screenshot 2026-02-14 201836" src="https://github.com/user-attachments/assets/0f7e122c-5348-4ee9-83a7-dd780249f4fe" />

<img width="600" height="400" alt="Screenshot 2026-02-14 201941" src="https://github.com/user-attachments/assets/5a475d6b-bd7e-4f5f-9c66-77d622df264d" />

<img width="620" height="550" alt="Screenshot 2026-02-14 202101" src="https://github.com/user-attachments/assets/4d74bf92-ed6a-414f-8314-e91a0019b59b" />

<img width="400" height="600" alt="Screenshot 2026-02-14 202210" src="https://github.com/user-attachments/assets/6205f8ed-6ae1-4529-bbb2-57679887b3ce" />
</p>

* **ROC Curve:**

<p align="center">
<img width="600" height="450" alt="Screenshot 2026-02-14 202904" src="https://github.com/user-attachments/assets/879048bb-386c-4ced-9b7f-43184621105e" />
</p>

* **FAR vs FRR Curve:**

<p align="center">
<img width="600" height="450" alt="Screenshot 2026-02-14 202924" src="https://github.com/user-attachments/assets/0e3f0675-78a3-4ceb-a5d5-949f8ae1bf3c" />
</p>
The FAR vs FRR curve illustrates how the system’s error rates change with different similarity thresholds. FAR represents the rate at which the system incorrectly accepts an imposter, while FRR represents the rate at which it incorrectly rejects a genuine user. In this experiment, both error rates approach zero near the optimal threshold, indicating that the model can clearly distinguish between identities on the given dataset. This demonstrates strong verification performance and a well-separated embedding space.
<br>
<br>


* **Confusion Matrix Heatmap:**

<p align="center">
  <img width="600" height="450" alt="Screenshot 2026-02-14 202914" src="https://github.com/user-attachments/assets/068c57a6-27ee-4a31-8f87-52ee997df31a" />
</p>

* **FPS and latency statistics:**

<p align="center">
  <img width="400" height="250" alt="Screenshot 2026-02-14 203052" src="https://github.com/user-attachments/assets/3776ef4e-e016-4ac3-85c8-ab8fd4d2c7a6" />

<img width="1520" height="536" alt="Screenshot 2026-02-14 203130" src="https://github.com/user-attachments/assets/3842f86a-59ab-42cb-a9ac-fdaa86e98e52" />
</p>


## Results

### Evaluation Summary

The system was evaluated on **test images**, and metrics were computed by comparing predicted identities with ground truth labels.

Despite pose variations, lighting changes, and occlusion, the model maintains strong recognition performance, demonstrating robustness of the embedding space.

### Metrics

* Accuracy: 100%
* Precision: 100%
* Recall: 100%
* F1 Score: 100%
* AUC: 1.00
* FAR: 0.00
* FRR: 0.00
* Rank-1 Accuracy: 88.46%

Perfect verification scores indicate clear separation between genuine and imposter pairs, while Rank-1 reflects multi-class identification difficulty.

## Performance Analysis

Testing conducted on **CPU only**.

### Latency Breakdown (Per Frame)

* Face Detection Time: ~0.32 s
* Embedding Extraction Time: ~0.21 s
* Matching Time: ~0.02 s
* Total Pipeline Latency (P50): ~0.66 s

### Latency Distribution

* P95 Latency: ~0.73 s
* P99 Latency: ~0.83 s

### Throughput

* Average FPS: ~1.5

This breakdown highlights that **face detection is the dominant computational cost**, which is expected in CPU-only inference.

## Expected GPU Performance

Using GPU acceleration typically yields:

* 5× – 10× higher FPS
* <=200 ms latency
* Smooth real-time experience

## Why RetinaFace + ArcFace (via InsightFace)?

| Model       | Reason                                   |
| ----------- | ---------------------------------------- |
| RetinaFace  | Robust detection with accurate alignment |
| ArcFace     | Highly discriminative embeddings         |
| InsightFace | Optimized production-ready framework     |

## Conclusion

* The end-to-end pipeline operates reliably.
* The system handles pose, lighting, and occlusion variations effectively.
* Verification performance is near perfect on the test dataset.
* CPU inference is near real-time, while GPU deployment can enable production-level performance.
