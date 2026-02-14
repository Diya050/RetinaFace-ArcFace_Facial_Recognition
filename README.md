# Real-Time Facial Recognition using RetinaFace + ArcFace (InsightFace)  

This project implements a real-time face recognition system designed to evaluate and benchmark modern face detection and recognition pipelines. It implements a complete Proof-of-Concept (POC) using **RetinaFace for face detection** and **ArcFace for face recognition**, powered by the **InsightFace framework**.

## Features

- High-accuracy face detection using RetinaFace  
- Discriminative face embeddings using ArcFace  
- Real-time webcam-based recognition  
- Offline benchmarking and evaluation  
- ROC, AUC, FAR, FRR, EER and confusion matrix analysis  
- FPS and latency measurement  
- Modular architecture suitable for research and production prototyping  

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

FaceTrack/
│
├── train_images/         # Known identities (training images)
│   ├── Person1/
│       ├──1.jpg
│       ├──2.jpg
|
│   ├── Person2/
│       ├──1.jpg
│       ├──2.jpg
│
├── test_images/          # Testing dataset
│   ├── Person1/
│       ├──1.jpg
│       ├──2.jpg
|
│   ├── Person2/
│       ├──1.jpg
│       ├──2.jpg
│
├── main.py               # Complete pipeline + evaluation framework
├── requirements.txt
├── README.md

```



## Installation Guide (Windows)

### Python Version

Use:

```
Python 3.9.13 (recommended)
```

Avoid Python 3.11 and 3.12 due to limited compatibility with current computer vision and deep learning libraries.



### Virtual Environment Setup

```bash
python -m venv facetrack_env
facetrack_env\Scripts\activate
````

Upgrade pip:

```bash
python -m pip install --upgrade pip
```


## InsightFace Installation (For Windows)

### Installation Requirement

```
Microsoft Visual C++ 14.0 or greater is required
```

If you want to use the latest versions of InsightFace, install the Microsoft C++ Build Tools from:

[https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

During installation, select: `Desktop Development with C++` 
and in right panel select:

* MSVC v143 compiler
* Windows SDK
* C++ CMake tools

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python main.py
```

Press **ESC** to exit the real-time recognition window.

## Evaluation Metrics

The system computes the following metrics:

| Metric          | Description                           |
|  -- |--
| Accuracy        | Overall recognition correctness       |
| Precision       | Correct positive predictions          |
| Recall          | True positive detection rate          |
| F1 Score        | Harmonic mean of precision and recall |
| FAR             | False Acceptance Rate                 |
| FRR             | False Rejection Rate                  |
| ROC Curve       | Performance curve                     |
| AUC             | Area under ROC                        |
| EER             | Equal Error Rate                      |
| Rank-1 Accuracy | Identification accuracy               |
| FPS             | Frames per second                     |
| Latency         | Per-frame processing delay            |



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


## Why RetinaFace + ArcFace (via InsightFace)?

| Model       | Reason                                              |
| -- | -- |
| RetinaFace  | High detection accuracy and robust face alignment   |
| ArcFace     | Highly discriminative face embeddings               |
| InsightFace | Industry-grade optimized face recognition framework |


## Results

### Evaluation Summary

The system was tested on a small dataset:

* Training set: 5 images per person
* Test set: 15 images per person

Because the dataset is controlled and consistent, the model achieves very high recognition performance and correctly identifies the registered users.

### **Metrics**

* Accuracy: 100%
* Precision: 100%
* Recall: 100%
* F1-Score: 100%
* AUC: 1.00
* FAR (False Accept Rate): 0.00
* FRR (False Reject Rate): 0.00
* Rank-1 Accuracy: 88.46%

Perfect classification scores indicate strong verification performance, while Rank-1 accuracy reflects identification capability across multiple identities.


### Performance (Speed)

Testing was conducted on **CPU only**.

* Average FPS: ~1.5
* Average Latency (P50): ~0.66 s
* P95 Latency: ~0.73 s
* P99 Latency: ~0.83 s

Even without GPU acceleration, the system runs **near real-time**, which demonstrates efficient pipeline design. Performance is suitable for demos, prototypes, and low-scale deployments.



### Expected GPU Performance

With a compatible GPU, inference speed typically improves by **5× to 10×**, enabling smooth real-time recognition with significantly lower latency.



### Conclusion

* The pipeline operates reliably end-to-end.
* Recognition performance is strong on the test dataset.
* CPU performance is already near real-time.
* GPU deployment can further enhance responsiveness for production scenarios.
