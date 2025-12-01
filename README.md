# EASE
EEG Analysis for Seizure Evaluation – A Deep Transfer Learning Approach

This project uses the CHB-MIT Scalp EEG Database
Available at: https://physionet.org/content/chbmit/

### A Deep Transfer-Learning Framework with XAI and LLM Reporting

EASE is an EEG-based seizure detection system integrating deep learning, transfer learning, explainable AI (XAI), and LLM-generated clinical-style summaries. The system processes raw EEG, performs standardized preprocessing, extracts time–frequency features, trains multiple model types, and generates interpretable reports.

---

## Project Structure

```
1d model/
  architecture.py   # 1D CNN model
  trainer.py        # Training loop for 1D CNN

2dcnns/
  architecture_spectrograms.py  # 2D CNN for spectrogram images
  trainer.py                    # Training script for 2D CNN

HPO_&_XAI_LLM_reporting/
  effllm_optuna.py             # Optuna + LLM hyperparameter optimization
  xai_llm_report.py            # Grad-CAM + IG + LLM report generation


TL/                     # Transfer Learning models
  architecture_eff.py   # EfficientNet-B0 + LSTM
  architecture_mob.py   # MobileNetV2 + LSTM
  architecture_res.py   # ResNet34 on spectrograms
  train_eff.py          # Training script for EfficientNet-B0 + LSTM
  train_mob.py          # Training script for MobileNet + LSTM
  train_res.py          # Training script for ResNet
  trainer.py            # Shared TL trainer utilities

README.md
```

---

## Dataset

CHB-MIT Scalp EEG Dataset

* 23 pediatric patients
* 21–26 channels, 256 Hz sampling
* Segmented into 4-second windows (1-second overlap)

---

## Preprocessing

* Butterworth bandpass filtering (0.5–40 Hz)
* Wavelet denoising (db4)
* Fixed-window segmentation
* Labeling via seizure onset/offset annotations

---

## Feature Extraction

* Spectrograms (STFT)
* Gramian Angular Field (GAF) images

---

## Models Implemented

* **1D CNN** for raw EEG
* **2D CNN** for spectrograms
* **Transfer Learning Models:**

  * EfficientNet-B0 + LSTM
  * MobileNetV2 + LSTM
  * ResNet-34

---

## Explainable AI (XAI)

* Grad-CAM for spatial-frequency activation
* Integrated Gradients for pixel-level attribution
* Combined XAI digest summarizing:

  * Dominant frequency bands
  * Temporal evolution
  * Key EEG channels
  * IG–GradCAM agreement metrics

---

## LLM Clinical Report Generation

* Uses **Gemini Flash 2.0 API**
* Generates structured clinical-style summaries
* Includes simplified patient-friendly interpretation
* Safety disclaimer included (non-diagnostic)

---

## Hyperparameter Optimization

* Optuna with TPE sampler
* LLM-guided hyperparameter proposals (JSON format)
* Objective: maximize

  ```
  0.5 × Accuracy + 0.5 × Sensitivity
  ```

---

## Best Results (EfficientNet-B0 + LSTM, after LLM-guided Optuna optimization)

* Accuracy: **98.27%**
* Sensitivity: **97.98%**
* Specificity: **98.42%**

---

## Authors

Jana Ramzi Ali, Shafaa Omer Al-Baiti, Ayesha Siddiqua, Nadia Mohamed Elsayed

Supervised by: Prof. Abbes Amira and Co-supervised by: Dr. Ayad Turky
