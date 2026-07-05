# Human Activity Recognition Using Hidden Markov Models

**Formative 2 — Machine Learning Techniques**

This project builds a Hidden Markov Model from scratch to classify four physical
activities using real smartphone sensor data collected with Sensor Logger on an iPhone 13 Pro.

## Project Overview

- **Activities:** Standing, Walking, Jumping, Still
- **Device:** iPhone 13 Pro
- **Sampling Rate:** 99.3 Hz
- **Windows:** 1 second, non-overlapping (99 samples per window)
- **Features:** 32 per window (time-domain + FFT frequency-domain)
- **Model:** Gaussian HMM with 4 hidden states, built from scratch in numpy

## Results

Tested on 8 unseen recordings (41 windows) held out before any training.

| Activity | Sensitivity | Specificity | Overall Accuracy |
|---|---|---|---|
| Standing | 1.000 | 1.000 | 1.000 |
| Walking | 1.000 | 1.000 | 1.000 |
| Jumping | 1.000 | 1.000 | 1.000 |
| Still | 1.000 | 1.000 | 1.000 |

Baum-Welch converged at iteration 8 (log-likelihood: 9072.61 → 9124.80).

## Repository Structure

├── HMM_Activity_Recognition.ipynb   — main notebook
├── report.pdf                       — written report
├── README.md
└── data/
└── raw_recordings/              — 60 labeled sensor recordings
├── jumping_1-.../
├── standing_1-.../
├── walking_1-.../
└── still_1-.../


## How to Run

1. Open `HMM_Activity_Recognition.ipynb` in Kaggle or Google Colab
2. Upload the `data/` folder or attach it as a dataset
3. Update `DATA_DIR` to point to your data path
4. Run all cells — takes 2-3 minutes on CPU, no GPU needed

## Implementation Details

- **Viterbi algorithm** — decodes the most likely activity sequence in log-space
- **Baum-Welch algorithm** — trains model parameters with a log-likelihood convergence check
- **Features** — mean, variance, RMS, SMA (time-domain) + dominant frequency, spectral energy via FFT (frequency-domain)
- **Normalization** — Z-score using training statistics only

## Dependencies
numpy, pandas, scipy, matplotlib, seaborn
