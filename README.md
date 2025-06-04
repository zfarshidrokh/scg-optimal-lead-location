# SCG Optimal Lead Location

This repository contains code to analyze Seismocardiogram (SCG) signals recorded from six different chest locations (V1–V6) to identify the optimal lead position based on signal clarity and quality metrics such as SNR, PCA variance, and frequency analysis.

## 📌 Features

- Load and process raw SCG data from CSV files
- Time-domain visualization of SCG signals
- High-pass and band-pass filtering
- Frequency-domain analysis (FFT and Welch PSD)
- PCA for dimensionality and orientation analysis
- SNR (Signal-to-Noise Ratio) estimation

## 🗂️ Folder Structure

```bash
.
├── data/                  # (Optional) Example input data files
├── src/
│   ├── preprocessing.py   # CSV loading, filtering
│   ├── visualization.py   # Time plots, FFT
│   ├── pca_analysis.py    # PCA on 3-axis SCG signals
│   └── snr_analysis.py    # Compute SNR using Welch method
├── main.py                # Run full analysis pipeline
├── requirements.txt       # Required Python libraries
└── README.md
