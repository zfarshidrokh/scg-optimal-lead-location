import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, welch
from sklearn.decomposition import PCA

# === Filtering Functions ===
def bandpass_filter(sig, fs, low=5.0, high=20.0, order=4):
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, sig)

def highpass_filter(sig, fs, cutoff=1.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, sig)

# === Data Loading ===
def process_scg_folder(folder_path):
    scg_signals = []
    for file in sorted(os.listdir(folder_path)):
        if not file.endswith('.csv'):
            continue
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path, comment='#')
            if df.shape[1] < 2:
                print(f"Skipping {file_path}: Not enough columns.")
                continue
            time = df['timestamp'].values
            scg_x = df['x'].values
            scg_y = df['y'].values
            scg_z = df['z'].values
            scg_signals.append(pd.DataFrame({
                'time': time,
                'scg_x': scg_x,
                'scg_y': scg_y,
                'scg_z': scg_z,
            }))
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
    return scg_signals

# === FFT Analysis ===
def plot_fft(df, index, axis='scg_z'):
    scg = df[axis].values
    N = len(scg)
    time_ms = (df["time"] - df["time"].iloc[0])
    dt = np.mean(np.diff(time_ms)) * 0.001
    fs = 1 / dt
    scg_detrended = scg - np.mean(scg)
    scg_filtered = highpass_filter(scg_detrended, fs)
    fft_vals = np.abs(fft(scg_filtered))[:N//2]
    freqs = fftfreq(N, d=1/fs)[:N//2]
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_vals)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 100)
    plt.title(f"FFT of {axis.upper()} - Signal #{index+1} (fs ≈ {fs:.1f} Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === PCA Analysis ===
def analyze_pca(df, index):
    scg_xyz = df[['scg_x', 'scg_y', 'scg_z']].values
    time_ms = (df["time"] - df["time"].iloc[0])
    dt = np.mean(np.diff(time_ms)) * 0.001
    fs = 1 / dt
    scg_xyz -= np.mean(scg_xyz, axis=0)
    scg_filtered = np.zeros_like(scg_xyz)
    for i in range(3):
        scg_filtered[:, i] = bandpass_filter(scg_xyz[:, i], fs)
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(scg_filtered)
    explained = pca.explained_variance_ratio_
    print(f"\n--- Signal #{index+1} ---")
    print("Explained Variance:")
    print(f"  PC1: {explained[0]*100:.2f}%")
    print(f"  PC2: {explained[1]*100:.2f}%")
    print(f"  PC3: {explained[2]*100:.2f}%")
    print("\nLoadings (rows: PC1, PC2, PC3; columns: x, y, z):")
    print(np.round(pca.components_, 3))
    plt.figure(figsize=(6, 4))
    plt.bar(['PC1', 'PC2', 'PC3'], explained * 100)
    plt.ylabel("Explained Variance (%)")
    plt.title(f"PCA - Signal #{index+1}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return pcs, explained, pca.components_

# === SNR Estimation ===
def compute_snr(df, axis='scg_y', signal_band=(5, 20), noise_band=(80, 100)):
    time_ms = df["time"] - df["time"].iloc[0]
    dt = np.mean(np.diff(time_ms)) * 0.001
    fs = 1 / dt
    sig = df[axis].values - np.mean(df[axis].values)
    sig_filt = highpass_filter(sig, fs, cutoff=1.0)
    freqs, psd = welch(sig_filt, fs=fs, nperseg=1024)
    signal_power = np.trapz(psd[(freqs >= signal_band[0]) & (freqs <= signal_band[1])], freqs[(freqs >= signal_band[0]) & (freqs <= signal_band[1])])
    noise_power = np.trapz(psd[(freqs >= noise_band[0]) & (freqs <= noise_band[1])], freqs[(freqs >= noise_band[0]) & (freqs <= noise_band[1])])
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

# === Main Execution ===
if __name__ == "__main__":
    data_path = r"C:\\Users\\zfarshid\\Intrenship\\code\\Data_Zahra\\scg_sit_2"
    all_scg_signals = process_scg_folder(data_path)
    print(f"Loaded {len(all_scg_signals)} ECG files.")

    for i, df in enumerate(all_scg_signals):
        plot_fft(df, i, axis='scg_x')
        plot_fft(df, i, axis='scg_y')
        plot_fft(df, i, axis='scg_z')

    pca_results = []
    for i, df in enumerate(all_scg_signals):
        pcs, explained, components = analyze_pca(df, i)
        pca_results.append({
            'index': i,
            'explained': explained,
            'components': components
        })

    snr_summary = []
    for i, df in enumerate(all_scg_signals):
        snr_x = compute_snr(df, axis='scg_x')
        snr_y = compute_snr(df, axis='scg_y')
        snr_z = compute_snr(df, axis='scg_z')
        print(f"Signal #{i+1}: SNR X = {snr_x:.2f} dB, Y = {snr_y:.2f} dB, Z = {snr_z:.2f} dB")
        snr_summary.append({
            'signal_index': i+1,
            'snr_x': snr_x,
            'snr_y': snr_y,
            'snr_z': snr_z
        })
