import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter

# Step 1: Read data from the CSV file
data = pd.read_csv('EMG_Datasets.csv')

# Step 2: Extract time and EMG values for relaxed and contracted muscles
time = data['Time (s)']
emg_relaxed = data['EMG_Relaxed (mV)']
emg_contracted = data['EMG_Contracted (mV)']

# Step 3: Calculate frequency spectra using FFT
fs = 1 / (time[1] - time[0])  # Sampling frequency
freq_relaxed = np.fft.fftfreq(len(time), 1/fs)
spectrum_relaxed = np.abs(np.fft.fft(emg_relaxed))

# Slice the arrays to keep only non-negative frequencies
non_neg_freq_relaxed = freq_relaxed[freq_relaxed >= 0]
non_neg_spectrum_relaxed = spectrum_relaxed[freq_relaxed >= 0]

freq_contracted = np.fft.fftfreq(len(time), 1/fs)
spectrum_contracted = np.abs(np.fft.fft(emg_contracted))

# Slice the arrays to keep only non-negative frequencies
non_neg_freq_contracted = freq_contracted[freq_contracted >= 0]
non_neg_spectrum_contracted = spectrum_contracted[freq_contracted >= 0]

# Step 4: Design and apply the notch and bandpass filters
def butterworth_notch_filter(data, cutoff_freq, fs):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    Q = 30.0  # Quality factor for notch filter
    b, a = butter(2, [normal_cutoff - 0.5/Q, normal_cutoff + 0.5/Q], btype='bandstop')
    filtered_data = lfilter(b, a, data)
    return filtered_data

def butterworth_bandpass_filter(data, low_cutoff, high_cutoff, fs):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(2, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data

cutoff_freq = 60.0  # 60 Hz power line frequency
low_cutoff = 0.1  # 0.1 Hz
high_cutoff = 450.0  # 450 Hz

# Apply the notch and bandpass filters to relaxed and contracted EMG signals
filtered_relaxed = butterworth_bandpass_filter(butterworth_notch_filter(emg_relaxed, cutoff_freq, fs), low_cutoff, high_cutoff, fs)
filtered_contracted = butterworth_bandpass_filter(butterworth_notch_filter(emg_contracted, cutoff_freq, fs), low_cutoff, high_cutoff, fs)

# Calculate the filtered frequency spectra using FFT
filtered_spectrum_relaxed = np.abs(np.fft.fft(filtered_relaxed))
filtered_freq_relaxed = np.fft.fftfreq(len(filtered_relaxed), 1/fs)

filtered_spectrum_contracted = np.abs(np.fft.fft(filtered_contracted))
filtered_freq_contracted = np.fft.fftfreq(len(filtered_contracted), 1/fs)

# Function to calculate RMS
def calculate_rms(signal):
    # Calculate the squared values of the signal
    squared_signal = signal**2

    # Calculate the mean of the squared values
    mean_squared_signal = np.mean(squared_signal)

    # Calculate the RMS value (square root of the mean)
    rms = np.sqrt(mean_squared_signal)

    return rms

# Calculate RMS before and after applying the filters
rms_before_filter_relaxed = calculate_rms(emg_relaxed)
rms_after_filter_relaxed = calculate_rms(filtered_relaxed)

rms_before_filter_contracted = calculate_rms(emg_contracted)
rms_after_filter_contracted = calculate_rms(filtered_contracted)

# Save RMS values to a CSV file
rms_data = pd.DataFrame({
    'Metric': ['RMS before filter - Relaxed EMG', 'RMS after filter - Relaxed EMG',
               'RMS before filter - Contracted EMG', 'RMS after filter - Contracted EMG'],
    'Value': [rms_before_filter_relaxed, rms_after_filter_relaxed,
              rms_before_filter_contracted, rms_after_filter_contracted]
})

rms_data.to_csv('rms_values.csv', index=False)

# Design notch (bandstop) and bandpass filters functions, and print the transfer function poles for each filter
b_notch, a_notch = signal.butter(2, [59 / (fs / 2), 61 / (fs / 2)], btype='bandstop', output='ba')
b_bandpass, a_bandpass = signal.butter(2, [0.1/(fs/2), 450/(fs/2)], btype='band', output='ba')

print("Notch Filter Transfer Function Poles:", "H(s) =", " * ".join(f"(s - {pole:.4f})" for pole in a_notch[1:]))

print("Bandpass Filter Transfer Function Poles:", "H(s) =", " * ".join(f"(s - {pole:.4f})" for pole in a_bandpass[1:]))

# Step 5: Plot the non-filtered and filtered frequency spectra for relaxed and contracted EMG signals
plt.figure(figsize=(14, 10))

# Non-filtered and Filtered FFT of the relaxed EMG
plt.subplot(2, 2, 1)
plt.plot(non_neg_freq_relaxed, non_neg_spectrum_relaxed, label='Non-Filtered')
plt.plot(filtered_freq_relaxed[filtered_freq_relaxed >= 0], filtered_spectrum_relaxed[filtered_freq_relaxed >= 0], label='Filtered')
plt.title('Frequency Spectrum - Relaxed EMG')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

# Non-filtered and Filtered FFT of the contracted EMG
plt.subplot(2, 2, 2)
plt.plot(non_neg_freq_contracted, non_neg_spectrum_contracted, label='Non-Filtered')
plt.plot(filtered_freq_contracted[filtered_freq_contracted >= 0], filtered_spectrum_contracted[filtered_freq_contracted >= 0], label='Filtered')
plt.title('Frequency Spectrum - Contracted EMG')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

# Non-filtered and Filtered relaxed EMG in the time domain
plt.subplot(2, 2, 3)
plt.plot(time, emg_relaxed, label='Non-Filtered')
plt.plot(time, filtered_relaxed, label='Filtered')
plt.title('Time-Domain EMG - Relaxed Muscle')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()

# Non-filtered and Filtered contracted EMG in the time domain
plt.subplot(2, 2, 4)
plt.plot(time, emg_contracted, label='Non-Filtered')
plt.plot(time, filtered_contracted, label='Filtered')
plt.title('Time-Domain EMG - Contracted Muscle')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()

plt.tight_layout()
plt.show()
