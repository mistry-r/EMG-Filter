# EMG Signal Processing and Analysis

This code is a Python script for processing and analyzing Electromyography (EMG) signals, which are electrical signals generated by muscle contractions. It performs various signal processing tasks on EMG data and includes visualizations. Below is an overview of the code's functionality and components.

## Code Overview

### Step 1: Read Data
- Reads EMG data from a CSV file named 'EMG_Datasets.csv'.

### Step 2: Extract Data
- Extracts time and EMG values for both relaxed and contracted muscles.

### Step 3: Calculate Frequency Spectra
- Uses Fast Fourier Transform (FFT) to calculate the frequency spectra of the EMG signals for both relaxed and contracted muscles.
- Filters out negative frequencies.

### Step 4: Filter the Data
- Defines functions for designing and applying notch (bandstop) and bandpass filters to the EMG signals.
- Applies these filters to the relaxed and contracted EMG signals.
- Calculates the filtered frequency spectra using FFT.

### Step 5: Plotting
- Generates four plots:
    1. Non-filtered and filtered frequency spectra for relaxed EMG.
    2. Non-filtered and filtered frequency spectra for contracted EMG.
    3. Time-domain representation of relaxed EMG.
    4. Time-domain representation of contracted EMG.

### Additional Functionality
- Calculates the Root Mean Square (RMS) values before and after applying the filters to both relaxed and contracted EMG signals.
- Saves RMS values to a CSV file named 'rms_values.csv'.
- Prints transfer function poles for the notch (bandstop) and bandpass filters used.

## How to Use

To use this code, follow these steps:

1. Ensure you have the required Python libraries (`pandas`, `numpy`, `matplotlib`, and `scipy`) installed.

2. Place your EMG data in a CSV file named 'EMG_Datasets.csv' with columns: 'Time (s)', 'EMG_Relaxed (mV)', and 'EMG_Contracted (mV)'.

3. Execute the Python script. It will perform the data processing and generate the specified plots.

4. Review the generated plots and CSV file to analyze the EMG data.

## Customization

You can customize this code by adjusting the following parameters:
- `cutoff_freq`, `low_cutoff`, and `high_cutoff` for the notch and bandpass filters.
- Plot appearance and layout.
- Data file input and output.

This code provides a starting point for analyzing EMG data, and you can extend it to suit your specific requirements.

Enjoy working with your EMG signal data and analyzing muscle activity!
