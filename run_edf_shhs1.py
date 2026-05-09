import os
import numpy as np
import pandas as pd
import pyedflib
import mne
import yasa
from joblib import cpu_count
from scipy.signal import medfilt
from dynam_o.multitaper import multitaper_spectrogram

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET     = 'SHHS1'  # 'MESA' or 'SHHS1'
SUBJECT     = 'shhs1-200001'
EDF_PATH    = f'/home/ubuntu/sleepdata/shhs/polysomnography/edfs/shhs1/{SUBJECT}.edf'
CHANNEL     = 'EEG'  # EEG channel name — set to None to use first channel
OUTPUT_DIR  = f'/home/ubuntu/pyDYNAM-O/output/{DATASET}'  # output directory for CSV files
STAGES_CSV  = f'{OUTPUT_DIR}/{SUBJECT}_stages.csv'
SPECT_CSV   = f'{OUTPUT_DIR}/{SUBJECT}_spect.csv'
STIMES_CSV  = f'{OUTPUT_DIR}/{SUBJECT}_stimes.csv'
SFREQS_CSV  = f'{OUTPUT_DIR}/{SUBJECT}_sfreqs.csv'
SMOOTH_WINDOW = 3  # epochs in majority-vote smoothing window
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# YASA stage label → pyDYNAM-O stage
# YASA:      'W'=Wake, 'N1', 'N2', 'N3', 'R'=REM
# pyDYNAM-O: 5=Wake,    3=N1, 2=N2, 1=N3, 4=REM
STAGE_REMAP = {'W': 5, 'N1': 3, 'N2': 2, 'N3': 1, 'R': 4}


def load_edf(edf_path, channel=None):
    f = pyedflib.EdfReader(edf_path)
    labels = f.getSignalLabels()
    print(f"Available channels: {labels}")

    if channel is None:
        print("Using first channel by default. Set CHANNEL to override.")
        idx = 0
    else:
        if channel not in labels:
            raise ValueError(f"Channel '{channel}' not found. Available: {labels}")
        idx = labels.index(channel)

    data = f.readSignal(idx).astype(np.float32)
    fs = int(f.getSampleFrequency(idx))
    f.close()

    print(f"Loaded channel '{labels[idx]}': {len(data)/fs:.1f} sec at {fs} Hz")
    return data, fs


def predict_stages_yasa(edf_path, channel, epoch_len=30, smooth_window=SMOOTH_WINDOW):
    """Use YASA to predict sleep stages from a single EEG channel, with majority-vote smoothing."""
    print(f"Predicting stages with YASA on channel '{channel}'...")
    raw = mne.io.read_raw_edf(edf_path, include=[channel], preload=True, verbose=False)
    sls = yasa.SleepStaging(raw, eeg_name=channel)
    hypno = sls.predict()  # Hypnogram object in newer YASA
    yasa_stages = hypno.hypno.values if hasattr(hypno, 'hypno') else np.asarray(hypno)
    yasa_stages = [str(s) for s in yasa_stages]

    times = [i * epoch_len for i in range(len(yasa_stages))]
    remapped = np.array([STAGE_REMAP.get(s, 5) for s in yasa_stages])

    # Smooth using a median filter to remove isolated single-epoch flips
    smoothed = medfilt(remapped, kernel_size=smooth_window)
    n_changed = (smoothed != remapped).sum()
    print(f"Smoothed: {n_changed}/{len(remapped)} epochs reassigned by median filter (kernel={smooth_window})")
    remapped = smoothed.astype(int).tolist()

    stages = pd.DataFrame({'Time': times, 'Stage': remapped})
    print(f"YASA predicted {len(stages)} epochs ({epoch_len}s each)")
    return stages


def main():
    data, fs = load_edf(EDF_PATH, CHANNEL)
    stages = predict_stages_yasa(EDF_PATH, CHANNEL)

    # Save the stages to CSV
    stages.to_csv(STAGES_CSV, index=False)
    print(f"Stages saved to {STAGES_CSV}")

    # Compute multitaper spectrogram (matching pipelines.py:75 settings)
    print("Computing multitaper spectrogram...")
    spect, stimes, sfreqs = multitaper_spectrogram(
        data, fs,
        frequency_range=[0, 5],
        time_bandwidth=2,
        num_tapers=3,
        window_params=[1, 0.05],
        min_nfft=2 ** 10,
        detrend_opt='constant',
        multiprocess=True,
        n_jobs=cpu_count(),
        weighting='unity',
        plot_on=False,
        clim_scale=False,
        verbose=False,
        xyflip=False,
    )
    pd.DataFrame({'time_sec': stimes}).to_csv(STIMES_CSV, index=False)
    print(f"Stimes saved to {STIMES_CSV}: {len(stimes)} time bins")

    pd.DataFrame({'freq_hz': sfreqs}).to_csv(SFREQS_CSV, index=False)
    print(f"Sfreqs saved to {SFREQS_CSV}: {len(sfreqs)} freq bins")

    spect_df = pd.DataFrame(spect, index=sfreqs, columns=stimes)
    spect_df.index.name = 'freq_hz'
    spect_df.to_csv(SPECT_CSV)
    print(f"Spectrogram saved to {SPECT_CSV}: shape={spect.shape}")


if __name__ == '__main__':
    main()