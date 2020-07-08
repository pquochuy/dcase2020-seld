from specaugment import SpecAugment
import numpy as np

def augment_spec(X):
    xlen = len(X)
    if len(X.shape) < 4:
        X = np.expand_dims(X, axis=-1)
    nchan = X.shape[-1]
    agument = SpecAugment()
    for i in range(xlen):
        for c in range(nchan):
            X_ = X[i, :, :, c]
            X_ = np.reshape(X_, (-1, X_.shape[0], X_.shape[1], 1))
            freq_masked = agument.freq_mask(X_)  # Applies Frequency Masking to the mel spectrogram
            time_masked = agument.time_mask(freq_masked)  # Applies Time Masking to the mel spectrogram
            X[i, :, :, c] = np.squeeze(time_masked)
    return np.squeeze(X)