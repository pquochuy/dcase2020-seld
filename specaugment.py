import random
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()

#https://github.com/pyyush/SpecAugment/blob/master/main.py
class SpecAugment():
    '''
    Augmentation Parameters for policies
    -----------------------------------------
    Policy | W  | F  | m_F |  T  |  p  | m_T
    -----------------------------------------
    None   |  0 |  0 |  -  |  0  |  -  |  -
    -----------------------------------------
    LB     | 80 | 27 |  1  | 100 | 1.0 | 1
    -----------------------------------------
    LD     | 80 | 27 |  2  | 100 | 1.0 | 2
    -----------------------------------------
    SM     | 40 | 15 |  2  |  70 | 0.2 | 2
    -----------------------------------------
    SS     | 40 | 27 |  2  |  70 | 0.2 | 2
    -----------------------------------------

    LB  : LibriSpeech basic
    LD  : LibriSpeech double
    SM  : Switchboard mild
    SS  : Switchboard strong
    W   : Time Warp parameter
    F   : Frequency Mask parameter
    m_F : Number of Frequency masks
    T   : Time Mask parameter
    p   : Parameter for calculating upper bound for time mask
    m_T : Number of time masks
    '''

    def __init__(self):
        self.W, self.F, self.m_F, self.T, self.p, self.m_T = 20, 10, 2, 10, 1.0, 2

        # Reshape to [Batch_size, time, freq, 1] for sparse_image_warp func.
        #self.mel_spectrogram = np.reshape(self.mel_spectrogram, (-1, self.mel_spectrogram.shape[0], self.mel_spectrogram.shape[1], 1))


    def time_warp(self, mel_spectrogram):

        v, tau = mel_spectrogram.shape[1], mel_spectrogram.shape[2]

        horiz_line_thru_ctr = mel_spectrogram[0][v//2]

        random_pt = horiz_line_thru_ctr[random.randrange(self.W, tau - self.W)] # random point along the horizontal/time axis
        w = np.random.uniform((-self.W), self.W) # distance

        # Source Points
        src_points = [[[v//2, random_pt[0]]]]

        # Destination Points
        dest_points = [[[v//2, random_pt[0] + w]]]

        mel_spectrogram, _ = tf.contrib.image.sparse_image_warp(mel_spectrogram.astype(np.float32), src_points, dest_points, num_boundary_points=2)
        mel_spectrogram = mel_spectrogram.eval()

        return mel_spectrogram


    def freq_mask(self, mel_spectrogram):

        v = mel_spectrogram.shape[1] # no. of mel freq channels

        # apply m_F frequency masks to the mel spectrogram
        for i in range(self.m_F):
            f = int(np.random.uniform(0, self.F)) # [0, F)
            f0 = random.randint(0, v - f) # [0, v - f)

            #mean = tf.reduce_mean(self.mel_spectrogram).numpy()
            #mean = tf.reduce_mean(self.mel_spectrogram).eval()
            mean = np.mean(mel_spectrogram)
            mel_spectrogram = np.array(mel_spectrogram)
            mel_spectrogram[:, f0:f0 + f, :, :] = mean

        return mel_spectrogram


    def time_mask(self, mel_spectrogram):

        tau = mel_spectrogram.shape[2] # time steps

        # apply m_T time masks to the mel spectrogram
        for i in range(self.m_T):
            t = int(np.random.uniform(0, self.T)) # [0, T)
            t0 = random.randint(0, tau - t) # [0, tau - t)

            #mean = tf.reduce_mean(self.mel_spectrogram).numpy()
            #mean = tf.reduce_mean(self.mel_spectrogram).eval()
            mean = np.mean(mel_spectrogram)
            mel_spectrogram[:, :, t0:t0 + t, :] = mean

        return mel_spectrogram
