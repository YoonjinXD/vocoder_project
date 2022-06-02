import numpy as np
from scipy import signal
import librosa
from carriers import sawtooth_osc, square_osc

class Channel_Vocoder():
    def __init__(self, n_channel, sr, freq_scale='linear', filter_type='butter'):
        self.n_channel = n_channel
        self.sr = sr
        self.freq_scale = freq_scale
        self.filter_type = filter_type
        if freq_scale not in ('linear', 'mel'):
            raise ValueError(f"freq_scale should be either \'linear\' or \'mel\'.")
        
        # Set band-pass filters
        nyq = sr * 0.5
        if self.freq_scale == 'linear':
            bp_range = nyq // self.n_channel
            self.bp_filters = [Bandpass_Filter(i*bp_range, (i+1)*bp_range, nyq, filter_type=self.filter_type) for i in range(self.n_channel)]
        elif self.freq_scale == 'mel':
            mel_freqs = librosa.mel_frequencies(n_mels=self.n_channel+1, fmin=0.0, fmax=nyq-1).tolist()
            self.bp_filters = [Bandpass_Filter(f0, f1, nyq, filter_type=self.filter_type) for f0, f1 in zip(mel_freqs[:-1], mel_freqs[1:])]
        
    def __call__(self, modulator_x, carrier_type='sawtooth', carrier_f0=440):
        # Set carrier signal
        dur = modulator_x.shape[0]
        if carrier_type == 'sawtooth':
            carrier_x = sawtooth_osc(f0=carrier_f0, dur=dur, sr=self.sr)
        elif carrier_type == 'square':
            carrier_x = square_osc(f0=carrier_f0, dur=dur, sr=self.sr)
        else:
            # TODO: 여기 종류 더 많아지면 개선 필요
            print("Carrier Type Error")
        
        # Synthesize each channel
        y = np.zeros(dur)
        for channel_idx in range(self.n_channel):
            modulator_block = self.bp_filtering(modulator_x, channel_idx)
            carrier_block = self.bp_filtering(carrier_x, channel_idx)
            y += modulator_block*carrier_block
            
        return y
        
    def bp_filtering(self, x, channel_idx):
        bp_filter = self.bp_filters[channel_idx]
        return bp_filter(x)
    

class Bandpass_Filter():
    def __init__(self, low, high, nyq, filter_type='butter', order=10):
        low = 1 if low == 0 else low
        high = nyq-1 if high == nyq else high
        filter_types = ['butter', 'cheby1', 'bessel', 'biquad']
        if filter_type not in filter_types:
            raise ValueError(f'filter_type should be one of {filter_types}.')
        
        if filter_type == 'butter':
            self.sos = signal.butter(order, Wn=[low/nyq, high/nyq], btype='bandpass', output='sos')
        elif filter_type == 'cheby1':
            self.sos = signal.cheby1(order, rp=0.1, Wn=[low/nyq, high/nyq], btype='bandpass', output='sos')
        elif filter_type == 'bessel':
            self.sos = signal.bessel(order, Wn=[low/nyq, high/nyq], btype='bandpass', output='sos', norm='phase')
        elif filter_type == 'biquad':
            cutoff_freq = (low + high) / 2
            Q = order
            sr = 2*nyq
            # bi-quad lowpass filter -> bandpass로 바꾸기..?
            theta = 2*np.pi*cutoff_freq/sr
            alpha = np.sin(theta)/2/Q
            b = [(1-np.cos(theta)), 2*(1-np.cos(theta)), (1-np.cos(theta))]
            a = [(1+alpha), -2*np.cos(theta), (1-alpha)]
            self.sos = signal.tf2sos(b, a)
        
    def __call__(self, x):
        return signal.sosfilt(self.sos, x)