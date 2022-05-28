import numpy as np
from scipy import signal

from carriers import sawtooth_osc, square_osc

class Channel_Vocoder():
    def __init__(self, n_channel, sr):
        self.n_channel = n_channel
        self.sr = sr
        
        # Set band-pass filters
        nyq = sr * 0.5
        bp_range = nyq // self.n_channel
        self.bp_filters = [Bandpass_Filter(i*bp_range, (i+1)*bp_range, nyq) for i in range(self.n_channel)]
        
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
    def __init__(self, low, high, nyq, order=5):
        low = 1 if low == 0 else low
        high = nyq-1 if high == nyq else high
        self.sos = signal.butter(order, [low/nyq, high/nyq], btype='bandpass', output='sos')
        
    def __call__(self, x):
        return signal.sosfilt(self.sos, x)