import numpy as np
import librosa
import scipy.signal as signal
from carrier import Carrier

class Channel_Vocoder():
    def __init__(self, n_channel, sr, n_fft=1024, freq_scale='mel', filter_type='butter'):
        self.n_channel = n_channel
        self.sr = sr
        self.hop_length = n_fft//4
        self.n_fft = n_fft
        self.freq_scale = freq_scale
        self.filter_type = filter_type
        
        nyq = sr * 0.5
        if self.freq_scale == 'linear':
            bp_range = nyq // self.n_channel
            self.bp_filters = [Bandpass_Filter(i*bp_range, (i+1)*bp_range, nyq, filter_type=self.filter_type) for i in range(self.n_channel)]
        elif self.freq_scale == 'mel':
            mel_freqs = librosa.mel_frequencies(n_mels=self.n_channel+1, fmin=0.0, fmax=nyq-1).tolist()
            self.bp_filters = [Bandpass_Filter(f0, f1, nyq, filter_type=self.filter_type) for f0, f1 in zip(mel_freqs[:-1], mel_freqs[1:])]
        
    def __call__(self, 
                 modulator_x, 
                 carrier_type='sawtooth', 
                 carrier_f0=440,
                 carrier_f_inc=880,
                 beta=0.9,
                 formant_step=0,
                 high_noise=False,
                 noise_amp=1, 
                 noise_Q=1):
        # modulator_x: np.array(float), modulator source audio
        # carrier_type: (str), carrier signal shape type e.g. sine, square or sawtooth
        # carrier_f0: (int), carrier fundamental frequency
        # beta: 0.0 ~ 1.0(float), modulator-carrier ratio
        # formant_step: (int), formant shifting step
        # high_noise: (bool), whether to add high frequency random noise in carrier signal
        # noise_amp: 0.0 ~ 1.0(float), amplitude of high frequency noise
        # noise_Q: (float) Q value of bi-quad highpass filter to generate high frequency noise from random noise
        
        # Set carrier signal
        dur = modulator_x.shape[0]
        carrier_x = Carrier(carrier_type, dur, self.sr)(carrier_f0, carrier_f_inc)
        
        # high-freq noise
        if high_noise:
            if not (noise_amp > 0 and noise_amp <= 1):
                raise ValueError('noise_amp should be in range (0,1)')
            rand_noise = (np.random.rand(dur)-0.5)*2*noise_amp
            
            # bi-quad highpass filter
            cut_off_freq = (8e3 + 16e3)//2
            Q = noise_Q

            theta = 2*np.pi*cut_off_freq/sr
            alpha = np.sin(theta)/2/Q
            b = [(1+np.cos(theta)), -2*(1+np.cos(theta)), (1+np.cos(theta))]
            a = [(1+alpha), -2*np.cos(theta), (1-alpha)]
            rand_noise = signal.lfilter(b,a,rand_noise)
            
            carrier_x = carrier_x + rand_noise
            
            # clipping
            if any(carrier_x > 1):
                carrier_x[carrier_x>1] = 1
            if any(carrier_x < -1):
                carrier_x[carrier_x<-1] = -1
        
        # Channel Vocoding
        y = np.zeros((self.n_fft//2 + 1, modulator_x.shape[0]//self.hop_length + 1))
        for channel_idx in range(self.n_channel):
            # Band-pass (Formant shifting)
            carrier_channel_idx = int((channel_idx + formant_step) % self.n_channel)
            modulator_block = self.bp_filters[channel_idx](modulator_x)
            carrier_block = self.bp_filters[carrier_channel_idx](carrier_x)
            
            # RMS & STFT
            modulator_rms = librosa.feature.rms(y=modulator_block, hop_length=self.hop_length)
            carrier_block = librosa.stft(carrier_block, window='hann', n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Multipy and stack it
            y = np.add(y, np.multiply(modulator_rms**beta, carrier_block), casting='unsafe')
        return librosa.istft(y)

    
class Bandpass_Filter():
    def __init__(self, low, high, nyq, filter_type='butter', order=10):
        self.low = int(1 if low == 0 else low)
        self.high = int(nyq-1 if high == nyq else high)
        filter_types = ['butter', 'cheby1', 'bessel', 'biquad']
        if filter_type not in filter_types:
            raise ValueError(f'filter_type should be one of {filter_types}.')
        
        if filter_type == 'butter':
            self.sos = signal.butter(order, Wn=[self.low/nyq, self.high/nyq], btype='bandpass', output='sos')
        elif filter_type == 'cheby1':
            self.sos = signal.cheby1(order, rp=0.1, Wn=[self.low/nyq, self.high/nyq], btype='bandpass', output='sos')
        elif filter_type == 'bessel':
            self.sos = signal.bessel(order, Wn=[self.low/nyq, self.high/nyq], btype='bandpass', output='sos', norm='phase')
        elif filter_type == 'biquad':
            cutoff_freq = (self.low + self.high) / 2
            Q = order
            sr = 2*nyq
            theta = 2*np.pi*cutoff_freq/sr
            alpha = np.sin(theta)/2/Q
            b = [(1-np.cos(theta)), 2*(1-np.cos(theta)), (1-np.cos(theta))]
            a = [(1+alpha), -2*np.cos(theta), (1-alpha)]
            self.sos = signal.tf2sos(b, a)
        
    def __call__(self, x):
        return signal.sosfilt(self.sos, x)
        