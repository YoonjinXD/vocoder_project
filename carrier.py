import os
import numpy as np
import librosa

class Carrier():
    def __init__(self, name, dur, sr):
        self.name = name
        self.dur = dur
        self.sr = sr
            
    def __call__(self, *freqs):
        if self.name == 'sawtooth':
            y = self.make_sawtooth_osc(freqs[0], self.dur, self.sr)
        elif self.name == 'square':
            y = self.make_square_osc(freqs[0], self.dur, self.sr)
        elif self.name == 'exp_sawtooth':
            y = self.make_exp_sawtooth_osc(freqs[0], freqs[1], self.dur, self.sr)
        else:
            f_path = f"./audio_resources/carrier/{self.name}.wav"
            if not os.path.exists(f_path):
                raise "Carrier Source File Not Exist"
            y, _ = librosa.load(f_path, sr=self.sr, duration=(self.dur//self.sr)+1) 
            y = y[:int(self.dur)]
        return y
        
    def make_sawtooth_osc(self, f0, dur, sr): 
        # f0:  fundamental frequency
        # dur: duration
        # sr:  sampling rate

        phase_inc = 2/(sr/f0)
        phase = 0
        x = np.zeros(dur)

        for n in range(len(x)):
            phase = phase + phase_inc
            if (phase > 1):
                phase = phase - 2

            x[n] = phase

        return x   

    def make_square_osc(self, f0, dur, sr): 
        # f0:  fundamental frequency
        # dur: duration
        # sr:  sampling rate

        phase_inc = 2/(sr/f0)
        phase = 0
        x = np.zeros(dur)

        for n in range(len(x)):
            phase = phase + phase_inc
            if (phase > 1):
                phase = phase - 2

            if phase > 0: 
                x[n] = 0.9
            else: 
                x[n] = -0.9

        return x  
    
    def make_exp_sawtooth_osc(self, f_start, f_stop, dur, sr):
        freq = self.powspace(f_start, f_stop, dur)

        phase = 0
        x = np.zeros(dur)

        for n in range(len(x)):
            phase_inc = 2/(sr/freq[n])
            phase = phase + phase_inc
            if (phase > 1):
                phase = phase - 2

            x[n] = phase

        return x  
    
    def powspace(self, start: float, stop: float, num: int):
        log_start, log_stop = np.log(start), np.log(stop)
        return np.exp(np.linspace(log_start, log_stop, num))