import numpy as np

def sawtooth_osc(f0, dur, sr): 
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

def square_osc(f0, dur, sr): 
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