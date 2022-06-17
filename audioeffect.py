import numpy as np

def compressor(x, threshold, ratio):
    # x: (numpy.ndarray) audio signal array in data type float32
    # threshold: (int) threshold value of compressor in dB
    # ratio: 1.0 ~ (float) compression ratio value of compressor for signal value above threshold
    #                      1.0 returns original signal

    ref = np.max(np.abs(x))
    x_dB = 10.0 * np.log10(np.abs(x)/ref)
    
    x_compressed = np.zeros(x_dB.shape)
    x_compressed[x_dB > threshold] = (x_dB[x_dB > threshold]-threshold)/ratio + threshold
    x_compressed[x_dB <= threshold] = x_dB[x_dB <= threshold]
    
    x_compressed = 10.0**(x_compressed/10 + np.log10(ref))
    x_compressed = np.sqrt((x_compressed**2)* (np.max(np.abs(x))**2)/(np.max(np.abs(x_compressed))**2))
    x_compressed[x<0] = -x_compressed[x<0]
    
    return x_compressed


def expander(x, threshold, ratio): # noise gate
    # x: (numpy.ndarray) audio signal array in data type float32
    # threshold: (int) threshold value of expander in dB
    # ratio: 1.0 ~ (float) compression ratio value of expander for signal value below threshold
    #                      1.0 returns original signal

    ref = np.max(np.abs(x))
    x_dB = 10.0 * np.log10(np.abs(x)/ref)
    
    x_expanded = np.zeros(x_dB.shape)
    x_expanded[x_dB < threshold] = (x_dB[x_dB < threshold]-threshold)/ratio + threshold
    x_expanded[x_dB >= threshold] = x_dB[x_dB >= threshold]
    
    x_expanded = 10.0**(x_expanded/10 + np.log10(ref))
    x_expanded = np.sqrt((x_expanded**2)* (np.max(np.abs(x))**2)/(np.max(np.abs(x_expanded))**2))
    x_expanded[x<0] = -x_expanded[x<0]   
    
    return x_expanded