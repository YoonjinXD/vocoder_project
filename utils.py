import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def get_f0(x):
    f0, _, _ = librosa.pyin(x,
                            fmin=librosa.note_to_hz('C2'), # 1, 
                            fmax=librosa.note_to_hz('C7'), # sr * 0.5,
                            sr=sr,
                            frame_length=n_fft,
                            hop_length=hop_length)
    times = librosa.times_like(f0)
    return f0, times

def compare_stft(m_audio, c_audio, output_audio, show_f0=False, sr=44100, n_fft=1024, hop_length=256):
    # m_audio: (numpy.ndarray)[sr*time, ] modulator audio
    # c_audio: (numpy.ndarray)[sr*time, ] carrier audio
    # output_audio: (numpy.ndarray)[sr*time, ] vocoded output audio
    
    fig, axes = plt.subplots(1,3, figsize=(30,8))
    
    D = np.abs(librosa.stft(m_audio, window='hann', n_fft=n_fft, hop_length=hop_length))
    img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis="time", y_axis="log", sr=sr, ax=axes[0])
    axes[0].set_title('Modulator STFT')
    plt.colorbar(img, ax=axes[0], format="%+2.f dB")
    if show_f0:
        f0, times = get_f0(m_audio)
        axes[0].plot(times, f0, color='cyan', linewidth=3)
    
    D = np.abs(librosa.stft(c_audio, window='hann', n_fft=n_fft, hop_length=hop_length))
    img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis="time", y_axis="log", sr=sr, ax=axes[1])
    axes[1].set_title('Carrier STFT')
    plt.colorbar(img, ax=axes[1], format="%+2.f dB")
    if show_f0:
        f0, times = get_f0(c_audio)
        axes[1].plot(times, f0, color='cyan', linewidth=3)

    D = np.abs(librosa.stft(output_audio, window='hann', n_fft=n_fft, hop_length=hop_length))
    img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis="time", y_axis="log", sr=sr, ax=axes[2])
    axes[2].set_title('Vocoded Output')
    plt.colorbar(img, ax=axes[2], format="%+2.f dB")
    if show_f0:
        f0, times = get_f0(output_audio)
        axes[2].plot(times, f0, color='cyan', linewidth=3)
        