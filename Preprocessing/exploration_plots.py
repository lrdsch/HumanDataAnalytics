#libraries
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.io import wavfile
import pandas as pd
import IPython.display
import random
import os
from scipy import signal
from scipy.fft import fft,ifft,fftfreq, fftshift
from scipy.signal import stft,spectrogram,periodogram

def one_random_audio(main_dir, end = 1000): #end max is 220500
    dir_path = os.path.join(main_dir, 'data', 'ESC-50')
    audio_files = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
    i = random.randint(0,len(audio_files))
    clip = audio_files[i]
    samplerate = 44100
    y,sr = librosa.load(clip,sr=samplerate)
    # alternatively we can use wavfile.read
    #samplerate * seconds_clip_audio = length_np_array

    # first look at the output
    print(f'The sample rate is {sr}')
    print(f'The data shape is {np.shape(y)}')

    #load the metadata
    file_path = os.path.join(main_dir, 'data', 'meta', 'esc50.csv')
    meta_data = pd.read_csv(file_path)

    #listen
    print(f'Audio category: {list(meta_data.category[meta_data.filename==os.path.basename(os.path.normpath(clip))])[0]}')
    display( IPython.display.Audio(data = y, rate=samplerate)  )
    plt.subplot(1,1,1)
    plt.plot(y[:end])

    return y, sr

def plot_clip_overview(df, sample_rate=44100, segment=25, overlapping=10, column=5, preprocessing='STFT'):

    segment_samples = round(sample_rate * segment / 1000)  # Calculate the number of samples per segment
    overlap_samples = round(sample_rate * overlapping / 1000)

    categories = list(set(df.category))
    row = len(categories)
    
    plt.subplots(row, column, figsize=(12, 1.5 * row))
    plt.tight_layout(pad=0.7)
    
    for j, audio_type in enumerate(categories):
        paths = list(df.full_path[df.category == audio_type])
        paths = random.sample(paths, column)

        for i, audio_sample in enumerate(paths):
            data, samplerate = librosa.load(audio_sample, sr=44100)
            
            if preprocessing == 'STFT':
                # STFT transformation
                D = librosa.stft(data, win_length=segment_samples, hop_length=overlap_samples)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                
                plt.subplot(row,column,j*column+i+1)
                plt.title(audio_type)
                librosa.display.specshow(S_db)
                #plt.colorbar(format='%+2.0f dB')
                
            elif preprocessing == 'MFCC':
                # MFCC transformation
                mfccs = librosa.feature.mfcc(y=data, sr=samplerate, n_mfcc=13)
                max = np.max(np.abs(mfccs))
                mfccs = mfccs/max
                
                plt.subplot(row,column,j*column+i+1)
                plt.title(audio_type)
                librosa.display.specshow(mfccs)
                plt.colorbar()

            elif preprocessing == 'MEL':
                # MEL transformation
                D = librosa.feature.melspectrogram(y = data, win_length=segment_samples, hop_length=overlap_samples, sr = sample_rate)
                M_db = librosa.power_to_db(D, ref=np.max)
                
                plt.subplot(row,column,j*column+i+1)
                plt.title(audio_type)
                librosa.display.specshow(M_db)
                #plt.colorbar(format='%+2.0f dB')
                
    plt.show()

def Spectral_Analysis(audio,
                      sample_rate = 44100,
                      segment = 20,
                      n_fft = None, #padd the frames with zeros before DFT
                      overlapping=10,
                      cepstral_num = 40, #number of mel frequencies cepstral coefficients
                      N_filters = 50, #number of mel filters in frequency domain
                      plot = False,
                      verbose = False,
                      STFT_decibel = False,
                      Mel_spectrogram_decibel = False,
                      MFCC = True):
    if n_fft==None:
        n_fft = segment
    nperseg = round(sample_rate * segment / 1000)  # Calculate the number of samples per segment win_length = nperseg
    print(f'A segment of {segment} ms has {nperseg} samples')
    noverlap = round(sample_rate * overlapping / 1000)
    n_fft = round(sample_rate * n_fft /1000)
    hop_length = nperseg-noverlap

    # SCIPY

    y_hat = fft(audio)
    freq_scipy_periodogram, y_norm = periodogram(audio)
    freq_scipy, time_scipy, stft_scipy = stft(audio,
                                    fs = sample_rate, 
                                    window='hann', 
                                    nperseg=nperseg, 
                                    noverlap=noverlap, 
                                    nfft=n_fft)
    f,t , spec_y = spectrogram(audio, fs = sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    #LIBROSA

    frequencies = librosa.fft_frequencies(sr = sample_rate,n_fft=audio.shape[0]) #non so cosa sia
    stft_librosa = librosa.stft(audio,
                                  hop_length = hop_length, 
                                  win_length = nperseg, 
                                  n_fft = n_fft)
    sample_librosa = librosa.frames_to_samples([i for i in range(stft_librosa.shape[1])],
                                             hop_length=noverlap, 
                                             n_fft=n_fft) # returns time (in samples) of each given frame number
    time_librosa = librosa.frames_to_time([i for i in range(stft_librosa.shape[1])],
                                          hop_length = hop_length,
                                          sr = sample_rate,
                                           n_fft = n_fft )
    freq_librosa = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    S_db = librosa.amplitude_to_db(np.abs(stft_librosa), ref=np.max)

    #librosa other types of spectral data

    mel_y = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft = n_fft, hop_length = hop_length, win_length=nperseg) 
    M_db = librosa.power_to_db(mel_y, ref=np.max)
     
    #Per-channel energy normalization (PCEN)
    S_db_pcen = librosa.pcen(stft_librosa*(2**31), max_size=5) 
    
    #mel frequency cepstral coefficients
    mfcc_y = librosa.feature.mfcc(  y=audio, 
                                    sr=sample_rate, 
                                    n_mfcc=cepstral_num,
                                    n_fft = n_fft,  
                                    hop_length=hop_length, 
                                    htk=True, 
                                    fmin = 40,
                                    n_mels = N_filters)
    
    if verbose:

        print(f'Frame length is {nperseg}')
        print(f'Overlap length is {noverlap}')
        print(f'The length of the windowed signal after padding with zeros (frames) is {n_fft}. ')
        print('\n')
        print(f'Scipy STFT shape {stft_scipy.shape}')
        print(f'Scipy; length of frequencies vector {freq_scipy.shape}')
        print(f'Scipy; length of time vector {time_scipy.shape}')
        print('\n')
        print(f'librosa STFT shape {stft_librosa.shape}')
        print(f'Librosa frames_to_time has shape {time_librosa.shape}, (the time vector for STFT)')
        try:
            print(f'Is it equal to the time vector of Scipy? {(time_librosa-0.01==time_scipy).all()}')
        except:
            pass
        print(f'Librosa fft_frequencies has shape {freq_librosa.shape} (compute the frequencies given the sample_rate and the windowed length)')
        print(f'Is it equal to Scipy frequencies? {(freq_librosa==freq_scipy).all()}')
        print('\n')
        print(f'The STFT converted in decibell domain ha shape {S_db.shape}')
        print(f'Librosa Mel spectrogram of the audio has shape {mel_y.shape} ') #different da feature.mfcc
        print(f'Librosa MFCC features has shape {mfcc_y.shape}')
        print('\n')
        print(f'Librosa per-channel energy normalization (PCEN) has shape f{S_db_pcen.shape}')
        print('\n')

    if plot:

        plt.subplots(11, 1, figsize=(7, 27))
        plt.tight_layout(pad=3)

        plt.subplot(11,1,1)
        plt.plot(np.abs(y_hat))
        plt.title(f'Scipy norm of FFT: Input {audio.shape} > Output {y_hat.shape}')
        plt.xlabel('Frequency [Hz]')

        plt.subplot(11,1,2)
        plt.plot(freq_scipy_periodogram , y_norm)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Linear spectrum [V RMS]')
        plt.title(f'Scipy Periodogram: Input {audio.shape} > Output {y_norm.shape}')

        plt.subplot(11,1,3)
        plt.pcolormesh(time_scipy, freq_scipy, np.abs(stft_scipy),shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'Scipy STFT: Input {audio.shape, nperseg, noverlap} > Output {stft_scipy.shape}')

        plt.subplot(11,1,4)
        plt.pcolormesh(t, f, spec_y)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'Scipy Spectrogram: Input {audio.shape, nperseg, noverlap} > Output {spec_y.shape}')

        plt.subplot(11,1,5)
        plt.pcolormesh(time_librosa, freq_librosa, np.abs(stft_librosa))
        plt.title(f'Librosa STFT: Input {audio.shape, hop_length, nperseg} > Output {stft_librosa.shape}')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        plt.subplot(11,1,6)
        librosa.display.specshow(S_db, x_axis='time', y_axis='linear')
        plt.title(f'Librosa STFT + amplitude conversion into decibel domain. {S_db.shape} ')
        plt.colorbar(format="%+2.f dB")

        plt.subplot(11,1,7)
        librosa.display.specshow(S_db, x_axis='time', y_axis='log')
        plt.title(f'Same as before but using a logarithmic frequency axis')
        plt.colorbar(format="%+2.f dB")

        plt.subplot(11,1,8)
        plt.imshow(mel_y)
        plt.colorbar(format="%+2.f dB")
        plt.title(f'Librosa Mel spectrogram: Input {audio.shape, sample_rate, nperseg, hop_length} > Output {mel_y.shape}')

        plt.subplot(11,1,9)
        librosa.display.specshow(M_db, y_axis='mel', x_axis='time')
        plt.title(f'Mel spectrogram + amplitude to decibel conversion')
        plt.colorbar(format="%+2.f dB")

        plt.subplot(11,1,10)
        librosa.display.specshow(mfcc_y, x_axis='time')
        plt.title(f'Mel Frequency Cepstral Coefficients {cepstral_num}')
        plt.colorbar(format="%+2.f dB")

        plt.subplot(11,1,11)
        librosa.display.specshow(S_db_pcen)
        plt.title(f'Librosa per-channel energy normalization (PCEN)')
        plt.colorbar(format="%+2.f dB")

# decide what return if needed
    if MFCC:
        if Mel_spectrogram_decibel:
            if STFT_decibel:
                return mfcc_y, M_db, S_db
            else:
                return mfcc_y, M_db
        else:
            if STFT_decibel:
                return mfcc_y, S_db
            else:
                return mfcc_y
    else:
        return None
    

    