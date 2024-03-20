import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import stft,spectrogram
from scipy.fft import fft,ifft


def segmentation(data, sample_rate=44100, segment=25, overlapping=10):
    #segment and overlapping are expressed in milliseconds
    if type(data)=='str':
        data, sample_rate = librosa.load(data,sr=sample_rate)
        
    N_tot = len(data)
    N = round(sample_rate * segment / 1000)  # Calculate the number of samples per segment
    overlap = round(sample_rate * overlapping / 1000)  # Calculate the number of overlapping samples
    
    total_segments = (N_tot - N) / (N - overlap) +1
    if not int(total_segments)==total_segments:
        zeros_to_add = ((N_tot-N)//(N-overlap)+1)*(N-overlap)-N_tot+N
        data = np.append(data,np.zeros((zeros_to_add,1)))
        total_segments = (N_tot-N)//(N-overlap)+2
    else:
        total_segments = int(total_segments)

    frames = np.zeros((total_segments, N))  # Initialize an array to store the segmented data
    start = 0  # Starting index for each segment

    for i in range(total_segments):
        frames[i] = data[start:start + N]  # Assign the segment of data to the array
        start += N - overlap  # Move the starting index forward with the overlap
    
    return frames, N, total_segments


M = lambda x: 2595 * np.log(1+x/700) # Mel Scale
M_inverse = lambda y:(np.exp(y/2595)-1)*700 # Inverse of Mel Scale
alpha = lambda u,N: N**(-0.5)*(u==0)+(2/N)**0.5*(u!=0)

def hamming_window(n,N):
    return 0.54-0.46*np.cos(2*np.pi*n/(N-1))

def DFT(frame): #discrete fourier transform
    N = len(frame)
    #S = np.zeros((1,N),dtype='complex')
    windowed_frame = [frame[n]*hamming_window(n,N) for n in range(N)]
    S = fft(windowed_frame)
    #for k in range(N):
    #    S[0,k]=np.sum([frame[n]*hamming_window(n,N)*np.exp(-2*np.pi*k*n*1j/N) for n in range(N)])
    return S

def DCT(vec):
    N = len(vec)
    C = np.zeros((1,N))
    for u in range(N):
        C[0,u] = alpha(u,N)*np.sum([vec[n]*np.cos(np.pi*(2*n+1)*u/(2*N)) for n in range(N)])

    return C

def Delta(vec,M):
    if not M%2==0:
        print('M is not even!')
        return
    else:
        N = len(vec)
        vec = np.pad(vec, (M, M), 'constant', constant_values=(0, 0))
        delta = np.zeros((N,1))
        den = M*(M+1)*(2*M+1)/3
        for i in range(N):
            delta[i]=np.sum([ m*(vec[M+i+m]-vec[M+i-m]) for m in range(1,M+1)])/den

        return delta

def triangular(f1,f2,f3,value_in):

    if value_in<f1 or value_in>f3:
        return 0
    elif value_in<=f2:
        return (value_in-f1)/(f2-f1)
    elif value_in>f2:
        return (f3-value_in)/(f3-f2)

def Mel_filterbank(f_min,f_max,N_filters,N,sr):
    equispaced = np.linspace(M(f_min),M(f_max),num=N_filters+2)
    frequencies = np.asarray(list(map(M_inverse,equispaced)))
    N_half = (N+1)//2
    filters = np.zeros((N_filters,N_half))
    for m in range(N_filters):
        f1,f2,f3 = frequencies[m:m+3]
        filters[m] = np.asarray([triangular(f1,f2,f3,sr/N*i) for i in range(1,N_half+1)])

    return filters

def MFCC(audio, cepstral_num = 20, 
                sr=44100, 
                segment=25, 
                overlapping=10, 
                f_min = 20, 
                N_filters=50,
                energy_feature = False,
                delta_feature = True,
                M = 2,
                delta_delta_feature = True ): #Mel-frequency cepstral coefficients 
    if type(audio)==type('abc'):
        data, sample_rate = librosa.load(audio,sr=sr)
    N_total = len(data)
    # step 1: frame the audio
    frames, N, n_frames = segmentation(data, sample_rate=sr, segment=segment, overlapping=overlapping)

    #define the filterbank once for all
    f_max = sr/2 #Nyquist Critical Frequency
    if f_min<sr/N:
        f_min=sr/N
    filters = Mel_filterbank(f_min,f_max,N_filters,N,sr)

    feature_vectors = []
    for i,frame in enumerate(frames):
        #step 2: power spectrum
        power_spectrum = DFT(frame)
        periodogram = np.absolute(power_spectrum)**2/2
        periodogram = periodogram[:(N+1)//2]

        #step 3: filterbank
        energy = np.dot(filters,periodogram.reshape((N+1)//2,1))

        #step 4: logarithms
        log_energy = np.log(energy)

        #step 5: discrete cosine transform
        cepstral = DCT(log_energy)

        #step 6: cepstral coefficients
        if cepstral.shape[1]<cepstral_num-1:
            cepstral_num = cepstral.shape[1]-1
        full_feature_vector = cepstral[0,1:cepstral_num+1]

        #additional features:

        if delta_feature:
            delta = Delta(full_feature_vector,M)
            full_feature_vector = np.concatenate((full_feature_vector,delta.reshape(-1)),axis=0)
        if delta_delta_feature:
            delta_delta = Delta(delta,M)
            full_feature_vector = np.concatenate((full_feature_vector, delta_delta.reshape(-1)),axis=0)
        if energy_feature:
            E_tot = np.log10(np.sum(frame**2)).reshape((1,))
            full_feature_vector = np.concatenate((E_tot,full_feature_vector),axis = 0)

        feature_vectors.append(full_feature_vector)

    return np.asarray(feature_vectors)


# CODE TO CONVERT THE OGG FILES. NO MORE REQUIRED

#FFMPEG WAY

def is_folder_empty(folder_path):
    # INPUT: str of the folder path
    # OUTPUT: TRUE / FALSE if the folder is empty 
    return len(os.listdir(folder_path)) == 0

if '01_conv' not in os.listdir(os.path.join(main_dir,'data','ESC-US')):
    os.mkdir(os.path.join(main_dir,'data','ESC-US','01_conv'))

path_input = os.path.join(main_dir,'data','ESC-US','01')
path_output = os.path.join(main_dir,'data','ESC-US','01_conv')

# Get a list of all files and directories in the specified directory
files_in = os.listdir(path_input)
files_out = os.listdir(path_output)
files_check = [file[:-3] + "wav" for file in files_in if file[:-3] + "wav" not in files_out]

def convert_ogg_to_wav(input_file, output_file):
    # INPUT: input_file = str path of the input file .ogg we want to convert, output_file = path of the output file .wav we want ot create
    # For this function to work you need the ffmpeg program installed on your computer
    command = ['ffmpeg', '-i', input_file, output_file]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for file in files_check:
    input_file = os.path.join(main_dir,'data','ESC-US','01',file[:-3]+'ogg')
    output_file = os.path.join(main_dir,'data','ESC-US','01_conv',file)
    #!ffmpeg -i {input_file} {output_file}
    convert_ogg_to_wav(input_file, output_file)

number_files = len(os.listdir(path_output))

#PYDUB WAY (FORCED IN COLAB)
if '02_conv' not in os.listdir(os.path.join(main_dir,'data','ESC-US')):
    os.mkdir(os.path.join(main_dir,'data','ESC-US','02_conv'))

path_input = os.path.join(main_dir,'data','ESC-US','02')
path_output = os.path.join(main_dir,'data','ESC-US','02_conv')

# Get a list of all files and directories in the specified directory
files_in = os.listdir(path_input)
files_out = os.listdir(path_output)
files_check = [file[:-3] + "wav" for file in files_in if file[:-3] + "wav" not in files_out]

for file in files_check:
    input_file = os.path.join(main_dir,'data','ESC-US','02',file[:-3]+'ogg')
    output_file = os.path.join(main_dir,'data','ESC-US','02_conv',file)
    x = AudioSegment.from_file(input_file)
    x.export(output_file, format='wav') 


number_files = len(os.listdir(path_output))

#BATCH TRAINING
def batch_training(main_dir, dataset_size = 1000, delete = True, shuffle = True ): 
    # the parameters to insert the model are missing
    data_dir = os.path.join(main_dir,'data','ESC-US')

    list_dir = os.listdir(data_dir)
    list_path = []

    for folder in list_dir:
        folder_path = os.path.join(data_dir,folder)
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file[-3:]=='ogg'  ]
        list_path.extend(files)


    if 'temp_conv' not in os.listdir(data_dir):
        os.mkdir(os.path.join(data_dir,'temp_conv'))

    num_batch = len(list_path)//dataset_size
    print(len(list_path))
    while len(list_path)>dataset_size:
        if shuffle:
            data = random.sample(list_path, dataset_size)
        else:
            data = list_path[:dataset_size]
        list_path = [p for p in list_path if p not in data]
        print(len(list_path))

        for input_file in data:
            out_file_name = input_file.split('\\')[-1:][0].replace('ogg','wav')
            output_file = os.path.join(data_dir,'temp_conv',out_file_name)
            x = AudioSegment.from_file(input_file)
            x.export(output_file, format='wav') 


        #TRAIN THE MODEL ON THE temp_conv DIR

        if delete:
            temp_files = os.listdir(os.path.join(data_dir,'temp_conv' ))
            temp_files = [os.path.join(data_dir, 'temp_conv', file) for file in temp_files]
            for temp in temp_files:
                os.remove(temp)

    return 



