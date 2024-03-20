import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import librosa
import IPython.display as ipd
import seaborn as sb
import random


def reduce(y):
    ss = list(set(y))
    ss.sort()
    dict = { target:i for i,target in enumerate(ss)}
    yy = [dict[i] for i in y]
    return yy


def confusion_matrix(y_true, y_pred, labels, show_figure = True):
    #convert to list y_true and y_pred if they are tensors
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    #print the types 

    n = len(labels)
    annot = False
    h,w = 10,8
    if n<50:
        annot=True
        y_true = reduce(y_true)
        y_pred = reduce(y_pred)
        h,w = 8,6
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred, num_classes = n)
    if show_figure:
        plt.figure(figsize=(h, w))
        sb.heatmap(confusion_mtx,
                    xticklabels=labels,
                    yticklabels=labels,
                    annot=annot, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('True Label')
        plt.show()

    return np.asarray(confusion_mtx)


def get_misclassified_audio_paths(true_labels, predicted_labels, data_frame, labels):
    misclassified_indices = np.where((predicted_labels != true_labels))[0]
    misclassified_targets = set(true_labels[misclassified_indices])
    misclassified_audio_paths = []

    for target in misclassified_targets:
        target_name = labels[target]
        path = data_frame[data_frame["category"] == target_name].iloc[0]["full_path"]
        misclassified_audio_paths.append(path)

    return misclassified_audio_paths


def listen_to_wrong_audio(data_frame, true_labels, predicted_labels, labels, confusion_mtx = None):
    if confusion_mtx is None:
        confusion_mtx = confusion_matrix(true_labels, predicted_labels, labels, show_figure = False)

    samplerate = 44100
    row = np.sum(confusion_mtx, axis=0) #predictions
    col = np.sum(confusion_mtx, axis=1) #true labels
    ind = np.argmax(np.abs(row - col))  # index of maximum error in confusion matrix
    if row[ind]==col[ind]:
        print('No misclassified audio')
        return
    cat_true = labels[ind]  # category of the corresponding index
    print(f'The most misclassified class is {cat_true}')

    #retrive the corresponding path for a random audio of that class
    path_true = data_frame[data_frame["category"] == cat_true].iloc[0]["full_path"] 
    y,sr = librosa.load(path_true,sr=samplerate)
    print(f'Audio category: {cat_true}')
    display( ipd.Audio(data = y, rate=samplerate)  )

    # find the target classified as cat_true in y_pred but not in y_true
    if row[ind] > col[ind]: #more predicted than true
        print(f'The are more audio predicted as {cat_true} then actually is:')
        index_of_missclassified = np.argsort(confusion_mtx[:,ind], axis=0)[-1]
        if index_of_missclassified == ind:
            index_of_missclassified = np.argsort(confusion_mtx[:,ind], axis=0)[-2]
        cat_false = labels[index_of_missclassified]
        print(f'Audio category misclassified as {cat_true}: {cat_false}')
        path_missclassified = data_frame[data_frame["category"] == cat_false].iloc[0]["full_path"] 
        y,sr = librosa.load(path_missclassified,sr=samplerate)
        display( ipd.Audio(data = y, rate=samplerate)  )

    elif row[ind] < col[ind]: #more true than predicted
        print(f'The are more true {cat_true} audio than the predicted ones:')
        index_of_missclassified = np.argsort(np.max(confusion_mtx[ind,:], axis=0))[-1]
        if index_of_missclassified == ind:
            index_of_missclassified = np.argsort(np.max(confusion_mtx[ind,:], axis=0))[-2]
        cat_false = labels[index_of_missclassified]
        print(f'Some {cat_true} are classified as {cat_false}, which is, for example:')
        path_missclassified = data_frame[data_frame["category"] == cat_false].iloc[0]["full_path"]
        y,sr = librosa.load(path_missclassified,sr=samplerate)
        display( ipd.Audio(data = y, rate=samplerate)  )


def plot_history(history):
    keys = list(history.history.keys())

    if keys[1] != 'mse':
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.grid(True)
        plt.plot(100*np.array(history.history[keys[0]]), label=keys[0])
        plt.plot(100*np.array(history.history[keys[2]]), label=keys[2])
        plt.xlabel('Epoch')
        plt.ylabel(keys[0])
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.grid(True)
        plt.plot(history.history[keys[1]], label=keys[1])
        plt.plot(history.history[keys[3]], label=keys[3])
        plt.xlabel('Epoch')
        plt.ylabel(keys[1])
        plt.legend()
        plt.show()
    
    else: 
        plt.figure(figsize=(7, 5))
        plt.grid(True)
        plt.plot(history.history[keys[0]], label=keys[0])
        plt.plot(history.history[keys[2]], label=keys[2])
        plt.xlabel('Epoch')
        plt.ylabel(keys[1])
        plt.legend()
        plt.show()


def visualize_the_weights(model, layer_name = None, layer_number = None, n_filters = 16, verbose = 1):

    #extract the weights of the layer
    if layer_name is None:
        number_of_layers = len(model.get_weights())      
        if layer_number is None:
            layer_number = 0
        if layer_number>number_of_layers-1:
            print(f'The model has only {number_of_layers} layers. Please select a layer number between 0 and {number_of_layers-1}')
            return

        w0=model.get_layer(index = layer_number).get_weights()[0] #weights of the first layer

    else:
        layer = model.get_layer(layer_name)
        w0 = layer.get_weights()[0][:,:,0,:] #weights of the first layer

    if verbose == 1:
        print(f'The shape of the weights is {w0.shape}')
        print(f'The number of filters is {w0.shape[3]}')
    
    (height, width) = w0.shape[:2]
    figsize = (width*3, height*3)
    #plot the filters
    plt.figure(figsize=figsize)
    for i in range(n_filters):
        plt.subplot(4,4,i+1)
        plt.imshow(w0[:,:,:,i], interpolation='none')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    if verbose > 0:
        return w0
    else:
        return None
    

def plot_latent_space(encoder, dataset, show_labels = "all", numeric_labels = False, verbose = False, expected_shape = (1,220500, 1)):
    # encoder must be pre-trained 
    # dataset must be a dataset train, val or test returned by create_dataset
    # code_size must be 2 or 3, otherwise the plot will not be able to be displayed
    # show_labels can be "all", a dict of pairs (numeric label/string label, 0 if we do not want to plot or 1 if we want to plot that label)
    # show_labels can be incomplete (for example {0:1, 1:0, 2:1}), in this case the labels not present or with 0 value in the dict will not be plotted 
    # Classes in ESC10 are [0:'rain', 1:'sea_waves', 2:'clock_tick', 3:'chainsaw', 4:'crying_baby', 5:'rooster', 6:'crackling_fire', 7:'dog', 8:'helicopter', 9:'sneezing']
    # google_plot is a boolean, if True the plot will be displayed with plotly, if False the plot will be displayed with matplotlib
    # expected_shape of raw audio is (1,220500, 1), for STFT is (1,64,128,1), for MFCC hopefully it is (1,64,128,1)

    #defining the mapping dictionaries
    lab = ['rain', 'sea_waves', 'clock_tick', 'chainsaw', 'crying_baby', 'rooster', 'crackling_fire', 'dog', 'helicopter', 'sneezing']
    mapping_dict = dict(zip(lab, range(10)))
    inverse_mapping_dict = dict(zip(range(10), [string.replace('_', ' ') for string in lab]))
    
    if show_labels=="all":
        show_labels = dict.fromkeys(range(10), 1)
    if type(show_labels)==dict:
        #check if the keys of the dict are 'rain', 'sea_waves', 'clock_tick', 'chainsaw', 'crying_baby', 'rooster', 'crackling_fire', 'dog', 'helicopter', 'sneezing', in this case we need to convert the keys to numbers
        if set(show_labels.keys()).issubset(set(lab)): #we use subset because we can have a dict with only some of the labels
            show_labels = {mapping_dict[k]: v for k, v in show_labels.items()}
        show_labels = {**show_labels, **{k: 0 for k in range(10) if k not in show_labels.keys()}} # out of the if because we want to add the 0 values also if the dict with numeric keys is incomplete

    dataset_encoded = dataset.map(lambda y,lab: (encoder(y),lab))

    #define the code_size
    code_size = len(encoder(tf.zeros(expected_shape))[0])
    
    if code_size>3 or code_size<2:
        print("code_size must be 2 or 3")
        return

    #create a list of encoded vectors and a list of labels
    encoded_list = []
    label_list = []

    #unbatch the dataset
    unbatched_encoded_dataset = dataset_encoded.unbatch()

    #number of elements in the unbatched_dataset
    num_elements = sum(1 for _ in unbatched_encoded_dataset)

    for i,j in unbatched_encoded_dataset.take(num_elements):
        encoded_list.append(np.array(i))
        label_list.append(np.array(j))

    #invert the one hot encoding of the label list
    label_list = np.argmax(label_list, axis=1)

    #create a dataframe with the encoded vectors and the labels
    df = pd.DataFrame(encoded_list, columns=['x', 'y', 'z'][:code_size])
    df['label'] = label_list

    #remove the labels that has a 0 in show_labels
    for i in show_labels:
        if show_labels[i]==0:
            df = df[df['label']!=i]

    #transform label to string with mapping dict where the keys are the string labels and the values are the numeric labels        
    if not numeric_labels:
        df['label'] = df['label'].map(inverse_mapping_dict)

    #transform label to string required to have discrete legend in the plots
    if numeric_labels:
        df['label'] = df['label'].astype(str)

    #plot the latent space
    if code_size==2:
        fig = px.scatter(df, x='x', y='y', color='label', opacity=1)
        fig.show()

    if code_size==3:
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', opacity=1)
        fig.show()

    if verbose:
        #count how many examples we have for each label
        print(df['label'].value_counts())

    return

'''
example of use:
plot_latent_space(encoder, train, show_labels = {0:1, 1:0, 2:1}, numeric_labels = False)
plot_latent_space(encoder, train, show_labels = {'rain':1, 'sea_waves':0, 'chainsaw':1}, numeric_labels = True)
'''


def plot_original_reconstructed(model, n_figures, test):

    # function to plot the images (STFT, MEL or MFCC) and the reconstructed images
    # model : the autoencoder model
    # n_figures: how many images to plot
    # test: the test dataset
    n=n_figures
    plt.figure(figsize=(10, 15))
    for index, (image, copy) in zip(range(1, n + 1), test.unbatch().shuffle(buffer_size = n+1).take(n)):

        ax = plt.subplot(n, 2, 2*index-1)
        plt.imshow(image.numpy(),  cmap = 'viridis')
        plt.title('Original')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        reconstructed = model.predict(np.expand_dims(image, axis=0))
        reconstructed = reconstructed.reshape(64,128,1)
        ax = plt.subplot(n, 2, 2*index)
        plt.imshow(reconstructed,  cmap = 'viridis')
        plt.title('Reconstructed')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return plt.show()


def plot_original_reconstructed_raw(model, test, n_figures = 5):

    n=n_figures
    plt.figure(figsize=(10, 3*n))
    for index, (audio, copy) in zip(range(1, n + 1), test.unbatch().shuffle(buffer_size = n+1).take(n)):
        samplerate = 44100
        display(ipd.Audio(data = audio.numpy().reshape(220500), rate=samplerate))
        reconstructed = model.predict(np.expand_dims(audio, axis=0))
        display(ipd.Audio(data = reconstructed.reshape(220500), rate=samplerate) )

        ax = plt.subplot(n, 2, 2*index-1)
        plt.plot(audio.numpy())
        plt.title('Original')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        reconstructed = model.predict(np.expand_dims(audio, axis=0), verbose=0)
        ax = plt.subplot(n, 2, 2*index)
        plt.plot(reconstructed.reshape(220500))
        plt.title('Reconstructed')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return plt.show()

def plot_original_reconstructed_latent(model, n_figures, test):
    # function to plot the images (STFT, MEL, or MFCC) and the reconstructed images
    # model: the autoencoder model
    # n_figures: how many images to plot
    # test: the test dataset

    encoder = model.layers[1]
    random_input = np.random.rand(1, 64, 128, 1)
    print(encoder(random_input).shape)

    n = n_figures
    plt.figure(figsize=(10, 15))
    for index, (image, copy) in zip(range(1, n + 1), test.unbatch().shuffle(buffer_size=n + 1).take(n)):
        ax = plt.subplot(n, 3, 3 * index - 2)  # Original image
        plt.imshow(image.numpy(), cmap='viridis')
        plt.title('Original')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        latent = image
        latent = np.resize(latent, (64, 128))
        latent = np.expand_dims(latent, axis=2)
        latent = np.expand_dims(latent, axis=0)
        latent = encoder.predict(latent, verbose=0)
        latent = np.squeeze(latent, axis=0)
        latent = latent.transpose(1, 2, 0)

        #min-max normalization
        latent = (latent - np.min(latent)) / (np.max(latent) - np.min(latent))

        ax = plt.subplot(n, 3, 3 * index - 1)  # Latent representation
        plt.imshow(latent, cmap='viridis', aspect=0.4)
        plt.title('Latent')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        reconstructed = model.predict(np.expand_dims(image, axis=0))
        reconstructed = reconstructed.reshape(64, 128, 1)

        ax = plt.subplot(n, 3, 3 * index)  # Reconstructed image
        plt.imshow(reconstructed, cmap='viridis')
        plt.title('Reconstructed')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return plt.show()


def plot_clip_overview_latent_space(encoder, test, label_names, n_figures = 5, column=5):
    # test must be a dataset with labels

    categories = label_names
    row = len(categories)
    
    plt.subplots(row, column, figsize=(8, 1.7 * row))
    plt.tight_layout(pad=0.7)
    
    for j, audio_type in enumerate(categories):
        test_unbatched = test.unbatch().shuffle(buffer_size=1000)
        
        def filter_func(x, y):
            return tf.math.argmax(y) == j

        test_filtered = test_unbatched.filter(filter_func)
       
        i = 0
        k = 0
        for stft, label in test_filtered.take(400):
            if i == column:
                break
            else: 
                if tf.math.argmax(label) == j:
                    #plot latent img
                    stft = np.resize(stft, (64, 128))
                    stft = np.expand_dims(stft, axis=2)
                    stft = np.expand_dims(stft, axis=0)
                    latent = encoder.predict(stft, verbose=0)
                    latent = np.squeeze(latent, axis=0)
                    latent = latent.transpose(1, 2, 0)

                    #min-max normalization
                    latent = (latent - np.min(latent)) / (np.max(latent) - np.min(latent))

                    ax = plt.subplot(row,column,j*column+i+1)
                    plt.imshow(latent, cmap='viridis')#, aspect=0.4)
                    plt.title(audio_type)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    i += 1

        #plt.colorbar()

    plt.subplots_adjust(hspace=0.5)