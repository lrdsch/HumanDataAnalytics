We describe the github folder here. We have not marked all the subfolders and all the files present. Some folders may be missing as they are too heavy to be uploaded to github. Following the name of files or folders is often a brief description of them. 


.
├── AE_Conv_prep_flatten_MFCC
│	It contains grid search results
│
├── AE_Conv_prep_flatten_STFT
│	It contains grid search results
│
├── AE_Conv_prep_flatten_STFT_Augmented
│	It contains grid search results
│
├── AudioSet.ipynb
│	This notebook was for managing the AudioSet database
│
├── Data
│	├── AudioSet
│	│	Database files
│	│
│	├── ESC-10-depth
│	│	Database files
│	│
│	├── ESC-50
│	│	Database files
│	│
│	├── ESC-50-depth
│	│	Database files
│	│
│	├── ESC-US
│	│	Database files
│	│
│	└── meta
│		│ Here we store all the metadata of the datasets ESC
│		├── dataset_ESC10.csv
│		├── dataset_ESC50.csv
│		├── ESC-US.csv
│		└── esc50.csv
│
├── Data_augmentation.ipynb
│	This notebook was for data augmentation purpose 
│
├── Dense_AE_ffnn
│	It contains grid search results
│
├── Fully_Convolutional_AE_MFCC
│	It contains grid search results
│
├── Fully_Convolutional_AE_STFT
│	It contains grid search results
│
├── LICENCE
├── Logs_Masked
│	This folder contains the logs of the training on the augmented dataset
│
├── Main_Notebook.ipynb
│	The main notebook containing all the work done
│
├── Models
│	├── ann_utils.py
│	│	Script for neural network and dataset management
│	│
│	├── basic_ml.py
│	│	Script for basic machine learning algorithms
│	│
│	├── grid_search_results.txt
│	├── models_info
│	│	Just a table of results 
│	│
│	└── old_grid_search_results.txt
│
├── notebook_2.4_RNN.ipynb
│	Section of the main notebook
│
├── notebook_3.1_Dense_AE_ffnn.ipynb
│	Section of the main notebook
│
├── notebook_3.2_AE_Conv_prep_flatten_MFCC.ipynb
│	Section of the main notebook
│
├── notebook_3.2_AE_Conv_prep_flatten_STFT_Augmented.ipynb
│	Section of the main notebook
│
├── notebook_3.2_AE_Conv_prep_flatten_STFT.ipynb
│	Section of the main notebook
│
├── notebook_3.3_Fully_Convolutional_AE_MFCC.ipynb
│	Section of the main notebook
│
├── notebook_3.3_Fully_Convolutional_AE_STFT.ipynb
│	Section of the main notebook
│
├── notebook_4.1_to_end.ipynb
│	Section of the main notebook
│
├── notebook_up_to_2.3.ipynb
│	Section of the main notebook
│
├── Papers
│	References for our work
│
├── Preprocessing
│	├── data_loader.py
│	│	Script for data loading
│	│
│	└── exploration_plots.py
│		Script for the first data exploration (initial section of notebook)
│
├── Projects_B2.pdf
│	University assignment 
│
├── removed_stuff.ipynd
│	Notebook with old or not used code but ideas
│
├── Saved_Datasets
│	Folder where we saved all the datasets
│
├── Saved_Models
│	Folder where we saved all the models
│
└── Visualization
	├── clip_utils.py
	│	Script for clip transformation
	│
	└── model_plot.py
		Script for plots production
