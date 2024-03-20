import time
import numpy as np
import pandas as pd
import sys
import random
import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
import inspect

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
from sklearn.metrics import RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
import librosa



def train_test(X, y, test_size): # nothing more than the sklearn train_test_split, just a shorter name
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)
  return X_train, X_test, y_train, y_test

def extract_flatten_MFCC(
                         audio, # already loaded with librosa or similar
                         cepstral_num = 20,  #number of cepstral coefficients
                         N_filters = 50, #number of filters in the mel filterbank
                         transpose = False #we can flatten the mfcc matrix in two ways
                         ):

    sample_rate = 44100
    segment = 20  #our default segments length in milliseconds
    overlapping = 10 #our defaut segments overlap in milliseconds
    nperseg = round(sample_rate * segment / 1000) #effettive number of point per frame
    noverlap = round(sample_rate * overlapping / 1000)
    hop_length = nperseg-noverlap
    if len(audio)>200000: #it may happend that the audio is already transformed somehow...
        mfcc_y = librosa.feature.mfcc(  y=audio, 
                                        sr=sample_rate, 
                                        n_mfcc=cepstral_num,
                                        n_fft = nperseg,  
                                        hop_length=hop_length, 
                                        htk=True, 
                                        fmin = 40,
                                        n_mels = N_filters)
        #output shape is ((len(audio)-nperseg)//hop_length + 3 )*cepstral_num
        if transpose:
            mfcc_y = np.transpose(mfcc_y)
        return mfcc_y.flatten()
    else:
        return audio
    
def extract_labels(df): #extract the category names from a dataframe (df_ESC10 or df_ESC50) in the correct 
                        # order fot the confusion matrix labels
    labels_name = {target:name for (target,name) in np.asarray(df[['target','category']])}
    my_list = list(labels_name.keys())
    my_list.sort()
    labels = [labels_name[i] for i in my_list ]
    return labels 
    

# build a dataset as numpy array in several ways
def build_dataset(df, subset = False, num_classes = 10):
    # df is df_ESC10 or df_ESC50
    if not subset and 'category' in df.columns: #df_ESC10 or df_ESC50
        audio_paths = df.full_path

        start_time = time.time()
        k = len(set(df.category))
        X_ESC = np.empty((40*k,220500)) #initialiaze an empty array
        for i, audio_path in enumerate(audio_paths):
            if (i+1)%400==0:
                print(f'Loading the {i+1}-th labelled audio.') #load the audio from the file path
            audio, _ = librosa.load(audio_path, sr=44100)  # Set sr=None to load the audio file with its original sampling rate
            X_ESC[i,:] = audio
        y_ESC = np.asarray(df['target']) # assign the target to a numpy vector (NOT one hot encoding)

        #print some statiscs 
        print(f'To build the dataset we need {round(time.time()-start_time,2)} seconds seconds.')
        print(f'The Numpy Array of the dataset occupies {sys.getsizeof(X_ESC)/1000} kbytes')

        # return also the labels for the confusion matrix
        labels = extract_labels(df)
        return X_ESC, y_ESC, labels

    # with this option we build a subset with  num_classes categories from df = df_ESC50
    elif subset and 'category' in df.columns:
        categories = list(set(df.category))

        if num_classes>50: #check that we are not asking more than 50 classes
            print('Max number of classes is 50')
            num_classes = 50

        subset = random.sample(categories,num_classes) #chose randomly the categories
        df_subset = df[df.category.isin(subset)] #extract the sub-dataframe

        start_time = time.time()
        audio_paths = df_subset.full_path
        X_subset = np.empty((40*num_classes,220500)) #initialize the empty array
        for i, audio_path in enumerate(audio_paths):
            audio, _ = librosa.load(audio_path, sr=44100)  
            X_subset[i,:] = audio
        y_subset = np.asarray(df_subset.target) # assign the target to a numpy vector (NOT one hot encoding)

        #print some statiscs 
        print(f'Random dataset with {num_classes} classes built in {round(time.time()-start_time,2)} seconds seconds.')
        print(f'Classes choosen: {set(df_subset.category)}')
        labels = extract_labels(df)
        return df_subset, X_subset, y_subset, labels
    
    # build the numpy array also for the unlabelled dataset-not practical (too much ram used)
    elif 'category' not in df.columns: 
        audio_paths = df.full_path
        start_time = time.time()
        X_ESC_US = np.zeros((20000,220500), dtype=np.float16) # we must decrease a lot the precision to fit in our RAM.
        for i,audio_path in enumerate(audio_paths):
            if i%1000==0:
                print(f'Loading the {i}-th unlabeled audio')
            audio, _ = librosa.load(audio_path, sr=44100)
            X_ESC_US[i,:] = audio
        
        #print some statistics
        print(f'To build the ESC-US array we need {round(time.time()-start_time,2)} seconds seconds.')
        print(f'The Numpy Array for ESC-US occupies {sys.getsizeof(X_ESC_US)/1000} kbytes')
        return X_ESC_US

# implement some very time expensive gris search for basic machine learning models and save the result in a file.txt
def basic_ML_experiments_gridsearch(dataset, #numpy array with the raw audio loaded
                                    target,  #numpy vector with target classes
                                    test_size = 0.25, #test ratio from the total
                                    file = r'Models\grid_search_results.txt', #file were save the result
                                    cv=3, 
                                    verbose=False, # display also the results 
                                    logistic_regression = True, #model to test
                                    SVM = True,
                                    decision_tree = True,
                                    random_forest = True,
                                    KNN = True):
    
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=test_size, random_state=123)

    #initialization params for extract_flatten_MFCC(), changed in the grid search
    mfcc = 50
    n = 160
    transpose = False
 
    # some check to retrive the name of the labelled dataset
    if dataset.shape[0] == 2000:
        name_df = 'ESC50'
    elif dataset.shape[0] == 400:
        classes = set(target)
        original = {0, 1, 10, 11, 12, 20, 21, 38, 40, 41} # the orginal 10 target classes in ESC10
        if classes == original:
            name_df = 'ESC10'
        else:
            name_df = 'random10'
    else:
        cc = len(list(set(target)))
        name_df = f'random{cc}'
    
    #grid search values for extract_flatten_MFCC()
    values_mfcc = {'mfcc': [10,20,30,40,50], 'n':[40,80,120,160]}
    mfcc_args_list = [ {'mfcc':a,'n':b} for a in values_mfcc['mfcc'] for b in values_mfcc['n']]
    
    # sklearn models
    models = [LogisticRegression(random_state=123), 
              SVC(random_state=123),
              DecisionTreeClassifier(random_state=123),
              RandomForestClassifier(random_state=123),
              KNeighborsClassifier()]
    
    #grid search parameters based on the model
    params = [{#logistic regression params
                  'MFCC__kw_args':mfcc_args_list,
                  'Classifier__C': [0.1, 0.5, 1, 5, 10, 15, 20],
                  'Classifier__fit_intercept': [True, False]                  
              },
              {#SVM params
                  'MFCC__kw_args':mfcc_args_list,
                  'Classifier__C' : [ 1 , 10  ,100 ], 
                  'Classifier__gamma': [1, 0.01, 0.001],
                  'Classifier__degree':[2,3,4],
                  'Classifier__kernel':['poly', 'rbf', 'sigmoid']
              },
              {#decision tree params
                  'MFCC__kw_args':mfcc_args_list,
                  'Classifier__max_depth': [5,10,50,None], 
                  'Classifier__min_samples_split': [2,8,20],
                  'Classifier__min_samples_leaf': [1,4,8] 
              },
              {#random forest params
                  'MFCC__kw_args':mfcc_args_list,
                  'Classifier__n_estimators':[50,150],
                  'Classifier__criterion':['gini','entropy'],
                  'Classifier__max_depth':[None,8],
                  'Classifier__min_samples_leaf':[2,3]
              },
              {#KNN params
                  'MFCC__kw_args':mfcc_args_list,
                  'Classifier__n_neighbors': [8,32,64], 
                  'Classifier__weights' : ['uniform', 'distance'],
                  'Classifier__p':[1,2]
              }
             ]

    #indicators of what models run (to run just some of them, full run time is ~24h)
    indicators = [logistic_regression,
                  SVM,
                  decision_tree,
                  random_forest,
                  KNN]

    #select only the model to test from the indicator
    models_to_test = [obj for obj, indicator in zip(models, indicators) if indicator]
    params_to_test = [par for par, indicator in zip(params, indicators) if indicator]

    #auxiliary function to apply extract_flatten_MFCC() to each audio in the dataset
    def extract_mfcc(array, mfcc, n=n, transpose=transpose):
        my_func = lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc, N_filters = n, transpose=transpose)
        return np.apply_along_axis(my_func, 1, array)
    
    # we need this function to to allow extract_mfcc to be used in the sklear GridSearch
    trans = FunctionTransformer(func = extract_mfcc, kw_args = {'mfcc':mfcc,'n':n,'transpose':transpose})

    #run the models
    for model,param_grid in zip(models_to_test, params_to_test):

        #extract the model name 
        name_model = str(model).split('(')[0]
        print(' '*15 + name_model)

        #build the skelarn pipeline
        clf_pipe = Pipeline(steps= [('MFCC', trans),
                                    ('Scaler', StandardScaler()),
                                    ('Classifier', model)])
        
        #set the verbosity
        verb=0
        if verbose:
            #clf_pipe.get_params() #uncomment this to see all the parameters
            verb = 3 

        start_time = time.time()
        clf = GridSearchCV(clf_pipe, param_grid, n_jobs=None, cv = cv, verbose = verb)
        clf.fit(X_train, y_train)
        print(f'The grid search for the model {name_model} requires {round((time.time()-start_time)/60,2)} minutes.')
        print(f'Best params for model {name_model} are:{clf.best_params_}')
        y_predict_train = clf.predict(X_train)
        y_predict_test = clf.predict(X_test)
        print(f"{name_model} Accuracy on {name_df} training set\t: {accuracy_score(y_train, y_predict_train)}")
        print(f"{name_model} Accuracy on {name_df} test set\t: {accuracy_score(y_test, y_predict_test)}")
        with open(file, 'a') as f:
            f.write('\n')
            f.write(name_model)
            f.write('\n')
            f.write(f'The grid search for the model {name_model} requires {round((time.time()-start_time)/60,2)} minutes.')
            f.write('\n')
            f.write(f'Best params for model {name_model} are:{clf.best_params_}')
            f.write('\n')
            f.write(f"{name_model} Accuracy on {name_df} training set\t: {accuracy_score(y_train, y_predict_train)}")
            f.write('\n')
            f.write(f"{name_model} Accuracy on {name_df} test set\t: {accuracy_score(y_test, y_predict_test)}")

def basic_ML_experiments(dataset, target, test_size = 0.25,
                         MFCC = True,
                         raw_audio  =True,
                         logistic_regression = True,
                         SVM = True,
                         decision_tree = True,
                         random_forest = True,
                         KNN = True):
    main_time = time.time()
    if dataset.shape[0] == 2000:
        name_df = 'ESC50'
    elif dataset.shape[0] == 400:
        classes = set(target)
        original = {0, 1, 10, 11, 12, 20, 21, 38, 40, 41}
        if classes == original:
            name_df = 'ESC10'
        else:
            name_df = 'random10'
    else:
        cc = len(list(set(target)))
        name_df = f'random{cc}'

    X_train, X_test, y_train, y_test = train_test(dataset, target, test_size)
    print(f'The shape of the train and test set for the {name_df} dataset are:')
    print(f'train shape: {X_train.shape}, \t target shape: {y_train.shape}')
    print(f'test shape: {X_test.shape}, \t target shape: {y_test.shape}')
    print('')

    data = {}

    # Checking the positivity of each indicator and adding the corresponding columns if True
    if logistic_regression:
        if raw_audio:
            data[('Logistic Regression', 'raw audio')] = [0, 0]
        if MFCC:
            data[('Logistic Regression', 'MFCC')] = [0, 0]
    if SVM:
        if raw_audio:
            data[('SVM', 'raw audio')] = [0, 0]
        if MFCC:
            data[('SVM', 'MFCC')] = [0, 0]
    if random_forest:
        if raw_audio:
            data[('Random Forest', 'raw audio')] = [0, 0]
        if MFCC:
            data[('Random Forest', 'MFCC')] = [0, 0]
    if decision_tree:
        if raw_audio:
            data[('Decision Tree', 'raw audio')] = [0, 0]
        if MFCC:
            data[('Decision Tree', 'MFCC')] = [0, 0]
    if KNN:
        if raw_audio:
            data[('KNN', 'raw audio')] = [0, 0]
        if MFCC:
            data[('KNN', 'MFCC')] = [0, 0]

    # Creating the pandas dataframe from the data dictionary
    result_df = pd.DataFrame(data, index=['train accuracy', 'test accuracy'])

    if logistic_regression:
        print(' '*15+'LOGISTIC REGRESSION')
        print('')
        if raw_audio:
            ''' the grid search over the parameters {'C': [0.1, 0.5, 1, 5, 10, 15, 20],'fit_intercept': [True, False]} 
            requires 2938 seconds. The best params for ESC10 are {'C': 0.1, 'fit_intercept': False}'''

            start_time = time.time()
            clf = LogisticRegression(C = 0.1, fit_intercept= False).fit(X_train,y_train)
            print(f'Fit the logistic regression on the {name_df} dataset with raw audio requires {round(time.time()-start_time,2)} seconds')

            y_predict_train = clf.predict(X_train)
            y_predict_test = clf.predict(X_test)
            result_df.loc['train accuracy', ('Logistic Regression', 'raw audio')] = accuracy_score(y_train, y_predict_train)
            result_df.loc['test accuracy', ('Logistic Regression', 'raw audio')] = accuracy_score(y_test, y_predict_test)
            print(f"(Logistic Regression) Accuracy on {name_df} training set with raw audio. \t: {accuracy_score(y_train, y_predict_train)}")
            print(f"(Logistic Regression) Accuracy on {name_df} test set with raw audio.\t: {accuracy_score(y_test, y_predict_test)}")
            print('')

        if MFCC:
            '''the grid search over the parameters {'C': [0.1, 0.5, 1, 5, 10, 15, 20],'fit_intercept': [True, False]} 
            requires 108 seconds. The best params are {'C': 0.1, 'fit_intercept': True}
            The grid search over the number of cepstral coefficients leads to cepstral_num = 20.
            The grid search over the number of fileter leads to N_filters=160 (even though it results in empty filter...)'''
            
            mfcc = 20
            n = 160

            X_train_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc, N_filters = n), 1, X_train)
            X_test_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc, N_filters = n), 1, X_test)

            start_time = time.time()
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C = 0.1, fit_intercept= True, max_iter = 1000))
            pipe.fit(X_train_mfcc,y_train)
            print(f'Fit the logistic regression on the {name_df} dataset with 20-MFCC and 160 filters requires {round(time.time()-start_time,2)} seconds')

            y_predict_train = pipe.predict(X_train_mfcc)
            y_predict_test = pipe.predict(X_test_mfcc)
            result_df.loc['train accuracy', ('Logistic Regression', 'MFCC')] = accuracy_score(y_train, y_predict_train)
            result_df.loc['test accuracy', ('Logistic Regression', 'MFCC')] = accuracy_score(y_test, y_predict_test)
            print(f"(Logistic Regression) Accuracy on {name_df} training set with {mfcc}-MFCC and {n} filters. \t: {accuracy_score(y_train, y_predict_train)}")
            print(f"(Logistic Regression) Accuracy on {name_df} test set with {mfcc}-MFCC and {n} filters.\t: {accuracy_score(y_test, y_predict_test)}")
            print('')

    if SVM:
        print(' '*15+'SVM')
        print('')
        if raw_audio:
            print('Using raw audio with SVM is unfeasible')
            print('')
        if MFCC:
            '''The grid search over the parameters {'C' : [0.01 , 0.1 , 1 , 10  ,100 ], 'kernel':['linear', 'rbf', 'sigmoid'],
            cepstral_num : [10,20,30,40,50], N_filters:[40,80,120,160] } gives as result that the best params are 
            {'C': 100, 'kernel': 'rbf', cepstral_num = 50, N_fitlers = 160}
            (SVM) Accuracy on training set with 50 MFCC and 160 filters. 	: 1.0
            (SVM) Accuracy on test set with 50 MFCC and 160 filters.	: 0.75'''
            mfcc = 50
            n = 160

            X_train_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc, N_filters = n), 1, X_train)
            X_test_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc , N_filters = n), 1, X_test)
            start_time = time.time()
            clf = SVC(C=100, kernel = 'rbf', random_state=123)
            clf.fit(X_train_mfcc,y_train)
            print(f'Fit the SVM on the {name_df} dataset with {mfcc}-MFCC and {n} filters requires {round(time.time()-start_time,2)} seconds')


            y_predict_train = clf.predict(X_train_mfcc)
            y_predict_test = clf.predict(X_test_mfcc)
            result_df.loc['train accuracy', ('SVM', 'MFCC')] = accuracy_score(y_train, y_predict_train)
            result_df.loc['test accuracy', ('SVM', 'MFCC')] = accuracy_score(y_test, y_predict_test)
            print(f"(SVM) Accuracy on {name_df} training set with {mfcc} MFCC and {n} filters. \t: {accuracy_score(y_train, y_predict_train)}")
            print(f"(SVM) Accuracy on {name_df} test set with {mfcc} MFCC and {n} filters.\t: {accuracy_score(y_test, y_predict_test)}")
            print('')

    if decision_tree:
        print(' '*15+'DECISION TREE')
        print('')
        if raw_audio:
            '''The grid search over the parameters {'max_depth': [5,10,50,None], 'min_samples_split': [2,8,20], 'min_samples_leaf': [1,4,8]}
            gives as result for ESC10 {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 20}
            (Decision Tree)) Accuracy on training set with raw audio. 	: 0.8266666666666667
            (Decision Tree) Accuracy on test set with raw audio.	: 0.22
            '''

            start_time = time.time()
            clf = DecisionTreeClassifier(max_depth= 10, min_samples_leaf= 1, min_samples_split= 20, random_state=123)
            clf.fit(X_train,y_train)
            print(f'Fit the Decision Tree on the {name_df} dataset with raw audio requires {round(time.time()-start_time,2)} seconds')

            y_predict_train = clf.predict(X_train)
            y_predict_test = clf.predict(X_test)
            result_df.loc['train accuracy', ('Decision Tree', 'raw audio')] = accuracy_score(y_train, y_predict_train)
            result_df.loc['test accuracy', ('Decision Tree', 'raw audio')] = accuracy_score(y_test, y_predict_test)
            print(f"(Decision Tree) Accuracy on {name_df} training set with raw audio. \t: {accuracy_score(y_train, y_predict_train)}")
            print(f"(Decision Tree) Accuracy on {name_df} test set with raw audio.\t: {accuracy_score(y_test, y_predict_test)}")
            print('')


        if MFCC:

            '''The grid search over the parameters { 'max_depth': [5,10,50,None], 'min_samples_split': [2,8,20],'min_samples_leaf': [1,4,8]
            ,cepstral_num : [10,20,30,40,50], N_filters:[40,80,120,160] } gives as result that the best params are 
            {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, cepstral_num = 10, N_filters = 120}
            (Decision Tree) Accuracy on training set with 10 MFCC and 120 filters. 	: 0.926
            (Decision Tree) Accuracy on test set with 10 MFCC and 120 filters.	: 0.5  '''
            mfcc = 10
            n = 120

            X_train_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc, N_filters = n), 1, X_train)
            X_test_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc , N_filters = n), 1, X_test)
            start_time = time.time()
            clf = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 1, min_samples_split=2, random_state=123)
            clf.fit(X_train_mfcc,y_train)
            print(f'Fit the Decision Tree on the {name_df} dataset with {mfcc}-MFCC and {n} filters requires {round(time.time()-start_time,2)} seconds')

            y_predict_train = clf.predict(X_train_mfcc)
            y_predict_test = clf.predict(X_test_mfcc)
            result_df.loc['train accuracy', ('Decision Tree', 'MFCC')] = accuracy_score(y_train, y_predict_train)
            result_df.loc['test accuracy', ('Decision Tree', 'MFCC')] = accuracy_score(y_test, y_predict_test)
            print(f"(Decision Tree) Accuracy on {name_df} training set with {mfcc} MFCC and {n} filters. \t: {accuracy_score(y_train, y_predict_train)}")
            print(f"(Decision Tree) Accuracy on {name_df} test set with {mfcc} MFCC and {n} filters.\t: {accuracy_score(y_test, y_predict_test)}")
            print('')

    if random_forest:
        print(' '*15+'RANDOM FOREST')
        print('')
        if raw_audio:
            '''The grid search over the parameters {'n_estimators':[50,100,150],'criterion':['gini','entropy'],
            'max_depth':[None,8,10,12],'min_samples_leaf':[1,2,3]}
            gives as result {'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 3, 'n_estimators': 100}
            (Random Forest)) Accuracy on training set with raw audio. 	: 0.9966666666666667
            (Random Forest) Accuracy on test set with raw audio.	: 0.26

            '''
            start_time = time.time()
            clf = RandomForestClassifier(random_state=123,criterion = 'gini', max_depth =  None, min_samples_leaf=3, n_estimators = 100 )
            clf.fit(X_train,y_train)
            print(f'Fit the Random Forest on the {name_df} dataset with raw audio requires {round(time.time()-start_time,2)} seconds seconds')

            y_predict_train = clf.predict(X_train)
            y_predict_test = clf.predict(X_test)
            result_df.loc['train accuracy', ('Random Forest', 'raw audio')] = accuracy_score(y_train, y_predict_train)
            result_df.loc['test accuracy', ('Random Forest', 'raw audio')] = accuracy_score(y_test, y_predict_test)
            print(f"(Random Forest) Accuracy on {name_df} training set with raw audio. \t: {accuracy_score(y_train, y_predict_train)}")
            print(f"(Random Forest) Accuracy on {name_df} test set with raw audio.\t: {accuracy_score(y_test, y_predict_test)}")
            print('')


        if MFCC:
            '''The grid search over the parameters parameters {'n_estimators':[50,100,150],'criterion':['gini','entropy'],
            'max_depth':[None,8,10,12],'min_samples_leaf':[1,2,3], cepstral_num : [10,20,30,40,50], N_filters:[40,80,120,160] }
            gives as result that the best params are {'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 2, 
            'n_estimators': 150, cepstral_num = 30, N_filters = 40}
            (Random Forest) Accuracy on training set with 30 MFCC and 40 filters. 	: 1.0
            (Random Forest) Accuracy on test set with 30 MFCC and 40 filters.	: 0.7
            '''

            mfcc = 30
            n = 40

            X_train_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc, N_filters = n), 1, X_train)
            X_test_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc , N_filters = n), 1, X_test)
            start_time = time.time()
            clf = RandomForestClassifier(random_state=123, criterion = 'gini', max_depth = 8, min_samples_leaf = 2, n_estimators = 150 )
            clf.fit(X_train_mfcc,y_train)
            print(f'Fit the Random Forest on the {name_df} dataset with {mfcc}-MFCC and {n} filters requires {round(time.time()-start_time,2)} seconds seconds')


            y_predict_train = clf.predict(X_train_mfcc)
            y_predict_test = clf.predict(X_test_mfcc)
            result_df.loc['train accuracy', ('Random Forest', 'MFCC')] = accuracy_score(y_train, y_predict_train)
            result_df.loc['test accuracy', ('Random Forest', 'MFCC')] = accuracy_score(y_test, y_predict_test)
            print(f"(Random Forest) Accuracy on {name_df} training set with {mfcc} MFCC and {n} filters. \t: {accuracy_score(y_train, y_predict_train)}")
            print(f"(Random Forest) Accuracy on {name_df} test set with {mfcc} MFCC and {n} filters.\t: {accuracy_score(y_test, y_predict_test)}")
            print('')

    if KNN:
        print(' '*15+'KNN')
        print('')
        if raw_audio:
            '''The grid search over the parameters {'n_neighbors': [8,32,64], 'weights' : ['uniform', 'distance'],'p':[1,2]} 
            gives as result that best pars are {'n_neighbors': 32, 'p': 1, 'weights': 'uniform'}
            (KNN) Accuracy on training set with raw audio. 	: 0.10666666666666667
            (KNN) Accuracy on test set with raw audio.	: 0.08
            '''
            start_time = time.time()
            clf = KNeighborsClassifier(n_neighbors = 32, p = 1, weights = 'uniform')
            clf.fit(X_train,y_train)
            print(f'Fit the KNN on the {name_df} dataset with raw audio requires {round(time.time()-start_time,2)} seconds')

            y_predict_train = clf.predict(X_train)
            y_predict_test = clf.predict(X_test)
            result_df.loc['train accuracy', ('KNN', 'raw audio')] = accuracy_score(y_train, y_predict_train)
            result_df.loc['test accuracy', ('KNN', 'raw audio')] = accuracy_score(y_test, y_predict_test)
            print(f"(KNN) Accuracy on {name_df} training set with raw audio. \t: {accuracy_score(y_train, y_predict_train)}")
            print(f"(KNN) Accuracy on {name_df} test set with raw audio.\t: {accuracy_score(y_test, y_predict_test)}")
            print('')


        if MFCC:
            '''The grid search over the parameters parameters {'n_neighbors': [8,32,64], 'weights' : ['uniform', 'distance'],'p':[1,2]},
            , cepstral_num : [10,20,30,40,50], N_filters:[40,80,120,160]}
            gives as result that the best params are {'n_neighbors': 8, 'p': 1, 'weights': 'distance', cepstral_num :'10', N_filters: '80'}
            (KNN) Accuracy on training set with 10 MFCC and 80 filters. 	: 1.0
            (KNN) Accuracy on test set with 10 MFCC and 80 filters.	: 0.74
            '''

            mfcc = 10
            n = 80

            X_train_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc, N_filters = n), 1, X_train)
            X_test_mfcc = np.apply_along_axis(lambda x: extract_flatten_MFCC(audio = x, cepstral_num = mfcc , N_filters = n), 1, X_test)
            start_time = time.time()
            clf = KNeighborsClassifier(n_neighbors = 8, p = 1, weights = 'distance')
            clf.fit(X_train_mfcc,y_train)
            print(f'Fit the KNN on the {name_df} dataset with {mfcc}-MFCC and {n} filters requires {round(time.time()-start_time,2)} seconds')


            y_predict_train = clf.predict(X_train_mfcc)
            y_predict_test = clf.predict(X_test_mfcc)
            result_df.loc['train accuracy', ('KNN', 'MFCC')] = accuracy_score(y_train, y_predict_train)
            result_df.loc['test accuracy', ('KNN', 'MFCC')] = accuracy_score(y_test, y_predict_test)
            print(f"(KNN) Accuracy on {name_df} training set with {mfcc} MFCC and {n} filters. \t: {accuracy_score(y_train, y_predict_train)}")
            print(f"(KNN) Accuracy on {name_df} test set with {mfcc} MFCC and {n} filters.\t: {accuracy_score(y_test, y_predict_test)}")
            print('')
    
    print(f'The full basic machine learning experimentation requires {round((time.time()-main_time)/60,2)} minutes.')
    return result_df  

'''
FUNCTIONS TO RETRIVE NAMES NOT UTILIZED

#function to extract the pointer's name
def get_var_name(variable):
    #a = 2  >>  get_var_name(a) = ['a']
    globals_dict = globals()
    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable]
    #return ['c_ciaone']

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def retrieve_name2(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]
'''