import scipy.stats
import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from neural_networks import MLP_TF, CNN_TF
import copy
from tensorflow.keras.models import clone_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# ------------------ MLP ------------------

def extract_features(audio_file, target_duration=4, target_sr=44100):
    '''
    Extracts audio features from the given audio file.
    Args:
        audio_file (str): The path to the audio file.
        target_duration (float): The target duration of the audio in seconds.
        target_sr (int): The target sample rate of the audio.
    Returns:
        dict: A dictionary containing the extracted features.
    '''


    # Load audio and define target sample rate
    y, sr = librosa.load(audio_file, sr=target_sr)

    # Ensure audio length is the same as the target duration using zero padding
    target_length = int(target_sr * target_duration)
    y = librosa.util.fix_length(y, size=target_length)

    # Normalize amplitude
    y = librosa.util.normalize(y)

    # Define variables for some features
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # spectral feature list
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    crossing_rate = librosa.feature.zero_crossing_rate(y)

    # rhythm feature list
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    fourier_tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)

    # Create a dictionary to store the mean and flattened values of the features
    features = {
        'chroma_stft_mean': np.mean(chroma_stft),
        'chroma_cqt_mean': np.mean(chroma_cqt),
        'chroma_cens_mean': np.mean(chroma_cens),
        'melspectogram_mean': np.mean(melspectogram),
        'rms_mean': np.mean(rms),
        'centroid_mean': np.mean(centroid),
        'bandwidth_mean': np.mean(bandwidth),
        'contrast_mean': np.mean(contrast),
        'flatness_mean': np.mean(flatness),
        'rolloff_mean': np.mean(rolloff),
        'crossing_rate_mean': np.mean(crossing_rate),
        'tempogram_mean': np.mean(tempogram),
        'fourier_tempogram_mean': np.mean(fourier_tempogram)
    }

    # Add the mean of each MFCC feature to the dictionary
    for i in range(1, 41):
        features[f"mfcc_{i}_mean"] = np.mean(mfccs[i - 1])

    return features


def process_data(base_dir):
    '''
    Extracts features from audio files in the given directory and saves them in CSV files.
    Args:
        base_dir (str): The base directory containing the audio files.
    '''
    for folder in os.listdir(base_dir):
        label_list = []
        features_list = []
        fold_dir = os.path.join(base_dir, folder)
        if os.path.isdir(fold_dir):
            for filename in os.listdir(fold_dir):
                file_path = os.path.join(fold_dir, filename)
                if filename.endswith('.wav'):
                    label = filename
                    features = extract_features(file_path)
                    features_list.append(features)
                    label_list.append(label)

        # create DataFrame for each folder
        df = pd.DataFrame(features_list)
        df['Label'] = label_list
        # save DataFrame as a CSV file
        df.to_csv('data/urbansounds_features' + folder + '.csv', index=False)
        

def deep_copy_mlp(model_instance):
    '''
    Creates a deep copy of the MLPClassifier instance, including the Sequential model.
    
    Args:
        model_instance (MLPClassifier): The instance of MLPClassifier to be copied.
    
    Returns:
        MLPClassifier: A new instance of MLPClassifier with a cloned model.
    '''
    # Create a shallow copy of the MLPClassifier instance
    copied_instance = copy.copy(model_instance)
    
    # Deep copy the parameters and other attributes if necessary
    copied_instance.params = copy.deepcopy(model_instance.params)
    
    # Clone the Keras model
    if model_instance.model is not None:
        copied_instance.model = clone_model(model_instance.model)
        # Compile the cloned model with the same configuration
        copied_instance.model.compile(
            optimizer=model_instance.model.optimizer.__class__(**model_instance.model.optimizer.get_config()),
            loss=model_instance.model.loss,
            metrics=model_instance.model.metrics,
        )
    else:
        copied_instance.model = None
    
    return copied_instance


def param_tuning_mlp(datasets, model, params):
    '''
    Perform hyperparameter tuning for the MLP model using a validation set, based on a Grid Search.

    Args:
        datasets (list): A list of datasets split into training, validation, and test sets.
        model (object): The neural network model object to be tuned.
        params (dict): A dictionary containing possible values for hyperparameters.
                       This dictionary is used to generate all combinations of parameters.

    Returns:
        dict: The best combination of hyperparameters with the lowest validation loss.
    '''

    # select the datasets that simulate the 1st iteraction
    test_set = datasets[0]
    validation_set = datasets[1]
    train_set = pd.concat(datasets[2:])
    best_val_loss = float('inf')

    # separate data and labels
    X_train, y_train = train_set.iloc[:, :-1].values, train_set.iloc[:, [-1]].values
    X_val, y_val = validation_set.iloc[:, :-1].values, validation_set.iloc[:, [-1]].values
    X_test, y_test = test_set.iloc[:, :-1].values, test_set.iloc[:, [-1]].values

    # convert data to tensors for TensorFlow input
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    # generate all combinations of parameters - Grid Search
    param_grid = ParameterGrid(params)

    # iterate through all parameter combinations and chose the one with the lowest validation loss
    for param_combination in param_grid:
        print("testing with parameters: ", param_combination)
        model_instance = model.copy()
        model_instance.build_model(hidden_layers=param_combination['hidden_layers'],
                                    learning_rate=param_combination['learning_rate'],
                                    dropout_rate=param_combination['dropout_rate'],
                                    optimizer=param_combination['optimizer'],
                                    l2_lambda=param_combination['l2_lambda'])
        model_instance.train(X_train, y_train, X_val, y_val, epochs=param_combination['epochs'], batch_size=param_combination['batch_size'])
        val_loss = model_instance.history.history['val_loss'][-1]
        print("Loss value (validation) for this combination: ", val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = param_combination

    return best_params


def cross_validation_mlp(datasets, model, params):
    '''
    Perform cross-validation on the datasets using the given model and parameters.

    Args:
        datasets (list): A list of datasets split into folds for cross-validation. Each dataset is assumed to be a Pandas DataFrame.
        model (object): The neural network model object to be trained and evaluated.
        params (dict): A dictionary containing the hyperparameters for training the model.

    Returns:
        tuple: A tuple containing:
            - accuracy_values (list): List of accuracy values for each fold.
            - loss_values (list): List of loss values for each fold.
            - cumulative_matrix (np.ndarray): A confusion matrix accumulated over all folds.
    '''

    accuracy_values = []
    loss_values = []
    cumulative_matrix = None
    num_classes = 10

    for i in range(len(datasets)):
        # select the datasets for test/validation/train sets
        test_set = datasets[i]
        validation_set = datasets[(i+1) % len(datasets)]
        train_set = pd.concat(datasets[:i] + datasets[(i+1) % len(datasets):])

        # separate data and labels
        X_train, y_train = train_set.iloc[:, :-1].values, train_set.iloc[:, [-1]].values
        X_val, y_val = validation_set.iloc[:, :-1].values, validation_set.iloc[:, [-1]].values
        X_test, y_test = test_set.iloc[:, :-1].values, test_set.iloc[:, [-1]].values

        # convert data to tensors for TensorFlow input
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

        y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
        
        # reset the model for each fold
        model_instance = model.copy()


        model_instance.build_model(hidden_layers=params['hidden_layers'],
                                        learning_rate=params['learning_rate'],
                                        dropout_rate=params['dropout_rate'],
                                        optimizer=params['optimizer'],
                                        l2_lambda=params['l2_lambda'])
    
        model_instance.train(X_train, y_train, X_val, y_val, epochs=params['epochs'], batch_size=params['batch_size'])
        

        fold_loss, fold_accuracy = model_instance.evaluate(X_test, y_test)
        accuracy_values.append(fold_accuracy)
        loss_values.append(fold_loss)

        # get predictions for the test set
        y_pred = model_instance.predict(X_test)
        # compute the confusion matrix for this fold and add to the cumulative one
        cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
        if cumulative_matrix is None:
            cumulative_matrix = cm
        else:
            cumulative_matrix += cm
        

        model_instance.plot_graphs(X_test, y_test, class_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'])

    return accuracy_values, loss_values, cumulative_matrix




# ------------------ CNN ------------------

def extract_spectrogram(audio_path, target_duration=4.0, sr=22050, n_mels=128, hop_length=512):
    '''
    Extracts the mel-spectrogram from an audio file with a fixed duration.

    Args:
        audio_path (str): Path to the audio file.
        target_duration (float): Duration in seconds to standardize the audio. If the audio is longer, it will be truncated; if shorter, it will be padded.
        sr (int): Sampling rate for the audio (default: 22050 Hz, standard for librosa).
        n_mels (int): Number of Mel bands to generate in the spectrogram.
        hop_length (int): Number of samples between successive frames in the spectrogram.

    Returns:
        ndarray: Log-scaled mel-spectrogram with consistent dimensions (n_mels x time frames).
    '''

    # load the audio file
    y, sr = librosa.load(audio_path, sr=sr, duration=target_duration)

    # extract the mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)

    # converte to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec


def plot_spectrogram(spectrogram, sr=22050, title="", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    img = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    plt.colorbar(img, ax=ax)  

def extract_mfcc_features(audio_path, target_duration=4.0, sr=22050, n_mfcc=40, n_fft=2048, hop_length=512):
    """
    Extrai os MFCCs de um arquivo de áudio com duração fixa.

    Args:
        audio_path (str): Caminho para o arquivo de áudio.
        target_duration (float): Duração em segundos para padronizar o áudio.
        sr (int): Taxa de amostragem.
        n_mfcc (int): Número de coeficientes MFCC.
        n_fft (int): Tamanho da FFT.
        hop_length (int): Tamanho do salto entre frames.

    Returns:
        ndarray: Matriz de MFCCs com dimensões consistentes.
    """
    # Carregar o áudio com duração fixa
    y, sr = librosa.load(audio_path, sr=sr, duration=target_duration)

    # Extrair os MFCCs
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    return mfcc_features


def plot_mfcc(spectrogram, sr=22050, title="", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    img = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    plt.colorbar(img, ax=ax)  


def extract_chroma_features(audio_path, target_duration=4.0, sr=22050, n_chroma=12, hop_length=512):
    """
    Extrai as features chroma de um arquivo de áudio com duração fixa.

    Args:
        audio_path (str): Caminho para o arquivo de áudio.
        target_duration (float): Duração em segundos para padronizar o áudio.
        sr (int): Taxa de amostragem.
        n_chroma (int): Número de bandas chroma (típico é 12).
        hop_length (int): Tamanho do salto entre frames.

    Returns:
        ndarray: Matriz de features chroma com dimensões consistentes.
    """
    # Carregar o áudio com duração fixa
    y, sr = librosa.load(audio_path, sr=sr, duration=target_duration)

    # Extrair as features chroma
    chroma_features = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma, hop_length=hop_length)

    return chroma_features

def plot_chroma(spectrogram, sr=22050, title="", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    img = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    plt.colorbar(img, ax=ax)  

def normalize_and_resize(spectrograms):
    
    resized_spectrograms = []
    
    global_mean = np.mean(spectrograms)
    global_std = np.std(spectrograms)
    
    for spec in spectrograms:
    # Padronizar o espectrograma
        standardized_spec = (spec - global_mean) / global_std
        
        resized_spectrograms.append(standardized_spec)
        
        
    # Converter para ndarray e adicionar uma dimensão extra (canal)
    resized_spectrograms = np.array(resized_spectrograms)
    return np.expand_dims(resized_spectrograms, axis=-1)


def param_tuning_cnn(folds_data, params):
    """
    Performs parameter tuning for the CNN using a validation set, based on the accuracy metric.

    Args:
        folds_data (dict): Dictionary with fold names as keys and tuples of (data, labels) as values.
        params (dict): Grid of parameters to test.

    Returns:
        best_params (dict): Best parameters based on validation accuracy.
    """
    fold_names = list(folds_data.keys())  # Get fold names
    if len(fold_names) < 2:
        raise ValueError("At least two folds are required for parameter tuning (validation and training).")
    
    # select validation and remaining folds for training
    val_fold_name = fold_names[0]  # Use the first fold as validation set
    validation_set = folds_data[val_fold_name]

    train_fold_names = fold_names[1:]  # Use the rest as training folds
    train_spectrograms = np.concatenate([folds_data[name][0] for name in train_fold_names])
    train_labels = np.concatenate([folds_data[name][1] for name in train_fold_names])

    # Separate data and labels
    X_train, y_train = train_spectrograms, train_labels
    X_val, y_val = validation_set

    # Normalize and resize datasets
    X_train = normalize_and_resize(X_train)
    X_val = normalize_and_resize(X_val)

    best_val_acc = 0
    best_params = None

    # Generate parameter grid
    param_grid = ParameterGrid(params)
    print("PARAM_GRID: ", param_grid)

    # Iterate through parameter combinations
    for param in param_grid:
        print("Testing with parameters: ", param)

        # Initialize and train CNN
        cnn = CNN_TF.CNN_Class(input_shape=X_train.shape[1:], num_classes=len(np.unique(y_train)))
        cnn.initialize_model(param)
        
        cnn.trainar(
            train_data=X_train,
            train_labels=y_train,
            validation_data=(X_val, y_val),
            epochs=param['epochs'],
            batch_size=param['batch_size']
        )
        
        # Evaluate on validation set
        val_loss, val_acc = cnn.evaluate(X_val, y_val)
        print("Validation accuracy for this combination: ", val_acc)

        # Update best parameters if current accuracy is higher
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = param

    return best_params


def split_folds(folds_data):
    '''
    Divids the folds in test/validation/train sets 

    Args:
        folds_data (dict): data organized by fold.

    Returns:
        dict: data divided in test, validation and training folds
    '''
    fold_names = list(folds_data.keys())

    # Separar o último fold como teste
    test_fold_name = fold_names[-1]
    test_fold = folds_data[test_fold_name]

    # Separar o penúltimo fold como validação
    val_fold_name = fold_names[-2]
    val_fold = folds_data[val_fold_name]

    # Restantes folds para treino
    train_fold_names = fold_names[:-2]
    train_folds = {name: folds_data[name] for name in train_fold_names}

    return train_folds, val_fold, test_fold


def cross_validation_cnn(folds_data, best_params):
    """
    Performs cross-validation on folds and uses the best parameters after the first iteration.

    Args:
        folds_data (dict): Input data organized by fold.
        params (dict): Grid of parameters to use.
        best_params (dict, optional): Previously known best parameters.

    Returns:
        accuracy_values: List of accuracies per fold.
        loss_values: List of losses per fold.
    """
    accuracy_values = []
    loss_values = []
    fold_names = list(folds_data.keys())  # Get the fold names
    num_classes = 10   
    cumulative_matrix = None

    for i, test_fold_name in enumerate(fold_names):
        print(f"=== Iteration {i + 1} ===")

        test_set = folds_data[test_fold_name]
        val_fold_name = fold_names[(i + 1) % len(fold_names)]
        validation_set = folds_data[val_fold_name]

        train_fold_names = [name for name in fold_names if name != test_fold_name and name != val_fold_name]
        train_spectrograms = np.concatenate([folds_data[name][0] for name in train_fold_names])
        train_labels = np.concatenate([folds_data[name][1] for name in train_fold_names])

        X_train, y_train = train_spectrograms, train_labels
        X_val, y_val = validation_set
        X_test, y_test = test_set

        # Normalize and resize data
        X_train = normalize_and_resize(X_train)
        X_val = normalize_and_resize(X_val)
        X_test = normalize_and_resize(X_test)

        print("Using Best Parameters: ", best_params)

        cnn = CNN_TF.CNN_Class(input_shape=X_train.shape[1:], num_classes=len(np.unique(y_train)))
        cnn.initialize_model(best_params)
        
        history = cnn.trainar(
            train_data=X_train,
            train_labels=y_train,
            validation_data=(X_val, y_val),
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size']
        )

        fold_loss, fold_accuracy = cnn.evaluate(X_test, y_test)
        accuracy_values.append(fold_accuracy)
        loss_values.append(fold_loss)

         # get predictions for the test set
        y_pred = cnn.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1) 
        # compute the confusion matrix for this fold and add to the cumulative one
        cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
        if cumulative_matrix is None:
            cumulative_matrix = cm
        else:
            cumulative_matrix += cm

        cnn.plot_graphs(history, X_test, y_test, class_labels=['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'])

    return accuracy_values, loss_values, cumulative_matrix








# ------------------ DeepFool ------------------

# def deepfool_mlp(model, x0, y0, max_iter=50, epsilon=1e-6):
#     """    
#     Args:
#         model: A TensorFlow/Keras model.
#         x0: The input image (numpy array of shape [H, W, C] or [1, H, W, C]).
#         y0: The true label (integer).
#         max_iter: Maximum number of iterations for the attack.
#         epsilon: A small value to avoid division by zero.
    
#     Returns:
#         x_adv: The adversarial example.
#     """
#     x = tf.convert_to_tensor(x0, dtype=tf.float32)
#     x_adv = tf.identity(x)  # Initialize adversarial example as the original input
#     logits = model(x_adv)  # Get initial logits
#     pred_label = tf.argmax(logits, axis=-1).numpy()[0]
    
#     if pred_label != y0:
#         return x0  # If already misclassified, return the original input

#     # Iterative DeepFool
#     for i in range(max_iter):
#         with tf.GradientTape() as tape:
#             tape.watch(x_adv)
#             logits = model(x_adv)  # Shape: (1, num_classes)
        
#         gradients = tape.gradient(logits, x_adv)  # Shape: (1, H, W, C)
#         logits = logits.numpy()[0]  # Convert logits to 1D array (num_classes)
#         gradients = gradients.numpy()  # Convert gradients to numpy array

#         current_label = np.argmax(logits)
#         if current_label != y0:
#             break  # Stop if misclassified

#         # Compute perturbation for each class
#         w = gradients[0] - gradients[0][y0]  # Fix shape mismatch
#         f = logits - logits[y0]

#         perturbations = []
#         for k in range(len(logits)):  # Iterate over all classes
#             if k != y0:
#                 # Avoid division by zero in norm calculation
#                 norm_w_k = np.linalg.norm(w[k].flatten()) + epsilon
#                 perturbations.append((abs(f[k]) / norm_w_k, k))
        
#         # Find the closest class decision boundary
#         perturbations.sort()
#         r_min, k_min = perturbations[0]

#         # Apply the perturbation
#         x_adv = x_adv + (1 + epsilon) * tf.convert_to_tensor(r_min * w[k_min], dtype=tf.float32)

#     return x_adv.numpy()


# def cross_validation_mlp_deepfool(datasets, model, params):
#     accuracy_values = []
#     loss_values = []
#     robustness_values = []

#     for i in range(len(datasets)):
#         print(f"=== Fold {i+1} ===")

#         # Split the datasets
#         test_set = datasets[i]
#         validation_set = datasets[(i + 1) % len(datasets)]
#         train_set = pd.concat(datasets[:i] + datasets[(i + 1) % len(datasets):])

#         X_train, y_train = train_set.iloc[:, :-1].values, train_set.iloc[:, -1].values
#         X_val, y_val = validation_set.iloc[:, :-1].values, validation_set.iloc[:, -1].values
#         X_test, y_test = test_set.iloc[:, :-1].values, test_set.iloc[:, -1].values

#         # Convert data to tensors
#         X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
#         X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
#         X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

#         y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
#         y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
#         y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

#         # Initialize and train the model
#         model_instance = model.copy()
#         model_instance.build_model(
#             hidden_layers=params['hidden_layers'],
#             learning_rate=params['learning_rate'],
#             dropout_rate=params['dropout_rate'],
#             optimizer=params['optimizer'],
#             l2_lambda=params['l2_lambda']
#         )
#         model_instance.train(
#             X_train, y_train, X_val, y_val,
#             epochs=params['epochs'],
#             batch_size=params['batch_size']
#         )

#         # Evaluate the model on the test set
#         fold_loss, fold_accuracy = model_instance.evaluate(X_test, y_test)
#         accuracy_values.append(fold_accuracy)
#         loss_values.append(fold_loss)

#         # Apply DeepFool to evaluate robustness
#         adversarial_examples = []
#         for j in range(len(X_test)):
#             x0 = tf.expand_dims(X_test[j], axis=0)  # Single test example
#             y0 = y_test[j]  # True label for this example

#             try:
#                 # Generate adversarial example
#                 x_adv = deepfool_mlp(model_instance.model, x0, y0)
#                 adversarial_examples.append(x_adv)

#                 # Evaluate adversarial example
#                 adv_pred = tf.argmax(model_instance.model(x_adv), axis=-1)
#                 if adv_pred != tf.cast(y0, dtype=tf.int64):
#                     print(f"Example {j}: Successfully fooled. True: {y0}, Adversarial: {adv_pred}")
#                 else:
#                     print(f"Example {j}: Failed to fool. True: {y0}, Adversarial: {adv_pred}")

#             except ValueError as e:
#                 print(f"DeepFool error on example {j}: {e}")
#                 continue

#         # Calculate robustness
#         adv_correct = 0
#         for adv_x, true_y in zip(adversarial_examples, y_test):
#             adv_pred = tf.argmax(model_instance.model(adv_x), axis=-1)
#             if adv_pred == tf.cast(true_y, dtype=tf.int64):
#                 adv_correct += 1

#         robustness_accuracy = adv_correct / len(y_test) if len(y_test) > 0 else 0.0
#         robustness_values.append(robustness_accuracy)
#         print(f"Robustness Accuracy: {robustness_accuracy}")
#         model_instance.plot_graphs(X_test, y_test, class_labels=['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'])

#     return accuracy_values, loss_values, robustness_values


# def deepfool_cnn(model, x0, y0, max_iter=50, epsilon=1e-6):
#     """    
#     Args:
#         model: A TensorFlow/Keras model.
#         x0: The input image (numpy array of shape [H, W, C] or [1, H, W, C]).
#         y0: The true label (integer).
#         max_iter: Maximum number of iterations for the attack.
#         epsilon: A small value to avoid division by zero.
    
#     Returns:
#         x_adv: The adversarial example.
#     """
#     x = tf.convert_to_tensor(x0, dtype=tf.float32)
#     x_adv = tf.identity(x)  # Initialize adversarial example as the original input
#     logits = model(x_adv)  # Get initial logits
#     pred_label = tf.argmax(logits, axis=-1).numpy()[0]
    
#     if pred_label != y0:
#         return x0  # If already misclassified, return the original input

#     # Iterative DeepFool
#     for i in range(max_iter):
#         with tf.GradientTape() as tape:
#             tape.watch(x_adv)
#             logits = model(x_adv)  # Shape: (1, num_classes)
        
#         gradients = tape.gradient(logits, x_adv)  # Shape: (1, H, W, C)
#         logits = logits.numpy()[0]  # Convert logits to 1D array (num_classes)
#         gradients = gradients.numpy()  # Convert gradients to numpy array

#         current_label = np.argmax(logits)
#         if current_label != y0:
#             break  # Stop if misclassified

#         # Compute perturbation for each class
#         w = gradients[0] - gradients[0][y0]  # Fix shape mismatch
#         f = logits - logits[y0]

#         perturbations = []
#         for k in range(len(logits)):  # Iterate over all classes
#             if k != y0:
#                 # Avoid division by zero in norm calculation
#                 norm_w_k = np.linalg.norm(w[k].flatten()) + epsilon
#                 perturbations.append((abs(f[k]) / norm_w_k, k))
        
#         # Find the closest class decision boundary
#         perturbations.sort()
#         r_min, k_min = perturbations[0]

#         # Apply the perturbation
#         x_adv = x_adv + (1 + epsilon) * tf.convert_to_tensor(r_min * w[k_min], dtype=tf.float32)

#     return x_adv.numpy()



# def cross_validation_cnn_deepfool(folds_data, params, best_params=None):
#     accuracy_values = []
#     loss_values = []
#     robustness_values = []
#     best_val_acc = 0
#     fold_names = list(folds_data.keys())  # Obter as chaves do dicionário

#     for i, test_fold_name in enumerate(fold_names):
#         print(f"=== Iteração {i+1} ===")
#         test_set = folds_data[test_fold_name]
#         val_fold_name = fold_names[(i + 1) % len(fold_names)]
#         validation_set = folds_data[val_fold_name]

#         # Prepare training data
#         train_fold_names = [name for name in fold_names if name != test_fold_name and name != val_fold_name]
#         train_spectrograms = np.concatenate([folds_data[name][0] for name in train_fold_names])
#         train_labels = np.concatenate([folds_data[name][1] for name in train_fold_names])

#         X_train, y_train = train_spectrograms, train_labels
#         X_val, y_val = validation_set
#         X_test, y_test = test_set

#         # Normalize and resize data
#         X_train = normalize_and_resize(X_train)
#         X_val = normalize_and_resize(X_val)
#         X_test = normalize_and_resize(X_test)

#         # Tune hyperparameters for the first iteration
#         if i == 0 and best_params is None:
#             param_grid = ParameterGrid(params)

#             for param in param_grid:
#                 print("Testing with parameters: ", param)

#                 cnn = CNN_TF.CNN_Class(input_shape=X_train.shape[1:], num_classes=len(np.unique(y_train)))

#                 cnn.initialize_model(param)
#                 history = cnn.trainar(
#                     train_data=X_train,
#                     train_labels=y_train,
#                     validation_data=(X_val, y_val),
#                     epochs=param['epochs'],
#                     batch_size=param['batch_size']
#                 )
#                 val_loss, val_acc = cnn.evaluate(X_val, y_val)
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     best_params = param

#         # Train the final model with the best parameters
#         cnn = CNN_TF.CNN_Class(input_shape=X_train.shape[1:], num_classes=len(np.unique(y_train)))
#         cnn.initialize_model(best_params)
#         history = cnn.trainar(
#             train_data=X_train,
#             train_labels=y_train,
#             validation_data=(X_val, y_val),
#             epochs=best_params['epochs'],
#             batch_size=best_params['batch_size']
#         )

#         # Evaluate on the test set
#         fold_loss, fold_accuracy = cnn.evaluate(X_test, y_test)
#         accuracy_values.append(fold_accuracy)
#         loss_values.append(fold_loss)

#         # Initialize a new list for adversarial examples in each fold
#         adversarial_examples = []  

#         # Apply DeepFool
#         for j in range(len(X_test)):  # Loop over test examples
#             x0 = X_test[j:j+1]  # Get a single test example
#             y0 = y_test[j]      # True label for this example

#             print(f"x0 shape: {x0.shape}, y0: {y0}, model output shape: {cnn.model.output_shape}")

#             try:
#                 # Generate adversarial example
#                 x_adv = deepfool_cnn(cnn.model, x0, y0)  # Call the manual DeepFool function
#                 adversarial_examples.append(x_adv)  # Append to the list

#                 # Evaluate robustness
#                 adv_pred = np.argmax(cnn.model(x_adv).numpy(), axis=-1)
#                 if adv_pred != y0:
#                     print(f"Example {j}: Successfully fooled. True Label: {y0}, Adversarial Label: {adv_pred}")
#                 else:
#                     print(f"Example {j}: Failed to fool. True Label: {y0}, Adversarial Label: {adv_pred}")

#             except ValueError as e:
#                 print(f"DeepFool error on example {j}: {e}")
#                 continue

#         # Convert adversarial_examples to numpy array only after the loop
#         adversarial_examples_ok = np.array(adversarial_examples)

#         # Calculate adversarial accuracy
#         adv_correct = 0
#         for adv_x, true_y in zip(adversarial_examples_ok, y_test):
#             adv_pred = np.argmax(cnn.model(adv_x).numpy(), axis=-1)
#             if adv_pred == true_y:
#                 adv_correct += 1

#         robustness_accuracy = adv_correct / len(y_test)
#         robustness_values.append(robustness_accuracy)
#         print(f"Robustness Accuracy: {robustness_accuracy}")

#     return accuracy_values, loss_values, robustness_values



