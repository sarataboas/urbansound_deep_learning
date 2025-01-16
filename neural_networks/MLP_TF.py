import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.metrics import confusion_matrix, classification_report
#from tensorflow.keras.optimizers import Adam 
import numpy 
from tensorflow.keras.utils import plot_model


class MLPClassifier:
    def __init__(self, input_shape, num_classes=10, params=None):
        self.model = None
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.params = params  # Save the params for later use

        self.build_model
        print("MLP Model initialized successfully with input shape:", input_shape)
    
    def build_model(self, hidden_layers, learning_rate, dropout_rate, optimizer, l2_lambda):
        hidden_layers = hidden_layers
        learning_rate = learning_rate
        dropout_rate = dropout_rate
        optimizer = optimizer
        l2_lambda = l2_lambda

        self.model = Sequential()
        
        # Primeira camada (entrada)
        self.model.add(Dense(hidden_layers[0], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)))
        self.model.add(Dropout(dropout_rate))
        
        # Camadas ocultas
        for units in hidden_layers[1:]:
            self.model.add(Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)))
            self.model.add(Dropout(dropout_rate))
        
        # Camada de saída
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compilação do modelo
        self.model.compile(optimizer=optimizer(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print("Modelo MLP inicializado com sucesso.")

    

    def train(self, X_train, y_train, X_val, y_val,  epochs, batch_size, patience=4):
        epochs = epochs
        batch_size = batch_size
        print_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch + 1}/{epochs} - {logs}") 
        if (epoch + 1) % 10 == 0 else None
        )
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                print_callback
        
            ],
            verbose=0
            
        )
        print("Training completed.")
    
    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")
        return loss, accuracy
    
    def predict(self, X):
        print("Predicting: ", X.shape)
        return tf.argmax(self.model.predict(X), axis=1)
    

    def plot_graphs(self, X_test, y_test, class_labels):
        # Create a 1x3 grid for subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))

        # Plot accuracy values
        axs[0].plot(self.history.history['accuracy'])
        axs[0].plot(self.history.history['val_accuracy'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['Train', 'Validation'], loc='upper left')

        # Plot loss values
        axs[1].plot(self.history.history['loss'])
        axs[1].plot(self.history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['Train', 'Validation'], loc='upper left')

        # Plot confusion matrix
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=axs[2])
        axs[2].set_title('Confusion Matrix')
        axs[2].set_xlabel('Predicted')
        axs[2].set_ylabel('True')

        # Adjust layout
        plt.tight_layout()
        plt.show()


    def copy(self):
        """Create a deep copy of the MLPClassifier instance."""
        copied_instance = copy.deepcopy(self)
        print("Deep copy of the MLPClassifier instance created successfully.")
        return copied_instance
    
