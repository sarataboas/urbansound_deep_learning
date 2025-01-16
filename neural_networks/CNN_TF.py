import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tensorflow.keras.utils import plot_model

class CNN_Class:
    def __init__(self, input_shape, num_classes):
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None  # Inicializa como None


    def create_model(self,input_shape, num_classes,params):
        """
        Cria um modelo CNN com aumento progressivo de filtros nas camadas convolucionais.

        Args:
            input_shape (tuple): Dimensão da entrada (exemplo: (64, 64, 3)).
            num_classes (int): Número de classes para o problema de classificação.
            params (dict): Hiperparâmetros como número de camadas, filtros, kernel size, etc.

        Returns:
            model: Modelo CNN compilado.
        """
        model = Sequential()
        filters = params['filters']  # Número inicial de filtros

        for _ in range(params['num_conv_layers']):
            # Adicionar camada convolucional com os filtros atuais
            model.add(Conv2D(filters, params['kernel_size'], activation='relu', input_shape=input_shape, padding = 'same'))
            model.add(MaxPooling2D(pool_size=(2, 2),padding = 'same'))
            
            # Dobrar o número de filtros para a próxima camada
            if filters < 128: 
                filters *= 2

        # Camada Flatten
        model.add(Flatten())
        print(f"Shape após Flatten: {model.output_shape}")
        
        # Camada densa com unidades e dropout
        model.add(Dense(params['dense_units'], activation='relu'))
        model.add(Dropout(params['dropout_rate']))
        
        # Camada de saída
        model.add(Dense(num_classes, activation='softmax'))
        
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001))
        # Compilar o modelo
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model


    def initialize_model(self, params):
        self.model = self.create_model(self.input_shape, self.num_classes, params)

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Modelo ainda não foi inicializado. Use `initialize_model` primeiro.")
            
    
    def trainar(self, train_data, train_labels, validation_data, epochs=10, batch_size=32):
        if not self.model:
            raise ValueError("Modelo ainda não foi inicializado. Use `initialize_model` primeiro.")

        
         # Verificar formato dos dados de validação
        if not isinstance(validation_data, tuple) or len(validation_data) != 2:
            raise ValueError("`validation_data` deve ser uma tupla no formato `(X_val, y_val)`.")

        X_val, y_val = validation_data

        # Verificar consistência dos dados
        assert len(X_val) == len(y_val), "Os dados e rótulos de validação devem ter o mesmo comprimento."
            # Configurar callbacks
            
            
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        checkpoint = ModelCheckpoint(
            filepath=f"model_fold_checkpoint.keras",
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        
        print_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch + 1}/{epochs} - {logs}")
        )
        
        
        
        history = self.model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint, print_callback],
            validation_data=validation_data,
            verbose = 0
        )

        return history

    def evaluate(self, test_data, test_labels):
        if not self.model:
            raise ValueError("Model not initialized properly.")
        return self.model.evaluate(test_data, test_labels)

    def predict(self, data):
        if not self.model:
            raise ValueError("Model not initialized properly.")
        return self.model.predict(data)
        

    def plot_graphs(self, history, X_test, y_test, class_labels):
        # Create a 1x3 grid for subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        

        # Plot accuracy values
        axs[0].plot(history.history['accuracy'])
        axs[0].plot(history.history['val_accuracy'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['Train', 'Validation'], loc='upper left')

        # Plot loss values
        axs[1].plot(history.history['loss'])
        axs[1].plot(history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['Train', 'Validation'], loc='upper left')

        # Plot confusion matrix
        y_pred = self.predict(X_test).argmax(axis=1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=axs[2])
        axs[2].set_title('Confusion Matrix')
        axs[2].set_xlabel('Predicted')
        axs[2].set_ylabel('True')

        # Adjust layout
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath, model_name="model.keras"):
        """
        Salva o modelo no formato Keras (.keras).

        Args:
            filepath (str): Diretório onde o modelo será salvo.
            model_name (str): Nome do arquivo para salvar o modelo. (Padrão: "model.keras")
        """
        if not self.model:
            raise ValueError("Modelo ainda não foi inicializado. Treine ou inicialize o modelo antes de salvá-lo.")

        # Adicionar o nome do arquivo ao caminho
        full_path = os.path.join(filepath, model_name)
        
        # Salvar o modelo no formato Keras
        self.model.save(full_path)
        print(f"Modelo salvo em: {full_path}")


        
    
    def load_model(self, filepath):
        """
        Carrega um modelo salvo do caminho especificado.

        Args:
            filepath (str): Caminho do arquivo do modelo salvo.
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Modelo carregado de: {filepath}")

