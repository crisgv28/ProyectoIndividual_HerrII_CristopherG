from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input, AveragePooling2D,Activation
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import TensorBoard
import datetime, os

from tensorflow.keras.models import load_model
import cv2 
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.models import load_model
import numpy as np



# Clase modelo de deteccion de emociones en imagenes
class DetectorEmocion:
    # Constructor: inicializa parametros para el modelo.
    def __init__(self, ws, hs, nc, eps, bs, cn, train_data_dir, val_data_dir):
        self.width_shape = ws
        self.height_shape = hs
        self.num_classes = nc # Cantidad de emociones por clasificar.
        self.epochs = eps # Cantidad de epocas
        self.batch_size = bs #Tamaño del lote
        self.class_names = cn # Lista de las emociones 
        self.train_datagen = ImageDataGenerator() # generador de datos tipo imagen
        self.val_datagen = ImageDataGenerator()
        self.__train_datagen = ImageDataGenerator()
        self.__val_datagen = ImageDataGenerator()
        self.train_generator = self.__train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(ws, hs),
            batch_size=bs,
            color_mode='grayscale',
            class_mode='categorical', shuffle=True)
        
        self.val_generator = self.__val_datagen.flow_from_directory(
            val_data_dir,
            target_size=(ws, hs),
            batch_size=bs,
            color_mode='grayscale',
            class_mode='categorical', shuffle=True)
        # Construir el modelo al crear la instancia de la clase
        self.model = self.construirModelo()
    
    # Construye el modelo
    def construirModelo(self):
        model = Sequential() # clase para crear un modelo en los que las capas se colocan una tras otra
        # Extracción de Características
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(self.width_shape, self.height_shape, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # Clasificación
        model.add(Flatten())
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        return model
    
    # Compilar el modelo, se establecen los componentes esenciales necesarios para el entrenamiento, como la función de pérdida, el optimizador y las métricas de evaluación.
    def compilar_modelo(self):
        #opt = Adam(learning_rate=1e-4, decay=1e-4 / self.epochs)
        opt = Adam(learning_rate=1e-4)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Entrena el modelo con los generadores de datos de entrenamiento y validación
    def entrenar_modelo(self):
        # Estas líneas configuran y añaden un callback de TensorBoard al modelo para que durante el entrenamiento se generen registros de seguimiento en el directorio especificado. 
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(logdir, histogram_freq=1)
        
        # Entrenar modelo
        self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            steps_per_epoch=self.train_generator.n // self.batch_size,
            validation_steps=self.val_generator.n // self.batch_size,
            callbacks=[tensorboard_callback])
    
    # Guarda el modelo entrenado en el archivo especificado por filepath
    def guardar_modelo(self, filepath):
        self.model.save(filepath)
        
    # Carga un modelo entrenado desde el archivo especificado por filepath
    def cargar_modelo_entrenado(self, filepath):
        self.model = load_model(filepath)
        
    # Realiza una predicción utilizando el modelo entrenado.
    def detectar(self, image_path):
        face = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (self.width_shape, self.height_shape))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
    
        preds = self.model.predict(face)
        predicted_class = self.class_names[np.argmax(preds)]
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        return predicted_class
    
    # muestra resultados de la data de validación
    def evaluar_modelo(self):
        val_generator = self.__val_datagen.flow_from_directory(
            self.val_data_dir,
            target_size=(self.width_shape, self.height_shape),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode='categorical', shuffle=False)
        
        predictions = self.model.predict(val_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_real = val_generator.classes
    
        matc = confusion_matrix(y_real, y_pred)
        plot_confusion_matrix(conf_mat=matc, figsize=(5, 5), show_normed=False)
        plt.tight_layout()
        plt.show()
    
        print(classification_report(y_real, y_pred, target_names=self.class_names, digits=4))
    
    
        