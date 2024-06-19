from tensorflow.keras.preprocessing.image import ImageDataGenerator 
# Clase modelo de deteccion de emociones en imagenes

class DetectorEmocion:
    # Constructor
    def __init__(self, ws, hs, nc, eps, bs, cn, train_data_dir, val_data_dir):
        self.__width_shape = ws
        self.__height_shape = hs
        self.__num_classes = nc # Cantidad de clasificaciones de las emociones.
        self.__epochs = eps # Cantidad de epocas
        self.__batch_size = bs #Tamaño del lote
        self.__class_names = cn # Lista de las emociones 
        self.__train_datagen = ImageDataGenerator()
        self.__val_datagen = ImageDataGenerator()
        self.__train_generator = self.__train_datagen.flow_from_directory(  
            train_data_dir,
            target_size=(ws, hs),
            batch_size=bs,
            color_mode='grayscale',
            class_mode='categorical',shuffle=True)
        self.__val_generator = self.__val_datagen.flow_from_directory(  
            val_data_dir,
            target_size=(ws, hs),
            batch_size=bs,
            color_mode='grayscale',
            class_mode='categorical',shuffle=True)