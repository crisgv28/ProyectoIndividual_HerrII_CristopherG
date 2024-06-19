from DetectorEmocion import DetectorEmocion


# Definimos las rutas de la data de entrenamiento y validación
ruta_data_train = 'G:/Otros ordenadores/PC Cristopher/Documents/Documentos/U.C.R/Ciencias Actuariales/Herramientas II/ProyectoIndividual_HerrII_CristopherG/data/images/images/train'
ruta_data_val = 'G:/Otros ordenadores/PC Cristopher/Documents/Documentos/U.C.R/Ciencias Actuariales/Herramientas II/ProyectoIndividual_HerrII_CristopherG/data/images/images/validation'

# Definimos otros parámetros importantes para el modelo
ancho = 48 # parametros para redimensionar images del modelo
largo = 48 
num_clases = 7 # cantidad de tipos de clasificacion
epocas = 50 # cantidad de epocas del modelo
tamano_lote = 32 # tamaño del lote en cada entrenamiento
nombres_Declases = ['angry','disgust','fear','happy','neutral','sad','surprise'] # nombre de cada categoria



modelo = DetectorEmocion(ancho, largo, num_clases, epocas, tamano_lote, nombres_Declases, ruta_data_train, ruta_data_val)

# Construir, compilar y entrenar el modelo
modelo = modelo.construirModelo()
modelo.compilar_modelo()
modelo.entrenar_modelo()

# Guardar el modelo
modelo.guardar_modelo("modelFEC.h5")

# Cargar el modelo entrenado
modelo.cargar_modelo_entrenado("modelFEC.h5")

# Realizar una predicción en una imagen
image_path = "/content/images/validation/surprise/10185.jpg"
predicted_class = modelo.detectar(image_path)
print(f'Predicted Emotion: {predicted_class}')

# Evaluar el modelo
modelo.evaluar_modelo()