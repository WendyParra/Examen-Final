# Importar librerías necesarias
import cv2  # Librería para manejo de video e imágenes en tiempo real
import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos
import tensorflow as tf  # Framework para construir y entrenar modelos de aprendizaje profundo
from tensorflow.keras.models import Model, load_model  # Clases para construir y cargar modelos
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Capas para construir redes neuronales
from tensorflow.keras.applications import MobileNetV2  # Modelo preentrenado para reconocimiento de imágenes
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Herramienta para procesar y aumentar datos de imágenes

# *1. Preparar las rutas y el generador de datos*
train_dir = 'C:/Users/wendy/Downloads/python/Estaciones/'  # Carpeta con subcarpetas para cada estación

# Crear un generador de datos que incluye aumentación de imágenes
# (modificaciones para hacer el modelo más robusto)
datagen = ImageDataGenerator(
    rescale=1./255,  # Escala los valores de los píxeles a un rango entre 0 y 1
    rotation_range=30,  # Rotación aleatoria de hasta 30 grados
    width_shift_range=0.2,  # Desplazamiento horizontal aleatorio del 20% de la imagen
    height_shift_range=0.2,  # Desplazamiento vertical aleatorio del 20%
    horizontal_flip=True,  # Invertir imágenes horizontalmente
    validation_split=0.2  # Reservar el 20% de las imágenes para validación
)

# Crear conjuntos de datos de entrenamiento a partir de las imágenes
train_data = datagen.flow_from_directory(
    train_dir,  # Carpeta principal que contiene subcarpetas por estación
    target_size=(128, 128),  # Cambiar el tamaño de las imágenes a 128x128 píxeles
    batch_size=32,  # Número de imágenes procesadas por lote
    class_mode='sparse',  # Usar etiquetas en formato numérico (una clase por imagen)
    subset='training'  # Especificar que este conjunto es para entrenamiento
)

# Crear el conjunto de datos de validación a partir de las imágenes
val_data = datagen.flow_from_directory(
    train_dir,  # Carpeta principal que contiene subcarpetas por estación
    target_size=(128, 128),  # Cambiar el tamaño de las imágenes a 128x128 píxeles
    batch_size=32,  # Número de imágenes procesadas por lote
    class_mode='sparse',  # Usar etiquetas en formato numérico (una clase por imagen)
    subset='validation'  # Especificar que este conjunto es para validación
)

# *2. Crear el modelo*
# Cargar el modelo preentrenado MobileNetV2 sin las capas superiores
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Congelar las capas del modelo base para que no se modifiquen durante el entrenamiento

# Añadir capas personalizadas para clasificar las estaciones
x = base_model.output  # Salida del modelo base
x = GlobalAveragePooling2D()(x)  # Reducir las características a un vector de tamaño fijo
x = Dense(12, activation='relu')(x)  # Capa densa con 12 neuronas y activación ReLU
predictions = Dense(4, activation='softmax')(x)  # Capa de salida con 4 clases y activación softmax
model = Model(inputs=base_model.input, outputs=predictions)  # Crear el modelo final

# Compilar el modelo
model.compile(optimizer='adam',  # Optimizador para ajustar los pesos
              loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación categórica
              metrics=['accuracy'])  # Métrica para evaluar el rendimiento

# *3. Entrenar el modelo*
print("Entrenando el modelo...")
model.fit(train_data, validation_data=val_data, epochs=10)  # Entrenar por 10 épocas

# Guardar el modelo entrenado para uso futuro
model.save('C:/Users/wendy/Downloads/python/estaciones_model.h5')
print("Modelo guardado en 'estaciones_model.h5'")

# *4. Cargar el modelo para predicciones en tiempo real*
model = load_model('C:/Users/wendy/Downloads/python/estaciones_model.h5')  # Cargar el modelo entrenado

# *5. Detección en tiempo real con la cámara*
cap = cv2.VideoCapture(0)  # Activar la cámara para capturar video en tiempo real

while True:
    ret, frame = cap.read()  # Leer un cuadro de video de la cámara
    if not ret:  # Si no hay cuadro, salir del bucle
        break

    # Preprocesar la imagen capturada para que sea compatible con el modelo
    img = cv2.resize(frame, (128, 128)) / 255.0  # Cambiar tamaño y escalar valores de píxeles
    img = np.expand_dims(img, axis=0)  # Añadir una dimensión extra para procesar como lote

    # Realizar predicción
    pred = model.predict(img)  # Obtener predicción del modelo
    estacion = np.argmax(pred)  # Clase con mayor probabilidad
    estacion_nombre = ['Invierno', 'Otono', 'Primavera', 'Verano'][estacion]  # Convertir índice a nombre

    # Mostrar la predicción en la ventana de video
    cv2.putText(frame, f'Estacion: {estacion_nombre}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Detección de Estaciones', frame)  # Mostrar el cuadro con la predicción

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos de la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()