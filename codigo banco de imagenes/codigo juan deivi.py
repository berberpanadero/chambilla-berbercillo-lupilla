
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

# 1. Configuración
ruta_dataset = "dataset"

tamano_imagen = (128, 128)
lote = 32

# 2. Cargar banco de imágenes
entrenamiento = tf.keras.utils.image_dataset_from_directory(
    ruta_dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=tamano_imagen,
    batch_size=lote
)

validacion = tf.keras.utils.image_dataset_from_directory(
    ruta_dataset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=tamano_imagen,
    batch_size=lote
)

# Obtener nombres de los animales
clases = entrenamiento.class_names

print("Animales detectados:", clases)

# 3. Crear modelo CNN
modelo = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(clases), activation="softmax")
])

# 4. Compilar modelo
modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Entrenar modelo
modelo.fit(
    entrenamiento,
    validation_data=validacion,
    epochs=10
)

# 6. Guardar modelo entrenado
modelo.save("modelo_animales.keras")

print("Modelo entrenado correctamente.")

# 7. Detectar animal en una nueva imagen
def detectar_animal(ruta_imagen):

    img = image.load_img(
        ruta_imagen,
        target_size=tamano_imagen
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediccion = modelo.predict(img_array)

    indice = np.argmax(prediccion[0])

    animal = clases[indice]
    confianza = prediccion[0][indice] * 100

    print("Animal detectado:", animal)
    print("Confianza:", round(confianza, 2), "%")

    plt.imshow(img)
    plt.title(f"{animal} - {confianza:.2f}%")
    plt.axis("off")
    plt.show()


# 8. Probar detector
detectar_animal("prueba.jpg")
