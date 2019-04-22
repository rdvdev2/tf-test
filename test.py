from __future__ import absolute_import, division, print_function

# TensorFlow i la API keras
import tensorflow as tf
from tensorflow import keras

# Llibreries adicionals
import numpy as np
import matplotlib.pyplot as plt

print("Executant TensorFlow "+tf.__version__)

# Importar les imatges de prova
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Establir el nom de les categories
class_names = ['Samarreta / top', 'Pantaló', 'Pullover', 'Vestit', 'Abric',
                'Sandalia', 'Samarreta', 'Sabatilla', 'Bossa', 'Botí']

# Preprocessar les imatges
# El valor de cada pixel pasa de ser 0-255 a 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Mostrar les primeres 25 imatges amb les seves categories
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Crear el model de la xarxa neuronal
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # Capa que transforma la matriu de pixels en una llista de pixels
    keras.layers.Dense(128, activation=tf.nn.relu), # Capa de parametres calculats mitjançant ML
    keras.layers.Dense(10, activation=tf.nn.softmax) # Capa softmax: Retorna la possibilitat de que la imatge sigui de cada categoria
])

# Compila el model
model.compile(optimizer='adam', # Algoritme que ajusta els parametres
              loss='sparse_categorical_crossentropy', # Algoritme per mesurar la precissió del model
              metrics=['accuracy']) # Monitoratge del model

# Entrena el model
model.fit(train_images, train_labels, epochs=5)

# Comprova la precissió del model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Precissió de la prova: ", test_acc)

# Obtindre totes les prediccions
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Mostrar alguns resultats
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
