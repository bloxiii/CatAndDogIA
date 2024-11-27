import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import plotly.graph_objects as go
from utilities import *
from matplotlib.animation import FuncAnimation
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np


W = np.load('weights.npy')
b = np.load('bias.npy')


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A
def log_loss(A, y):
    epsilon = 1e-15  # Petite valeur pour éviter log(0)
    A = np.clip(A, epsilon, 1 - epsilon)  # Limiter A entre epsilon et 1 - epsilon
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)
def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)
def predict(X, W, b):
    A = model(X, W, b)
    # print(A)
    return A >= 0.5





def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X)

    Loss = []

    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = predict(X, W, b)
    print(accuracy_score(y, y_pred))

    plt.plot(Loss)
    plt.show()

    return (W, b)



image = Image.open('chien.jpg')

# Redimensionner l'image à la taille attendue (par exemple, 64x64)
image = image.resize((64, 64))

# Convertir l'image en tableau numpy
image_array = np.array(image)

# Si l'image est en RGB, convertir en niveaux de gris (si nécessaire)
image_array = image_array.mean(axis=-1)  # Conversion en image en niveaux de gris

# Normaliser l'image de la même manière que les données d'entraînement
image_array = image_array / 255.0  # Normalisation [0,1]

# Aplatir l'image pour qu'elle soit compatible avec l'entrée du modèle
image_array = image_array.reshape(1, -1)  # Reshape pour 1 exemple

def predict_image(image_array, W, b):
    A = model(image_array, W, b)  # Passer l'image dans le modèle
    print(A)
    prediction = (A >= 0.5)
    print (prediction)
    return "Chat" if prediction == 0 else "Chien"

# Charger les poids W, b après l'entraînement
result = predict_image(image_array, W, b)
print(f"Prédiction : {result}")