import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import plotly.graph_objects as go
from utilities import *
from matplotlib.animation import FuncAnimation
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
from matplotlib.animation import FuncAnimation


X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))





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





def artificial_neuron(X_train, y_train, X_test, y_test, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in (range(n_iter)):
        A = model(X_train, W, b)

        if i %10 == 0:
            # Train
            train_loss.append(log_loss(A, y_train))
            y_pred = predict(X_train, W, b)
            train_acc.append(accuracy_score(y_train, y_pred))

            # Test
            A_test = model(X_test, W, b)
            test_loss.append(log_loss(A_test, y_test))
            y_pred = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred))

        # mise a jour
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.legend()
    plt.show()

    return (W, b)
    
"""W, b = artificial_neuron(X, y)"""


X_train, y_train, X_test, y_test = load_data()
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))



print(X_test.shape)
print(y_test.shape)
print(np.unique(y_test, return_counts=True))



"""plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.tight_layout()
plt.show()"""



X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
X_train_reshape.shape

X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_train.max()
X_test_reshape.shape

W, b = initialisation(X_train_reshape)

W, b = artificial_neuron(X_train_reshape, y_train, X_test_reshape, y_test, learning_rate = 0.001, n_iter=10000)

# Charge l'image
image = Image.open('chaton.jpg')

# Redimensionner l'image à la taille attendue (par exemple, 64x64)
image = image.resize((64, 64))

# Convertir l'image en tableau numpy
image_array = np.array(image)
plt.imshow(image_array, cmap='gray')
plt.title("Image normalisée")
plt.axis('off')
plt.show()
# Si l'image est en RGB, convertir en niveaux de gris (si nécessaire)
image_array = image_array.mean(axis=-1)  # Conversion en image en niveaux de gris
plt.imshow(image_array, cmap='gray')
plt.title("Image normalisée")
plt.axis('off')
plt.show()
# Normaliser l'image de la même manière que les données d'entraînement
image_array = image_array / 255.0  # Normalisation [0,1]
plt.imshow(image_array, cmap='gray')
plt.title("Image normalisée")
plt.axis('off')
plt.show()
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




X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
X[:, 1] = X[:, 1] * 1

y = y.reshape(y.shape[0], 1)

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
plt.show()


def artificial_neuron_2(X, y, learning_rate=0.01, n_iter=10000):

  W, b = initialisation(X)
  W[0], W[1] = -7.5, 7.5

  nb = 1
  j=0
  history = np.zeros((n_iter // nb, 5))

  A = model(X, W, b)
  Loss = []
  

  Params1 = [W[0]]
  Params2 = [W[1]]
  Loss.append(log_loss(y, A))
  
  # Training
  for i in range(n_iter):
    A = model(X, W, b)
    Loss.append(log_loss(y, A))
    Params1.append(W[0])
    Params2.append(W[1])
    dW, db = gradients(A, X, y)
    W, b = update(dW, db, W, b, learning_rate = learning_rate)

  plt.plot(Loss)
  plt.show()

  return b, Loss, Params1, Params2



np.save('weights.npy', W)
np.save('bias.npy', b)


