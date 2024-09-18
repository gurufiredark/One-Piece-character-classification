import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Diretório principal onde estão as subpastas das classes
data_dir = 'Data'

# Caminho para o arquivo txt com os nomes das classes
classes_txt = 'classnames.txt'

# Le o arquivo de texto para obter a ordem das classes
with open(classes_txt, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Função para carregar as imagens e seus rótulos
def load_images_and_labels(directory, classes):
    images = []
    labels = []
    for idx, classe in enumerate(classes):
        classe_dir = os.path.join(directory, classe)
        for filename in os.listdir(classe_dir):
            img = cv2.imread(os.path.join(classe_dir, filename))
            img = cv2.resize(img, (100, 100))  # Redimensionar para 100x100 pixels
            images.append(img)
            labels.append(idx)  # Usar o índice como rótulo da classe
    return images, labels

# Carrega imagens e rótulos
images, labels = load_images_and_labels(data_dir, classes)

# Converte listas em arrays numpy para um tratamento melhor
images = np.array(images)
labels = np.array(labels)

# Normalizandoo os valores dos pixels para o intervalo [0, 1]
images = images / 255.0

# Separando os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definição do modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(classes), activation='softmax')  # Usando softmax para classificação multiclasse
])

# Compilando o modelo
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',  # Usando sparse_categorical_crossentropy para rótulos categóricos
            metrics=['accuracy'])

r = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Avaliando o modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Acurácia do modelo: {test_acc}')

# Plotando a perda
plt.plot(r.history['loss'], label='perda no treinamento')
plt.plot(r.history['val_loss'], label='perda nos testes')
plt.title('Perda durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotando a precisão
plt.plot(r.history['accuracy'], label='precisão no treinamento')
plt.plot(r.history['val_accuracy'], label='precisão nos testes')
plt.title('Precisão durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Precisão')
plt.legend()
plt.show()