# %% Import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
import seaborn as sns

np.random.seed(0)
# %% Téléchargement des samples d'essaie et de test

#x => images
#y => labels 
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(len(x_test), len(y_test))
num_classes = 10

# %% Visualiser les images

f, ax = plt.subplots(1, num_classes, figsize=(20,20))

for i in range(0, num_classes):
    sample = x_train[y_train == i][0]
    ax[i].imshow(sample, cmap='gray')
    ax[i].set_title("Label: {}".format(i), fontsize=16)
    
# plt.show() # Permet de l'afficher en dehors de Jupyter

# %% Visualiser les labels

for i in range(10):
    print(y_train[i])
    
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

for i in range(10):
    print(y_train[i])

# %% Normaliser les données

x_train = x_train/255.0
x_test = x_test/255.0

#Changer la forme des données (au lieux d'avoir 28x28 on a 28x28x1 donc tout sur la même ligne)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(y_test.shape)

# %% Create the model.

model = Sequential()
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0,25))
#units 10 since 10 numbers.
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# %% Train

#nb picture sent.
batch_size = 512    
epochs=30
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

# %% Tester le model

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss:{}, Test Accuracy: {}'.format(test_loss, test_acc))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

#probabilité
print(y_pred)
print(y_pred_classes)

# %% Test sur une image random

random_index = np.random.choice(len(x_test))
x_sample = x_test[random_index]
y_true = np.argmax(y_test, axis=1)
y_sample_true = y_true[random_index]
y_sample_pred_class = y_pred_classes[random_index]

plt.title('Predicted: {}, True: {}'.format(y_sample_pred_class, y_sample_true), fontsize=16)
plt.imshow(x_sample.reshape(28,28), cmap="gray")
# plt.show()

# %% Confusion Matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes)
#Plot
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix')

# %% Wrong prediction

errors = (y_pred_classes - y_true != 0)
y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
x_test_errors = x_test[errors]

#Trouver les erreurs
y_pred_errors_probability = np.max(y_pred_errors, axis=1)
true_probability_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
diff_errors_pred_true = y_pred_errors_probability - true_probability_errors

#Liste des indices
sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)
top_idx_diff_errors = sorted_idx_diff_errors[-5:] #les 5 derniers

# %% aff erreurs

num = len(top_idx_diff_errors)
f, ax = plt.subplots(1, num, figsize=(20,15))

for i in range(0, num):
    idx = top_idx_diff_errors[i]
    sample = x_test_errors[idx].reshape(28,28)
    y_t = y_true_errors[idx]
    y_p = y_pred_classes_errors[idx]
    ax[i].imshow(sample, cmap='gray')
    ax[i].set_title('Predicted label:{}, \nTrue label:{}'.format(y_p, y_t), fontsize=20)
# %%
