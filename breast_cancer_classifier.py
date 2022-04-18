# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:00:31 2022

"""

import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


file_path = r"B:\MSI\Downloads\shrdc\breast_cancer\dataset\data.csv"
cancer_data = pd.read_csv(file_path)

cancer_data = cancer_data.drop(['id', "Unnamed: 32"], axis=1)

cancer_features = cancer_data.copy()
cancer_label = cancer_features.pop('diagnosis')
cancer_label_OH = pd.get_dummies(cancer_label)

SEED = randint(100, 15000)
x_train, x_iter, y_train, y_iter = train_test_split(cancer_features, cancer_label_OH, test_size=0.4, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter, y_iter, test_size=0.5, random_state=SEED)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

number_input = x_train.shape[-1]
number_output = y_train.shape[-1]

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=number_input)) 
model.add(tf.keras.layers.Dense(48, activation='elu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(16, activation='elu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(4, activation='elu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(number_output, activation="softmax"))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tf.keras.utils.plot_model(model,
                          to_file='img\model.png',
                          show_shapes=True,
                          show_layer_activations=True)

base_log_path = r"B:\MSI\Downloads\shrdc\tensorboard_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2)
EPOCHS = 100
BATCH_SIZE = 32
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[tb_callback, es_callback]
                    workers=4,
                    use_multiprocessing=True)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('img\accuracy.png')
plt.show()
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('img\loss.png')
plt.show()
plt.clf()


test_result = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test accuracy = {test_result[1]}\n\n")

y_pred = model.predict(x_test,
                       batch_size=BATCH_SIZE,
                       callbacks=[tb_callback, es_callback],
                       workers=4,
                       use_multiprocessing=True)
y_pred = np.argmax(y_pred, axis=-1)

cm = tf.math.confusion_matrix(y_test['M'].to_numpy(),
                              y_pred,
                              num_classes=number_output,
                              dtype=tf.dtypes.int32)
ax = sns.heatmap(cm, xticklabels=y_test.columns, yticklabels=y_test.columns, annot=True, fmt='d', cmap='Blues')

ax.set_title('Confusion matrix for Winscosin breast cancer')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
fig = ax.get_figure()
fig.savefig("confusion_matrix.png") 