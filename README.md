# Breast Cancer Prediction Using Feedforward Neural Network

## 1. Summary
The aim of this project is to create a highly accurate deep learning model to predict breast cancer (whether the tumour is malignant or benign). The model is trained with Wisconsin Breast Cancer Dataset

## 2. IDE and Framework
This project is created using Sypder as the main IDE. The main frameworks used in this project are Pandas, Scikit-learn and TensorFlow Keras.

## 3. Methodology
### 3.1. Data Pipeline
The data is first loaded and preprocessed, such that unwanted features are removed, and label is encoded in one-hot format. Then the data is split into train-validation-test sets, with a ratio of 60:20:20.

## 3.2. Model Pipeline
A feedforward neural network is constructed that is catered for classification problem. The structure of the model is fairly simple. Figure below shows the structure of the model.

![model](http://github.com/aplatyps/shrdc_ai05_bc/blob/main/img/model.png)

The model is trained with a batch size of 32 and for 100 epochs. Early stopping is applied in this training. The two figures below show the graph of the training process.

![accuracy](http://github.com/aplatyps/shrdc_ai05_bc/blob/main/img/accuracy.png)
![loss](http://github.com/aplatyps/shrdc_ai05_bc/blob/main/img/loss.png)

## 4. Results
Upon evaluating the model with test data, the model obtain the following test results, as shown in figure below.
~~~
4/4 [==============================] - 0s 3ms/step - loss: 0.0636 - accuracy: 0.9912
Test loss = 0.06355356425046921
Test accuracy = 0.9912280440330505
~~~

![confusion](http://github.com/aplatyps/shrdc_ai05_bc/blob/main/img/confusion_matrix.png)
