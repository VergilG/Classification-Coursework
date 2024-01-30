import time
import pandas as pd  # data visualized
import numpy as np  # calculating
import matplotlib.pyplot as plt  # generating plots
from sklearn import preprocessing  # preprocessing data
from sklearn.model_selection import train_test_split  # splitting data
from sklearn.neighbors import KNeighborsClassifier  # k nearest neighbors method
from sklearn.svm import SVC  # support vector machine method
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # discriminant analysis method
from sklearn import metrics  # Classification performance analysis

# Load the original data
videoRaw = pd.read_csv('WLA.csv', header=None)
labelsRaw = pd.read_csv('Labels.csv', header=None)

# Cut down the first 16 rows
videoData = videoRaw.drop(range(16))
videoData = videoData.reset_index(drop=True)

labelsData = labelsRaw.drop(range(16))
labelsData = labelsData.reset_index(drop=True)

# Determine if area should be dropped by calculating coefficient
coeffVideo = np.corrcoef(videoData.T)

# Normalize
normVideo = preprocessing.MinMaxScaler().fit_transform(videoData)
motorIndex = [x for x in range(normVideo.shape[0]) if x % 2 == 0]
carIndex = [x for x in range(normVideo.shape[0]) if x % 2 == 1]
motorNorm = normVideo[motorIndex, :2]
carNorm = normVideo[carIndex, :2]

# Add labels
motorNorm = np.append(motorNorm, np.full((motorNorm.shape[0], 1), 0), axis=1)
carNorm = np.append(carNorm, np.full((carNorm.shape[0], 1), 1), axis=1)

# Combining the two datasets
combinedData = np.zeros((motorNorm.shape[0] + carNorm.shape[0], 3))
for i in range(motorNorm.shape[0]):
    for j in range(motorNorm.shape[1]):
        combinedData[i * 2, j] = motorNorm[i, j]
        combinedData[i * 2 + 1, j] = carNorm[i, j]

# Divide the training data and test data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(combinedData[:, :2], combinedData[:, 2], test_size=0.2)

# Plot the dataset
plt.Figure(figsize=(6, 4))
plt.scatter(motorNorm[:, 0], motorNorm[:, 1], marker='o')
plt.scatter(carNorm[:, 0], carNorm[:, 1], marker='o')
plt.xlabel('Width')
plt.ylabel('Length')
plt.savefig('motor_car.pdf')

plt.show()

# Classification

# LDA method
train_time_LDA = []
predict_time_LDA = []

for _ in range(1000):
    # Fit the model
    start_time = time.time()
    lda = LDA().fit(x_train, y_train)
    training_time = time.time() - start_time
    train_time_LDA.append(training_time)

    # Predict the test data
    start_time = time.time()
    y_predict_LDA = lda.predict(x_test)
    predict_time = time.time() - start_time
    predict_time_LDA.append(predict_time)

print("LDA training time: {:.4f} ms.".format(np.mean(train_time_LDA) * 1000))
print("LDA prediction time: {:.4f} ms.".format(np.mean(predict_time_LDA) * 1000))

# Calculate the accuracy and F1 score
accu_LDA = metrics.accuracy_score(y_test, y_predict_LDA)
precision_LDA = metrics.precision_score(y_test, y_predict_LDA, average='binary')
recall_LDA = metrics.recall_score(y_test, y_predict_LDA, average='binary')
F1_LDA = 2 * precision_LDA * recall_LDA / (precision_LDA + recall_LDA)
print('The accuracy of LDA is: {:.2f}'.format(accu_LDA))
print('The F1 score of LDA is: {:.2f}'.format(F1_LDA))

# Transform the data
lda_data = lda.transform(x_train)

# Visualize the data after LDA fitting
plt.Figure(figsize=(6, 4))

plt.style.use('ggplot')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_predict_LDA)
plt.xlabel('Width')
plt.ylabel('Length')

plt.show()

# KNN method
k = 10

train_time_kNN = []
predict_time_kNN = []

for _ in range(1000):
    start_time = time.time()
    kNN = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    training_time = time.time() - start_time
    train_time_kNN.append(training_time)

    start_time = time.time()
    y_predict_kNN = kNN.predict(x_test)
    predict_time = time.time() - start_time
    predict_time_kNN.append(predict_time)

print("kNN training time: {:.4f} ms.".format(np.mean(train_time_kNN) * 1000))
print("kNN prediction time: {:.4f} ms.".format(np.mean(predict_time_kNN) * 1000))

accu_kNN = metrics.accuracy_score(y_test, y_predict_kNN)
precision_kNN = metrics.precision_score(y_test, y_predict_kNN, average='binary')
recall_kNN = metrics.recall_score(y_test, y_predict_kNN, average='binary')
F1_kNN = 2 * precision_kNN * recall_kNN / (precision_kNN + recall_kNN)
print('The accuracy of kNN is: {:.2f}'.format(accu_kNN))
print('The F1 score of kNN is: {:.2f}'.format(F1_kNN))

# Visualize the data
coef_LDA = lda.coef_[0]
intercept = lda.intercept_[0]

slope = -coef_LDA[0] / coef_LDA[1]
intercept = -intercept / coef_LDA[1]

plt.scatter(combinedData[:, 0], combinedData[:, 1], c=combinedData[:, 2])

x = np.arange(combinedData[:, 0].min(), combinedData[:, 0].max(), 0.01)
y = slope * x + intercept
plt.plot(x, y, color='red')

plt.savefig('LDA.pdf')
plt.show()

# SVM method
train_time_SVM = []
predict_time_SVM = []

for _ in range(1000):
    start_time = time.time()
    svm_train = SVC(kernel='linear').fit(x_train, y_train)
    training_time = time.time() - start_time
    train_time_SVM.append(training_time)

    start_time = time.time()
    y_predict_SVM = svm_train.predict(x_test)
    predict_time = time.time() - start_time
    predict_time_SVM.append(predict_time)

print("SVM training time: {:.4f} ms.".format(np.mean(train_time_SVM) * 1000))
print("SVM prediction time: {:.4f} ms.".format(np.mean(predict_time_SVM) * 1000))

accu_SVM = metrics.accuracy_score(y_test, y_predict_SVM)
precision_SVM = metrics.precision_score(y_test, y_predict_SVM, average='binary')
recall_SVM = metrics.recall_score(y_test, y_predict_SVM, average='binary')
F1_SVM = 2 * precision_SVM * recall_SVM / (precision_SVM + recall_SVM)
print('The accuracy of SVM:', accu_SVM)
print('The F1-score of SVM:', F1_SVM)
