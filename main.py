# ML Project - All Models in One Python File
# Author: Batool Mohammed

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load data
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")
X_test = pd.read_csv("X_test.csv")
Y_test = pd.read_csv("Y_test.csv")

# Prepare results folder
os.makedirs("Results", exist_ok=True)

# Label encoder for converting predictions to Yes/No
label_encoder = LabelEncoder()
label_encoder.fit(['No', 'Yes'])

# 1. Decision Tree
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X, Y.values.ravel())
y_pred_dt = model_dt.predict(X_test)
acc_dt = accuracy_score(Y_test, y_pred_dt)
print("Decision Tree Accuracy:", round(acc_dt*100, 2))
pd.DataFrame(label_encoder.inverse_transform(y_pred_dt), columns=["Prediction"]).to_csv("Results/predictions_DecisionTree_model.csv", index=False)

# 2. Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X, Y.values.ravel())
y_pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(Y_test, y_pred_rf)
print("Random Forest Accuracy:", round(acc_rf*100, 2))
pd.DataFrame(label_encoder.inverse_transform(y_pred_rf), columns=["Prediction"]).to_csv("Results/predictions_RF_model.csv", index=False)

# 3. KNN
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X, Y.values.ravel())
y_pred_knn = model_knn.predict(X_test)
acc_knn = accuracy_score(Y_test, y_pred_knn)
print("KNN Accuracy:", round(acc_knn*100, 2))
pd.DataFrame(label_encoder.inverse_transform(y_pred_knn), columns=["Prediction"]).to_csv("Results/predictions_KNN_model.csv", index=False)

# 4. Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X, Y.values.ravel())
y_pred_nb = model_nb.predict(X_test)
acc_nb = accuracy_score(Y_test, y_pred_nb)
print("Naive Bayes Accuracy:", round(acc_nb*100, 2))
pd.DataFrame(label_encoder.inverse_transform(y_pred_nb), columns=["Prediction"]).to_csv("Results/predictions_NaiveBayes_model.csv", index=False)

# 5. SVM
model_svm = SVC(kernel='linear')
model_svm.fit(X, Y.values.ravel())
y_pred_svm = model_svm.predict(X_test)
acc_svm = accuracy_score(Y_test, y_pred_svm)
print("SVM Accuracy:", round(acc_svm*100, 2))
pd.DataFrame(label_encoder.inverse_transform(y_pred_svm), columns=["Prediction"]).to_csv("Results/predictions_SVM_model.csv", index=False)

# 6. ANN
model_ann = Sequential()
model_ann.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model_ann.add(Dense(32, activation='relu'))
model_ann.add(Dense(1, activation='sigmoid'))
model_ann.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model_ann.fit(X, Y, epochs=10, batch_size=32, verbose=1)
y_pred_prob = model_ann.predict(X_test)
y_pred_ann = (y_pred_prob > 0.5).astype(int).reshape(-1)
acc_ann = accuracy_score(Y_test, y_pred_ann)
print("ANN Accuracy:", round(acc_ann*100, 2))
pd.DataFrame(label_encoder.inverse_transform(y_pred_ann), columns=["Prediction"]).to_csv("Results/predictions_ANN_model.csv", index=False)

# 7. Linear Regression
model_lr = LinearRegression()
model_lr.fit(X, Y)
y_pred_continuous = model_lr.predict(X_test)
y_pred_binary = (y_pred_continuous > 0.5).astype(int)
acc_lr = accuracy_score(Y_test, y_pred_binary)
print("Linear Regression Accuracy:", round(acc_lr*100, 2))
pd.DataFrame(label_encoder.inverse_transform(y_pred_binary), columns=["Prediction"]).to_csv("Results/predictions_LinearRegression_model.csv", index=False)

print("\n✅ All models executed. Results saved in Results folder.")
