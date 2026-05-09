import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm


# Load dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Split features and target
X = diabetes_dataset.drop(columns='Outcome')
Y = diabetes_dataset['Outcome']

# Standardize data
scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Train model
classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

# Save model
pickle.dump(classifier, open('saved_model.sav', 'wb'))

# Save scaler
pickle.dump(scaler, open('scaler.sav', 'wb'))

print("Model and scaler saved successfully!")