#KNN VS DECISION TREE 

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('dataset.csv')
data_severity = pd.read_csv('Symptom-severity.csv')

# Preprocessing steps
def remove_space_between_words(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.replace(" ", "_")
    return df

def encode_symptoms(df, data_severity):
    for i in data_severity.index:
        symptom = data_severity["Symptom"][i]
        weight = data_severity["weight"][i]
        df = df.replace(symptom, weight)

    # Replace missing values with 0
    df = df.fillna(0)

    # Additional hardcoded replacements
    df = df.replace("foul_smell_of_urine", 5)
    df = df.replace("dischromic__patches", 6)
    df = df.replace("spotting__urination", 6)
    
    return df

# Apply preprocessing steps
df = remove_space_between_words(df)
new_df = encode_symptoms(df, data_severity)

# Define X (features) and Y (target labels)
X = new_df.drop(columns='Disease', axis=1)
Y = new_df['Disease']

# Standardize the features
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, Y_train)
Y_pred_knn = knn_classifier.predict(X_test)

accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
precision_knn = precision_score(Y_test, Y_pred_knn, average='macro')
recall_knn = recall_score(Y_test, Y_pred_knn, average='macro')
f1_knn = f1_score(Y_test, Y_pred_knn, average='macro')

print("K-Nearest Neighbors")
print(f'Accuracy: {accuracy_knn}')
print(f'Precision: {precision_knn}')
print(f'Recall: {recall_knn}')
print(f'F1-Score: {f1_knn}')
print("Confusion matrix")
conf_matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
print(conf_matrix_knn)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, Y_train)
Y_pred_dt = dt_classifier.predict(X_test)

accuracy_dt = accuracy_score(Y_test, Y_pred_dt)
precision_dt = precision_score(Y_test, Y_pred_dt, average='macro')
recall_dt = recall_score(Y_test, Y_pred_dt, average='macro')
f1_dt = f1_score(Y_test, Y_pred_dt, average='macro')

print("\nDecision Tree")
print(f'Accuracy: {accuracy_dt}')
print(f'Precision: {precision_dt}')
print(f'Recall: {recall_dt}')
print(f'F1-Score: {f1_dt}')
print("Confusion matrix")
conf_matrix_dt = confusion_matrix(Y_test, Y_pred_dt)
print(conf_matrix_dt)
