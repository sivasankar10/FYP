import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('dataset.csv')
data_severity = pd.read_csv('Symptom-severity.csv')
df.head(2)
data_severity.head()
for i in data_severity.index:
    print(data_severity['Symptom'][i], data_severity['weight'][i])

    #convert data_severity to dictionnary
data_dict = data_severity.set_index('Symptom').T.to_dict()
data_dict
df.shape
df.info()
def remove_space_between_words(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.replace(" ", "_")
    return df

df = remove_space_between_words(df)
df.head()
df[df['Disease']=='Acne'].values
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
new_df = encode_symptoms(df, data_severity)
new_df.head()

names = []

# Iterate through columns except for "Disease"
for col in new_df.columns:
    if col != "Disease":
        # Iterate through rows in the column
        for symptom in new_df[col]:
            # Check if the value is a string and not in the 'names' list
            if isinstance(symptom, str) and symptom not in names:
                names.append(symptom)

# Check if all symptoms have been replaced
all_replaced = all(symptom not in names for symptom in data_severity["Symptom"])

if all_replaced:
    print("\nAll symptoms have been replaced.")
else:
    print("\nThe following symptoms were not replaced:", names)

    X = new_df.drop(columns='Disease', axis=1)
    Y = new_df['Disease']
    print(X)
    print(Y)
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    print(standardized_data)

    X = standardized_data
    Y = new_df['Disease']
    print(X)
    print(Y)

    # Earlier code...

# After encoding symptoms and preprocessing
new_df = encode_symptoms(df, data_severity)

# Define X and Y
X = new_df.drop(columns='Disease', axis=1)
Y = new_df['Disease']

# Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

# Split the data into training and testing sets with a random state
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
   
# Create a k-NN classifier and specify the number of neighbors e.g. 3  
k=3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Fit the model on the training data
knn_classifier.fit(X_train, Y_train)

# Predict the labels for the test data
Y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)

precision = precision_score(Y_test, Y_pred, average='macro')
recall = recall_score(Y_test, Y_pred,average='macro')
f1 = f1_score(Y_test, Y_pred, average='macro')

print("KNN")
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(accuracy_score(Y_test, Y_pred,normalize=False))
print("Confusion matrix")
conf_matrix=confusion_matrix(Y_test,Y_pred)
print(conf_matrix)