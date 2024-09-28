#KNN VS NAIVE
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preprocess_data(df, data_severity):
    # Remove spaces between words in columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.replace(" ", "_")
    
    # Encode symptoms using data_severity
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

def compare_algorithms(file_path, severity_path):
    # Load data
    df = pd.read_csv(file_path)
    data_severity = pd.read_csv(severity_path)

    # Preprocess data
    df = preprocess_data(df, data_severity)

    # Split the data into features and target variable
    X = df.drop(columns='Disease', axis=1)
    Y = df['Disease']

    # Standardization
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.2, random_state=42)

    # KNN Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, Y_train)
    knn_predictions = knn_classifier.predict(X_test)

    # Naive Bayes Classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, Y_train)
    nb_predictions = nb_classifier.predict(X_test)

    # Calculate metrics for KNN
    knn_accuracy = accuracy_score(Y_test, knn_predictions)

    # Calculate metrics for Naive Bayes
    nb_accuracy = accuracy_score(Y_test, nb_predictions)

    # Print or return the metrics
    print(f"KNN Accuracy: {knn_accuracy}")
    print(f"Naive Bayes Accuracy: {nb_accuracy}")

    # Return the metrics if needed
    return knn_accuracy, nb_accuracy

# Example usage:
file_path = 'dataset.csv'
severity_path = 'Symptom-severity.csv'

compare_algorithms(file_path, severity_path)
