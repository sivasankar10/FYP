
#KNN VS SVM 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compare_algorithms(df, data_severity):
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

    df_cleaned = remove_space_between_words(df)
    new_df = encode_symptoms(df_cleaned, data_severity)

    # Split the data into features and target variable
    X = new_df.drop(columns='Disease', axis=1)
    Y = new_df['Disease']

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
    
    # SVM Classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, Y_train)
    svm_predictions = svm_classifier.predict(X_test)

    # Calculate metrics for KNN
    knn_accuracy = accuracy_score(Y_test, knn_predictions)
    knn_precision = precision_score(Y_test, knn_predictions, average='macro')
    knn_recall = recall_score(Y_test, knn_predictions, average='macro')
    knn_f1 = f1_score(Y_test, knn_predictions, average='macro')

    # Calculate metrics for SVM
    svm_accuracy = accuracy_score(Y_test, svm_predictions)
    svm_precision = precision_score(Y_test, svm_predictions, average='macro')
    svm_recall = recall_score(Y_test, svm_predictions, average='macro')
    svm_f1 = f1_score(Y_test, svm_predictions, average='macro')

    # Print or return the metrics
    print("KNN Metrics:")
    print(f'Accuracy: {knn_accuracy}')
    print(f'Precision: {knn_precision}')
    print(f'Recall: {knn_recall}')
    print(f'F1-Score: {knn_f1}')

    print("\nSVM Metrics:")
    print(f'Accuracy: {svm_accuracy}')
    print(f'Precision: {svm_precision}')
    print(f'Recall: {svm_recall}')
    print(f'F1-Score: {svm_f1}')

    # Return the metrics if needed
    return knn_accuracy, svm_accuracy


df = pd.read_csv('dataset.csv')
data_severity = pd.read_csv('Symptom-severity.csv')
# Assuming 'df' contains your dataset and 'data_severity' contains symptom severity data
knn_acc, svm_acc = compare_algorithms(df, data_severity)
print(f"KNN Accuracy: {knn_acc}")
print(f"SVM Accuracy: {svm_acc}")
