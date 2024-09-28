import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset = pd.read_csv("dataset.csv")

# Assuming 'risk level ' is the target column
target_column = 'risk level '

# If the target column is present, proceed with the analysis
if target_column in dataset.columns:
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify non-numeric features
    non_numeric_features = X.select_dtypes(include=['object']).columns

    # Create a column transformer with one-hot encoding for non-numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ('non_numeric', OneHotEncoder(handle_unknown='ignore'), non_numeric_features)
        ],
        remainder='passthrough'
    )

    # Create a Gradient Boosting classifier
    classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Create a pipeline with the preprocessor and classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

else:
    print(f"Error: The target column '{target_column}' not found in the dataset.")