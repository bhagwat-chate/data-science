import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic numerical data
X_num, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=42)

# Generate synthetic categorical data
np.random.seed(42)
X_cat = np.random.choice(['A', 'B', 'C', 'D', 'E'], size=(1000, 5))

# Generate synthetic temporal data
date_range = pd.date_range(start='1/1/2020', periods=1000, freq='D')
X_temp = np.random.choice(date_range, size=(1000, 3))

# Combine all features into a single DataFrame
df = pd.DataFrame(X_num, columns=[f'num_{i}' for i in range(1, 5)])
for i in range(1, 6):
    df[f'cat_{i}'] = X_cat[:, i-1]
for i in range(1, 4):
    df[f'date_{i}'] = X_temp[:, i-1]
df['target'] = y

# Convert temporal columns to datetime and extract useful features
for i in range(1, 4):
    df[f'date_{i}'] = pd.to_datetime(df[f'date_{i}'])
    df[f'day_{i}'] = df[f'date_{i}'].dt.day
    df[f'month_{i}'] = df[f'date_{i}'].dt.month
    df[f'year_{i}'] = df[f'date_{i}'].dt.year
df = df.drop(columns=[f'date_{i}' for i in range(1, 4)])

# Split the data into training and testing sets
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessor and pipelines
numerical_features = [f'num_{i}' for i in range(1, 5)] + [f'day_{i}' for i in range(1, 4)] + [f'month_{i}' for i in range(1, 4)] + [f'year_{i}' for i in range(1, 4)]
categorical_features = [f'cat_{i}' for i in range(1, 6)]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Define models and hyperparameters
models = [
    ('Logistic Regression', LogisticRegression(max_iter=10000), {
        'classifier__C': [0.01, 0.1, 1, 10, 100]
    }),
    ('Random Forest', RandomForestClassifier(random_state=42), {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30]
    }),
    ('Support Vector Machine', SVC(), {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    }),
    ('K-Nearest Neighbors', KNeighborsClassifier(), {
        'classifier__n_neighbors': [3, 5, 7, 9]
    }),
    ('Decision Tree', DecisionTreeClassifier(random_state=42), {
        'classifier__max_depth': [None, 10, 20, 30]
    }),
    ('Gaussian Naive Bayes', GaussianNB(), {}),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42), {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }),
    ('AdaBoost', AdaBoostClassifier(random_state=42), {
        'classifier__n_estimators': [50, 100, 200]
    }),
    ('Bagging', BaggingClassifier(random_state=42), {
        'classifier__n_estimators': [50, 100, 200]
    }),
    ('Extra Trees', ExtraTreesClassifier(random_state=42), {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30]
    }),
    ('MLP Classifier', MLPClassifier(max_iter=10000, random_state=42), {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'classifier__activation': ['tanh', 'relu']
    }),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis(), {}),
    ('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis(), {}),
    ('Ridge Classifier', RidgeClassifier(), {
        'classifier__alpha': [0.1, 1, 10, 100]
    }),
    ('SGD Classifier', SGDClassifier(random_state=42), {
        'classifier__alpha': [0.0001, 0.001, 0.01, 0.1]
    })
]

# Train and evaluate each model with hyperparameter tuning
for name, model, params in models:
    # Create a pipeline with preprocessor and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = grid_search.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print the results
    print(f"{name} Best Parameters: {grid_search.best_params_}")
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:\n{report}\n")
