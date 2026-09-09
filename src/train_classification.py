from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from load_data import load_data


# 1. Load data
df = load_data("data/raw/student-por.csv")


# 2. Create classification target
df["passed"] = (df["G3"] >= 10).astype(int)


# 3. Define features

numeric_features = [
    "age",
    "studytime",
    "failures",
    "absences"
]

categorical_features = [
    "school",
    "sex",
    "address"
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df["passed"]


# 4. Train / Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Classification Baseline

dummy_model = DummyClassifier(strategy="most_frequent")

dummy_model.fit(X_train, y_train)

y_pred_dummy = dummy_model.predict(X_test)

dummy_accuracy = accuracy_score(
    y_test,
    y_pred_dummy
)


# 6. Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# 7. Logistic Regression Pipeline

classification_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression())
    ]
)


# 8. Train model

classification_pipeline.fit(
    X_train,
    y_train
)


# 9. Predict

y_pred_logistic = classification_pipeline.predict(
    X_test
)


# 10. Evaluate

logistic_accuracy = accuracy_score(
    y_test,
    y_pred_logistic
)

cm = confusion_matrix(
    y_test,
    y_pred_logistic
)


# 11. Print results

print("DummyClassifier Accuracy:", dummy_accuracy)
print("LogisticRegression Accuracy:", logistic_accuracy)

print("\nConfusion Matrix:")
print(cm)