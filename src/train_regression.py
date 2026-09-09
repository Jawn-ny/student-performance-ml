from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from load_data import load_data

df = load_data("data/raw/student-por.csv")

X = df[["age", "studytime", "failures", "absences"]]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit(X_train, y_train)
y_pred_dummy = dummy_model.predict(X_test)
print(y_pred_dummy)
print("Dummy predictions:", y_pred_dummy[:5])

dummy_rmse = root_mean_squared_error(
    y_test,
    y_pred_dummy
)

print("DummyRegressor RMSE:", dummy_rmse)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_rmse = root_mean_squared_error(
    y_test,
    y_pred_linear
)

print("LinearRegression RMSE:", linear_rmse)

print("\nModel comparison:")
print("DummyRegressor:", dummy_rmse)
print("LinearRegression:", linear_rmse)