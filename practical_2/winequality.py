import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    dataset = pd.read_csv("./data/winequality-red.csv")
    print(f"Размерность датасета: {dataset.shape}")

    X = dataset.iloc[:, :-1]
    y = (dataset.iloc[:, -1] >= 6).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=20_000)
    model.fit(X_train_scaled, y_train)

    predict = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predict)
    conf_matrix = confusion_matrix(y_test, predict)
    print(f"Точность предсказаний модели: {accuracy}")
    print("Матрица ошибок:")
    print(conf_matrix)
    print(classification_report(y_test, predict))

    weights = model.coef_
    bias = model.intercept_
    print("Коэффициенты гиперпараметров w:", weights)
    print("Коэффициент смещения b:", bias)

    test_wine = [
        7.2,
        0.35,
        0.42,
        2.1,
        0.065,
        18.0,
        65.0,
        0.9958,
        3.30,
        0.72,
        11.8
    ]

    wine = pd.DataFrame([test_wine], columns=X.columns)
    wine_scaled = scaler.transform(wine)
    prediction = model.predict(wine_scaled)
    print('Вино хорошее' if prediction else 'Вино не качественное')


if __name__ == "__main__":
    main()
