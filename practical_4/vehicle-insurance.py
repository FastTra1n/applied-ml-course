from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

FILEPATH = "data/insclass_train.csv"

# Числовые признаки, использующиеся для обучения и тестов.
FEATURES = [
    "variable_4",
    "variable_6",
    "variable_12",
    "variable_13",
    "variable_14",
    "variable_16",
    "variable_17",
    "variable_18",
    "variable_19",
    "variable_27",
]

# Шаг
ALPHA = 0.01

# Общее число шагов
MAX_STEPS = 10_000


# Бинарная логистическая регрессия
class LogisticRegressionGD:
    def __init__(self):
        self.a = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x @ self.a))

    def predict(self, x):
        return self.sigmoid(x)

    def coefs(self):
        return self.a

    def LogLikelihood(self, x, Y):
        p = np.clip(self.predict(x), 1e-12, 1 - 1e-12)
        return np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    def CrossEntropy(self, x, Y):
        return -self.LogLikelihood(x, Y)

    def accuracy(self, x, Y):
        return ((self.predict(x) > 0.5) == Y).mean()

    def fit(self, x, Y, alpha=0.01, epsylon=0.01, max_steps=10000, Rtype="LL"):
        self.a = np.zeros((x.shape[1], 1))

        steps, errors = [], []
        for step in range(max_steps):
            p = self.predict(x)
            if Rtype == "LL":
                new_error = self.LogLikelihood(x, Y)
                dT_a = x.T @ (Y - p) / x.shape[0]
                self.a += alpha * dT_a
            elif Rtype == "CE":
                new_error = self.CrossEntropy(x, Y)
                dT_a = x.T @ (p - Y) / x.shape[0]
                self.a -= alpha * dT_a
            steps.append(step + 1)
            errors.append(new_error)
        return steps, errors


def prepare_data(part, scaler):
    X = scaler.transform(part[FEATURES])
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    Y = part[["target"]].to_numpy(dtype=float)
    return X, Y


def main():
    df = pd.read_csv(FILEPATH, usecols=FEATURES + ["target"])
    print("Размерность загруженного датасета:", df.shape)
    print("Строки с пропущенными значениями:\n", df.isna().sum())

    df_cleaned = df.dropna()
    print("Размерность датасета после очистки пустых строк:", df_cleaned.shape)

    df_cleaned = df_cleaned.sample(frac=1, random_state=0)

    n = len(df_cleaned)
    train = df_cleaned.iloc[: int(0.70 * n)]  # 70% выборки на обучение
    validate = df_cleaned.iloc[int(0.70 * n) : int(0.85 * n)]  # 15% на валидацию
    test = df_cleaned.iloc[int(0.85 * n) :]  # 15% на тестирование

    scaler = StandardScaler()
    scaler.fit(train[FEATURES])

    X_train, Y_train = prepare_data(train, scaler)
    X_validate, Y_validate = prepare_data(validate, scaler)
    X_test, Y_test = prepare_data(test, scaler)

    y_validate = Y_validate.ravel().astype(int)
    y_test = Y_test.ravel().astype(int)

    models = {}
    histories = {}
    rows = []

    for mode in ("LL", "CE"):
        model = LogisticRegressionGD()
        start = perf_counter()
        steps, errors = model.fit(
            X_train, Y_train, alpha=ALPHA, max_steps=MAX_STEPS, Rtype=mode
        )
        elapsed = perf_counter() - start
        pred_validate = (model.predict(X_validate).ravel() > 0.5).astype(int)
        report = classification_report(
            y_validate, pred_validate, labels=[0, 1], output_dict=True, zero_division=0
        )
        rows.append(
            {
                "Способ": mode,
                "Время обучения": elapsed,
                "Точность": report["accuracy"],
                "F1": report["1"]["f1-score"],
            }
        )
        models[mode] = model
        histories[mode] = (steps, errors)

    p_validate = models['LL'].predict(X_validate).ravel()
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []

    for treshold in thresholds:
        pred = (p_validate > treshold).astype(int)
        score = f1_score(y_validate, pred, zero_division=0)
        scores.append(score)

    best_index = np.argmax(scores)
    best_threshold = thresholds[best_index]
    print("Выбранный порог:", best_threshold)
    print("F1 на валидации:", scores[best_index])

    results = pd.DataFrame(rows)
    print("\nСравнение на валидации:")
    print(results.to_string(index=False))

    print(
        "\nКоэффициенты LL и CE совпадают:",
        np.allclose(models["LL"].coefs(), models["CE"].coefs()),
    )

    for mode, model in models.items():
        pred_test = (model.predict(X_test).ravel() > best_threshold).astype(int)

        print(f"\nСпособ {mode}: отчёт на тестовой части")

        print(
            classification_report(
                y_test, pred_test, labels=[0, 1], digits=4, zero_division=0
            )
        )

        matrix = confusion_matrix(y_test, pred_test, labels=[0, 1])

        matrix_df = pd.DataFrame(
            matrix, index=["Факт 0", "Факт 1"], columns=["Прогноз 0", "Прогноз 1"]
        )

        print("Матрица ошибок:")
        print(matrix_df)

    steps_LL, errors_LL = histories["LL"]
    steps_CE, errors_CE = histories["CE"]

    plt.plot(steps_CE, errors_CE, label="CE")
    plt.plot(steps_LL, -np.array(errors_LL), "--", label="-LL")

    plt.xlabel("Шаг")
    plt.ylabel("Средняя ошибка")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
