import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

FILEPATH = "data/insclass_train.csv"

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


class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * torch.randn(n_inputs, n_neurons)
        self.biases = torch.zeros(n_neurons)

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, dvalues):
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = dvalues.sum(dim=0)
        self.dinputs = dvalues @ self.weights.T


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return inputs.clip(min=0)

    def backward(self, dvalues):
        self.dinputs = dvalues.clone()
        self.dinputs[self.inputs <= 0] = 0


# class MSELoss:
#     def forward(self, y_pred, y_true):
#         return ((y_pred - y_true) ** 2).mean()

#     def backward(self, y_pred, y_true):
#         self.dinputs = 2 * (y_pred - y_true) / y_pred.numel()


class BinaryCrossEntropyWithLogits:
    def forward(self, logits, y_true):
        return (
            torch.logaddexp(torch.zeros_like(logits), logits) - y_true * logits
        ).mean()

    def backward(self, logits, y_true):
        probabilities = sigmoid(logits)

        self.dinputs = (probabilities - y_true) / y_true.numel()


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def predict(inputs, fc1, relu, fc2):
    # Первый слой
    out = fc1.forward(inputs)

    # Нелинейная активация
    out = relu.forward(out)

    # Предсказание
    return fc2.forward(out)


def prepare_data(part, scaler):
    X = torch.tensor(scaler.transform(part[FEATURES]), dtype=torch.float32)

    Y = torch.tensor(part[["target"]].to_numpy(), dtype=torch.float32)

    return X, Y


def main():
    torch.manual_seed(0)

    df = pd.read_csv(FILEPATH, usecols=FEATURES + ["target"])
    df = df.dropna()
    df = df.sample(frac=1, random_state=0)

    n = len(df)

    train = df.iloc[: int(0.70 * n)]
    validate = df.iloc[int(0.70 * n) : int(0.85 * n)]
    test = df.iloc[int(0.85 * n) :]

    scaler = StandardScaler()
    scaler.fit(train[FEATURES])

    X_train, y_train = prepare_data(train, scaler)
    X_validate, y_validate = prepare_data(validate, scaler)
    X_test, y_test = prepare_data(test, scaler)

    fc1 = Dense(len(FEATURES), 10)
    relu = ReLU()
    fc2 = Dense(10, 1)
    loss = BinaryCrossEntropyWithLogits()

    lr = 0.02  # Скорость обучения
    epochs = 2000  # Количество полных проходов
    train_history = []
    validate_history = []
    for epoch in range(epochs):
        pred = predict(X_train, fc1, relu, fc2)

        # Backpropagation
        loss.backward(pred, y_train)
        fc2.backward(loss.dinputs)
        relu.backward(fc2.dinputs)
        fc1.backward(relu.dinputs)

        # Корректировка весов первого слоя
        fc1.weights -= lr * fc1.dweights
        fc1.biases -= lr * fc1.dbiases

        # Корректировка весов второго слоя
        fc2.weights -= lr * fc2.dweights
        fc2.biases -= lr * fc2.dbiases

        train_loss = loss.forward(predict(X_train, fc1, relu, fc2), y_train).item()

        validate_loss = loss.forward(
            predict(X_validate, fc1, relu, fc2), y_validate
        ).item()

        train_history.append(train_loss)
        validate_history.append(validate_loss)

        if (epoch + 1) % 200 == 0:
            print(
                f"Эпоха {epoch + 1}:"
                f"\t- train BCE = {train_loss:.5f}"
                f"\t- validation BSE = {validate_loss:.5f}"
            )

    validate_logits = predict(X_validate, fc1, relu, fc2)
    p_validate = sigmoid(validate_logits).numpy().ravel()
    true_validate = y_validate.numpy().ravel().astype(int)

    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    for threshold in thresholds:
        pred_validate = (p_validate > threshold).astype(int)

        score = f1_score(true_validate, pred_validate, zero_division=0)

        scores.append(score)

    best_index = np.argmax(scores)

    best_threshold = float(thresholds[best_index])

    print(f"\nВыбранный порог: {best_threshold:.2f}")
    print(f"F1 на валидации: {scores[best_index]:.4f}")

    test_pred = predict(X_test, fc1, relu, fc2)
    test_bce = loss.forward(test_pred, y_test).item()
    probabilities = sigmoid(test_pred).numpy().ravel()

    threshold = 0.5
    predictions = (probabilities > best_threshold).astype(int)
    true_labels = y_test.numpy().ravel().astype(int)
    print("Test BCE:", test_bce)

    print(
        classification_report(
            true_labels, predictions, labels=[0, 1], digits=4, zero_division=0
        )
    )

    matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])

    print("Матрица ошибок:")
    print(
        pd.DataFrame(
            matrix, index=["Факт 0", "Факт 1"], columns=["Прогноз 0", "Прогноз 1"]
        )
    )

    print("Accuracy при прогнозе всем 0:", (true_labels == 0).mean())

    # График обучения
    plt.figure(1)
    plt.plot(train_history, label="Train")
    plt.plot(validate_history, label="Validation")

    plt.xlabel("Эпоха")
    plt.ylabel("Бинарная кросс-энтропия")
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    main()
