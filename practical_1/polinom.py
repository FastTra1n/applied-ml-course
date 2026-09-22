import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PolynomialRegression(object):
    def __init__(self, degree=2):
        self.degree = degree
        self.coef = None
        self.x_mean = None
        self.x_std = None

    def _normalize(self, x):
        if self.x_mean is None:
            self.x_mean = np.mean(x)
            self.x_std = np.std(x)
        return (x - self.x_mean) / self.x_std

    def _design_matrix(self, x):
        x_norm = self._normalize(x)
        X = np.column_stack([x_norm ** d for d in range(self.degree + 1)])
        return X

    def predict(self, x):
        return self._design_matrix(x) @ self.coef

    def MSE(self, x, Y):
        return ((Y - self.predict(x)) ** 2).mean()

    def fit(self, x, Y, alpha = 0.001, epsylon = 0.01, max_steps = 5000):
        self.coef = np.zeros(self.degree + 1)
        X = self._design_matrix(x)

        steps, errors = [], []
        for step in range(1, max_steps + 1):
            grad = -2 * X.T @ (Y - X @ self.coef)
            self.coef -= alpha * grad
            new_error = self.MSE(x, Y)
            steps.append(step)
            errors.append(new_error)

            if new_error < epsylon:
                break
        return steps, errors


def main():
    gipers = pd.read_csv(r'Гиперспектр кукурузы.csv', sep=';', decimal=',')
    print(gipers.head())

    x = gipers['wavelength'][:50]
    Y = gipers['Spectr'][:50]

    # x_train, x_val, Y_train, Y_val = train_test_split(x, Y, test_size=0.2, random_state=42)
    #
    # # Подбор наилучшей степени полинома
    # degrees = range(1, 11)
    # train_errors = []
    # val_errors = []

    # for d in degrees:
    #     model = PolynomialRegression(degree=d)
    #     alpha = 0.0001 if d <= 3 else 0.00001
    #     model.fit(x_train, Y_train, alpha=alpha, epsylon=1e-5, max_steps=10000)
    #     train_errors.append(model.MSE(x_train, Y_train))
    #     val_errors.append(model.MSE(x_val, Y_val))

    # plt.figure(figsize=(10, 6))
    # plt.plot(degrees, train_errors, 'o-', label='Ошибка на обучении')
    # plt.plot(degrees, val_errors, 's-', label='Ошибка на валидации')
    # plt.xlabel('Степень полинома')
    # plt.ylabel('MSE')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    degree = 7
    regr = PolynomialRegression(degree=degree)
    regr.fit(x, Y, alpha=1e-7, epsylon=1e-6, max_steps=20000)

    x_space = np.linspace(x.min(), x.max(), 500)
    Y_pred = regr.predict(x_space)

    mse = regr.MSE(x, Y)
    print(f"MSE: {mse:.5f}")

    plt.figure(figsize=(12,6))
    plt.plot(x, Y, color='blue', alpha=0.5, label='Исходные данные')
    plt.plot(x_space, Y_pred, color='red', label=f'Полином степени {degree}')
    plt.xlabel('Длина волны, нм')
    plt.ylabel('Спектральная яркость')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
