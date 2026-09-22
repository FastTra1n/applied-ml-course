import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SimpleRegression(object):
    def __init__(self):
        self.a0 = 0
        self.a1 = 0

    def predict(self, x):
        return self.a0 + self.a1 * x

    def MSE(self, x, Y):
        return ((Y - self.predict(x)) ** 2).mean()

    def MAE(self, x, Y):
        return abs(Y - self.predict(x)).mean()

    def MAPE(self, x, Y):
        return abs((Y - self.predict(x)) / Y).mean()

    def fit_analytic(self, x, Y):
        self.a1 = ((x - x.mean()) * (Y - Y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        self.a0 = Y.mean() - self.a1 * x.mean()
        return self.a0, self.a1

    def fit_gradient(self, x, Y, alpha = 0.001, epsylon = 0.01, max_steps = 5000):
        steps, errors = [], []

        for step in range(1, max_steps + 1):
            dT_a0 = -2 * (Y - self.predict(x)).mean()
            dT_a1 = -2 * ((Y - self.predict(x)) * x).mean()

            self.a0 -= alpha * dT_a0
            self.a1 -= alpha * dT_a1

            new_error = self.MSE(x, Y)
            steps.append(step)
            errors.append(new_error)

            if new_error < epsylon:
                break
        return steps, errors


def main():
    gipers = pd.read_csv(r'Гиперспектр кукурузы.csv', sep=';', decimal=',')
    print(gipers.head())

    x = gipers['wavelength']
    Y = gipers['Spectr']

    regr_analytic = SimpleRegression()
    regr_analytic.fit_analytic(x, Y)
    print(f'Аналитический метод: a0 = {regr_analytic.a0:.6f}, a1 = {regr_analytic.a1:.6f}, MSE = {regr_analytic.MSE(x, Y):.4f}')

    regr_gradient = SimpleRegression()
    regr_gradient.fit_gradient(x, Y, alpha=1e-7, epsylon=1e-6, max_steps=20000)
    print(f'Градиентный спуск: a0 = {regr_gradient.a0:.6f}, a1 = {regr_gradient.a1:.6f}, MSE = {regr_gradient.MSE(x, Y):.4f}')

    x_space = np.linspace(x.min(), x.max(), 1000)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_space,
        regr_analytic.predict(x_space),
        color="red",
        linewidth=2.5,
        label="Аналитическая регрессия"
    )
    plt.plot(
        x_space,
        regr_gradient.predict(x_space),
        color="green",
        linewidth=2.5,
        label="Градиентный спуск"
    )
    plt.xlabel("Длина волны (wavelength)")
    plt.ylabel("Спектр (Spectr)")
    plt.plot(x, Y, color='blue', alpha=0.5, label='Гиперспектр кукурузы')

    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
