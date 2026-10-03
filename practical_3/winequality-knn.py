import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

FILEPATH = "./data/winequality-red.csv"


def main():
    dataset = pd.read_csv(FILEPATH)
    print(f"Размерность датасета: {dataset.shape}")

    X = dataset.iloc[:, :-1]
    y = (dataset.iloc[:, -1] >= 6).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier()
    model.fit(X_train_scaled, y_train)

    predict = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predict)
    print(f"Точность предсказаний модели: {accuracy}")
    print(classification_report(y_test, predict))

    conf_matrix = confusion_matrix(y_test, predict)
    print("Матрица ошибок:")
    print(conf_matrix)

    predict_probs = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, predict_probs)
    print(f'ROC-AUC: {roc_auc:.4f}')
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
    plt.plot([0, 1], [0, 1], color="gray", linestyle='--')
    plt.legend()

    fig, ax = plt.subplots(figsize=(10, 6))  # noqa: RUF059
    conf_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=["Некачественное вино", "Хорошее вино"],
    )
    conf_disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

    ax.set_title("Матрица ошибок", fontsize=16)
    ax.set_xlabel("Предсказанный класс", fontsize=14)
    ax.set_ylabel("Истинный класс", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)

    plt.show()


if __name__ == "__main__":
    main()
