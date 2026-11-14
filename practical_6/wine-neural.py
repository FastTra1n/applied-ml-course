import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Perceptron(torch.nn.Module):
    def __init__(
        self,
        input_dim=11,
        num_layers=1,
        hidden_dim=32,
        output_dim=2,
        p=0.0,
        device="cpu",
    ):
        super().__init__()
        self.layers = torch.nn.Sequential()
        prev_size = input_dim
        for i in range(num_layers):
            self.layers.add_module(
                f"layer{i}", torch.nn.Linear(prev_size, hidden_dim)
            )
            self.layers.add_module(f"relu{i}", torch.nn.ReLU())
            self.layers.add_module(f"dropout{i}", torch.nn.Dropout(p=p))
            prev_size = hidden_dim
        self.layers.add_module("classifier", torch.nn.Linear(prev_size, output_dim))
        self.to(device)

    def forward(self, input):
        return self.layers(input)


def trainer(model, dataset, loss_function, optimizer, epochs, device="cpu"):
    for epoch in tqdm(range(epochs), leave=False):
        generator = DataLoader(dataset, batch_size=64, shuffle=True)

        for x, y in tqdm(generator, leave=False):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()


def testing(model, dataset, device):
    generator = DataLoader(dataset, batch_size=64)
    pred = []
    real = []
    with torch.no_grad():
        for x, y in tqdm(generator, leave=False):
            x = x.to(device)
            y = y.to(device)
            pred.extend(torch.argmax(model(x), dim=-1).cpu().numpy().tolist())
            real.extend(y.cpu().numpy().tolist())

    return np.mean(np.array(real) == np.array(pred)), classification_report(real, pred)


def prepare_data(x, y, scaler):
    X = torch.tensor(scaler.transform(x), dtype=torch.float32)
    Y = torch.tensor(y.to_numpy(), dtype=torch.long)
    return X, Y


FILEPATH = "data/winequality-red.csv"


def main():
    df = pd.read_csv(FILEPATH)
    print(f"Размерность датасета: {df.shape}")
    print(df.head())

    X = df.iloc[:, :-1]
    y = (df.iloc[:, -1] > 5).astype("int64")

    n = len(df)

    train_end = int(0.70 * n)
    validate_end = int(0.85 * n)

    X_train_raw = X.iloc[:train_end]
    y_train_raw = y.iloc[:train_end]

    X_validate_raw = X.iloc[train_end:validate_end]
    y_validate_raw = y.iloc[train_end:validate_end]

    X_test_raw = X.iloc[validate_end:]
    y_test_raw = y.iloc[validate_end:]

    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train, y_train = prepare_data(X_train_raw, y_train_raw, scaler)
    X_validate, y_validate = prepare_data(X_validate_raw, y_validate_raw, scaler)
    X_test, y_test = prepare_data(X_test_raw, y_test_raw, scaler)

    train_dataset = TensorDataset(X_train, y_train)
    validate_dataset = TensorDataset(X_validate, y_validate)
    test_dataset = TensorDataset(X_test, y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Perceptron(num_layers=2, device=device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    trainer(
        model=model,
        dataset=train_dataset,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=75,
        device=device,
    )

    model.eval()
    acc, report = testing(model, validate_dataset, device)
    print("Точность на валидационной выборке", acc)
    print(report)

    acc, report = testing(model, test_dataset, device)
    print("Точность на тестовой выборке", acc)
    print(report)

    state_dict = model.state_dict()
    for name, param in state_dict.items():
        print(name, param.size())


if __name__ == "__main__":
    main()
