import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# Конфигурация
MODEL_DIM = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 512
BATCH_SIZE = 32
SEQ_LEN = 100
FEATURES = 1
TEST_SIZE = 0.2


class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.data = torch.from_numpy(np.loadtxt(file_path)).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len * 2

    def __getitem__(self, idx):
        # Исходная последовательность (исторические данные)
        src_start = idx
        src_end = idx + self.seq_len
        src = self.data[src_start:src_end]

        # Целевая последовательность (будущие данные)
        tgt_start = src_end
        tgt_end = src_end + self.seq_len
        tgt = self.data[tgt_start:tgt_end]

        return src.unsqueeze(-1), tgt.unsqueeze(-1)  # Добавляем размерность фичи


class TimeSeriesTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(FEATURES, MODEL_DIM)
        self.pos_encoder = PositionalEncoding(MODEL_DIM)

        self.transformer = nn.Transformer(
            d_model=MODEL_DIM,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            batch_first=False
        )

        self.output_layer = nn.Linear(MODEL_DIM, FEATURES)

    def forward(self, src, tgt):
        src = self.input_proj(src).permute(1, 0, 2)
        src = self.pos_encoder(src)

        tgt = self.input_proj(tgt).permute(1, 0, 2)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt)
        return self.output_layer(output.permute(1, 0, 2))


def predict(model, src, steps):
    """Прогнозирование будущих значений"""
    model.eval()
    predictions = []
    with torch.no_grad():
        # Первый вход декодера - последний элемент исходной последовательности
        tgt = src[-1:].repeat(1, 1, 1)

        for _ in range(steps):
            output = model(src, tgt)
            next_val = output[:, -1:, :]
            tgt = torch.cat([tgt, next_val], dim=1)
            predictions.append(next_val.squeeze().cpu().numpy())
    return np.array(predictions)


def plot_prediction(original, prediction, example_idx):
    """Визуализация прогноза"""
    plt.figure(figsize=(12, 6))

    # Исходные данные
    time_original = np.arange(len(original))
    plt.plot(time_original, original, label='Original', linewidth=2)

    # Прогноз
    time_pred = np.arange(len(original) - 1, len(original) + len(prediction) - 1)
    plt.plot(time_pred, prediction, label='Prediction', linestyle='--', marker='o')

    plt.title(f"Прогноз временного ряда (Пример {example_idx})")
    plt.xlabel("Временные шаги")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.show()


# Полный процесс работы
if __name__ == "__main__":
    # 1. Загрузка данных
    dataset = TimeSeriesDataset("data.csv", SEQ_LEN)

    # 2. Разделение на train/test
    train_size = int((1 - TEST_SIZE) * len(dataset))
    train_set, test_set = random_split(dataset, [train_size, len(dataset) - train_size])

    # 3. Создание DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # 4. Инициализация модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeSeriesTransformer().to(device)

    # 5. Обучение модели
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(20):
        total_loss = 0
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output, tgt[:, 1:])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}")

    # 6. Прогнозирование на тестовых данных
    test_examples = [test_set[i] for i in np.random.choice(len(test_set), 3)]

    for idx, (src, tgt) in enumerate(test_examples):
        # Прогнозируем на длину целевой последовательности
        src = src.unsqueeze(0).to(device)  # Добавляем batch dimension
        prediction = predict(model, src, steps=SEQ_LEN)

        # Визуализация
        plot_prediction(
            original=np.concatenate([src.squeeze().cpu().numpy(), tgt.squeeze().cpu().numpy()]),
            prediction=prediction,
            example_idx=idx + 1
        )