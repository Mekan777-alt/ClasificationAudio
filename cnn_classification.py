import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
from scipy.ndimage import zoom

# Настройка seed для воспроизводимости результатов
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Параметры для обработки аудио
SAMPLE_RATE = 44100  # Частота дискретизации
DURATION = 2  # Длительность аудиофрагмента в секундах
N_MFCC = 40  # Количество MFCC коэффициентов
N_FFT = 2048  # Размер окна для FFT
HOP_LENGTH = 512  # Шаг между окнами
N_MELS = 128  # Количество мел-фильтров
MAX_SAMPLES = SAMPLE_RATE * DURATION  # Максимальное количество отсчетов в фрагменте

# Параметры для обучения
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
NUM_CLASSES = 7  # Число классов

# Имена классов
CLASS_NAMES = [
    "Accelerating,revving,vroom",
    "Ambulance(siren)",
    "Car",
    "Motorcycle",
    "Motorvehicle(road)",
    "Siren",
    "Truck"
]


# Класс для создания датасета
class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths = []
        self.labels = []
        self.class_indices = {}

        # Получаем список файлов и меток
        for i, class_name in enumerate(os.listdir(data_dir)):
            self.class_indices[class_name] = i
            class_dir = os.path.join(data_dir, class_name)

            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.wav'):
                        self.file_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(i)

        print(f"Загружено {len(self.file_paths)} файлов из {len(self.class_indices)} классов")
        print(f"Классы: {list(self.class_indices.keys())}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Загрузка аудиофайла
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Обеспечиваем одинаковую длину всех аудиофрагментов
        if len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        else:
            audio = audio[:MAX_SAMPLES]

        # Извлечение MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Нормализация MFCC
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)

        # Получаем мел-спектрограмму
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
                                                         hop_length=HOP_LENGTH)
        # Преобразуем в логарифмическую шкалу (в дБ)
        log_mel = librosa.power_to_db(mel_spectrogram)

        # Нормализация спектрограммы
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-10)

        # Вывод размеров для отладки
        if idx == 0:
            print(f"MFCC shape: {mfcc.shape}")
            print(f"Mel-spectrogram shape: {log_mel.shape}")

        # Изменение размера (ресэмплинг) мел-спектрограммы до размера MFCC
        # Определяем коэффициенты масштабирования для каждой оси
        scale_factors = (mfcc.shape[0] / log_mel.shape[0], mfcc.shape[1] / log_mel.shape[1])

        # Изменяем размер мел-спектрограммы
        log_mel_resized = zoom(log_mel, scale_factors, order=1)

        if idx == 0:
            print(f"Resized mel-spectrogram shape: {log_mel_resized.shape}")
            print(f"Now both features have the same shape!")

        # Объединяем признаки в один тензор (канал 0 - MFCC, канал 1 - мел-спектрограмма)
        features = np.stack([mfcc, log_mel_resized])

        # Преобразуем в тензор PyTorch
        tensor_features = torch.FloatTensor(features)
        tensor_label = torch.LongTensor([label])[0]

        return tensor_features, tensor_label


# Класс для обработки аудио с добавлением шума
class NoisyAudioDataset(Dataset):
    def __init__(self, data_dir, snr_db, target_class=None):
        self.data_dir = data_dir
        self.snr_db = snr_db
        self.file_paths = []
        self.labels = []
        self.class_indices = {}

        # Получаем список файлов и меток
        for i, class_name in enumerate(os.listdir(data_dir)):
            # Если указан конкретный класс, берем только его
            if target_class is not None and class_name != target_class:
                continue

            self.class_indices[class_name] = i
            class_dir = os.path.join(data_dir, class_name)

            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.wav'):
                        self.file_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(i)

        print(f"Загружено {len(self.file_paths)} файлов для SNR = {snr_db} dB")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Загрузка аудиофайла
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Обеспечиваем одинаковую длину всех аудиофрагментов
        if len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        else:
            audio = audio[:MAX_SAMPLES]

        # Добавляем шум
        audio = self.add_white_noise(audio, self.snr_db)

        # Извлечение MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Нормализация MFCC
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)

        # Получаем мел-спектрограмму
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
                                                         hop_length=HOP_LENGTH)
        # Преобразуем в логарифмическую шкалу (в дБ)
        log_mel = librosa.power_to_db(mel_spectrogram)

        # Нормализация спектрограммы
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-10)

        # Изменение размера (ресэмплинг) мел-спектрограммы до размера MFCC
        # Определяем коэффициенты масштабирования для каждой оси
        scale_factors = (mfcc.shape[0] / log_mel.shape[0], mfcc.shape[1] / log_mel.shape[1])

        # Изменяем размер мел-спектрограммы
        log_mel_resized = zoom(log_mel, scale_factors, order=1)

        # Объединяем признаки в один тензор (канал 0 - MFCC, канал 1 - мел-спектрограмма)
        features = np.stack([mfcc, log_mel_resized])

        # Преобразуем в тензор PyTorch
        tensor_features = torch.FloatTensor(features)
        tensor_label = torch.LongTensor([label])[0]

        return tensor_features, tensor_label

    def add_white_noise(self, audio, snr_db):
        # Переводим SNR из dB в линейную шкалу
        snr = 10 ** (snr_db / 10)

        # Вычисляем мощность сигнала
        signal_power = np.mean(audio ** 2)

        # Вычисляем мощность шума
        noise_power = signal_power / snr

        # Генерируем белый шум
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))

        # Добавляем шум к сигналу
        noisy_audio = audio + noise

        return noisy_audio


# Функция для проверки размерности данных перед обучением
def compute_output_shape(model, input_shape):
    with torch.no_grad():
        x = torch.rand(1, *input_shape)
        for name, layer in model.named_children():
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    x = sublayer(x)
            else:
                x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                print(f"Shape after {name} ({type(layer).__name__}): {x.shape}")
    return x.shape


# Архитектура CNN
class CNNModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_shape=None):
        super(CNNModel, self).__init__()

        # Первый сверточный блок
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)  # 2 входных канала: MFCC и мел-спектрограмма
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Второй сверточный блок
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Третий сверточный блок
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Динамическое определение размерности полносвязного слоя
        if input_shape is not None:
            # Создаем dummy тензор для расчета выходной размерности
            dummy_input = torch.zeros(1, *input_shape)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            fc_input_size = x.numel()
            print(f"Рассчитанная входная размерность для FC слоя: {fc_input_size}")
        else:
            # Если форма не указана, используем значение по умолчанию
            fc_input_size = 128 * 5 * 16  # Примерное значение, может потребоваться корректировка
            print(f"Используется размерность FC слоя по умолчанию: {fc_input_size}")

        # Полносвязные слои
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Первый блок
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Второй блок
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Третий блок
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Добавлена отладочная информация о размерности
        batch_size = x.size(0)
        if batch_size == 1:  # Только для первого батча выводим размерность
            print(f"Размерность перед reshape: {x.shape}")

        # Reshape для полносвязных слоев
        x = x.view(x.size(0), -1)

        # Полносвязные слои
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


# Функция для обучения модели
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    # Для отслеживания лучшей модели
    best_val_loss = float('inf')
    best_model_state = None

    # Для сохранения истории обучения
    train_losses = []
    val_losses = []

    # Добавлен early stopping
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        running_corrects = 0  # Добавлен счетчик правильных предсказаний

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]'):
            # Перемещаем данные на устройство
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Очищаем градиенты
            optimizer.zero_grad()

            try:
                # Forward pass
                outputs = model(inputs)

                # Вычисляем ошибку
                loss = criterion(outputs, labels)

                # Backpropagation
                loss.backward()

                # Обновляем веса
                optimizer.step()

                # Статистика
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            except RuntimeError as e:
                print(f"Ошибка в batch: {e}")
                print(f"Размер входных данных: {inputs.shape}")
                continue

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Валидация
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                # Перемещаем данные на устройство
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Вычисляем ошибку
                loss = criterion(outputs, labels)

                # Статистика
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs} - '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

        # Сохраняем лучшую модель
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print(f"Early stopping на эпохе {epoch + 1}")
            break

    # Загружаем состояние лучшей модели
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


# Функция для оценки модели
def evaluate_model(model, test_loader, device, class_names=None):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            # Перемещаем данные на устройство
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Получаем предсказания
            _, preds = torch.max(outputs, 1)

            # Сохраняем результаты
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Вычисляем метрики
    precision = precision_score(all_labels, all_predictions, average=None)
    recall = recall_score(all_labels, all_predictions, average=None)
    f1 = f1_score(all_labels, all_predictions, average=None)

    # Если заданы имена классов, выводим детальный отчет
    if class_names:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))

        # Матрица ошибок
        cm = confusion_matrix(all_labels, all_predictions)
        print("\nConfusion Matrix:")
        print(cm)

    # Возвращаем метрики
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }

    return metrics


# Функция для измерения времени инференса
def measure_inference_time(model, test_loader, device, num_runs=100):
    model.eval()

    # Выбираем один батч для тестирования
    for inputs, _ in test_loader:
        # Перемещаем на устройство
        inputs = inputs.to(device)

        # Прогреваем модель
        with torch.no_grad():
            _ = model(inputs[:1])

        # Измеряем время инференса для одного образца
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(inputs[:1])

        end_time = time.time()

        avg_time = (end_time - start_time) * 1000 / num_runs  # в миллисекундах

        return avg_time


# Функция для визуализации матрицы ошибок
def plot_confusion_matrix(y_true, y_pred, class_names, output_file="cnn_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))

    ax = sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     cbar_kws={'label': 'Нормализованное значение'})

    plt.title('Матрица ошибок для модели CNN')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


# Функция для визуализации кривых обучения
def plot_learning_curves(train_losses, val_losses, output_file="cnn_learning_curves.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label='Train Loss', color='#4472C4', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='#ED7D31', linewidth=2)

    plt.title('Кривые обучения CNN')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


# Функция для визуализации зависимости F1-меры от уровня шума
def plot_noise_robustness(snr_levels, f1_scores, target_class, output_file="cnn_noise_robustness.png"):
    plt.figure(figsize=(12, 7))

    plt.plot(snr_levels, f1_scores, 'o-', color='#5B9BD5', linewidth=2.5, markersize=8)

    plt.xlabel('Соотношение сигнал/шум (SNR), дБ')
    plt.ylabel(f'F1-мера для класса "{target_class}"')
    plt.title('Устойчивость CNN к шумовым помехам')

    plt.invert_xaxis()  # Шум увеличивается слева направо
    plt.grid(linestyle='--', alpha=0.7)
    plt.ylim(0.2, 1.0)

    # Добавляем подписи значений
    for i, y in enumerate(f1_scores):
        plt.annotate(f'{y:.2f}', (snr_levels[i], y), xytext=(0, 10),
                     textcoords='offset points', ha='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


# Основная функция
def main():
    # Создаем папку для результатов
    os.makedirs("results", exist_ok=True)

    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Пути к данным
    train_dir = "dataset_train"  # Путь к тренировочным данным
    test_dir = "dataset_test"  # Путь к тестовым данным

    # Проверка наличия директорий с данными
    if not os.path.exists(train_dir):
        print(f"Ошибка: Директория {train_dir} не существует!")
        return
    if not os.path.exists(test_dir):
        print(f"Ошибка: Директория {test_dir} не существует!")
        return

    # Создаем датасеты
    try:
        train_dataset = AudioDataset(train_dir)
        test_dataset = AudioDataset(test_dir)
    except Exception as e:
        print(f"Ошибка при создании датасетов: {e}")
        return

    # Получаем один пример данных для определения формы входных данных
    sample_input, _ = train_dataset[0]
    print(f"Форма входного тензора: {sample_input.shape}")

    # Разделяем тренировочные данные на обучающую и валидационную выборки
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(f"Размер обучающей выборки: {train_size}")
    print(f"Размер валидационной выборки: {val_size}")
    print(f"Размер тестовой выборки: {len(test_dataset)}")

    # Создаем даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Создаем модель с динамическим определением размеров
    model = CNNModel(num_classes=NUM_CLASSES, input_shape=sample_input.shape).to(device)

    # Выводим архитектуру модели
    print(model)

    # Определяем функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Обучаем модель
    print("Обучение модели...")
    try:
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, EPOCHS, device
        )

        # Визуализируем кривые обучения
        plot_learning_curves(train_losses, val_losses, "results/cnn_learning_curves.png")

        # Оцениваем модель на тестовой выборке
        print("Оценка модели на тестовой выборке...")
        metrics = evaluate_model(model, test_loader, device, CLASS_NAMES)

        # Сохраняем метрики
        np.savez("results/cnn_metrics.npz",
                 precision=metrics['precision'],
                 recall=metrics['recall'],
                 f1=metrics['f1'])

        # Визуализируем матрицу ошибок
        plot_confusion_matrix(metrics['labels'], metrics['predictions'], CLASS_NAMES,
                              "results/cnn_confusion_matrix.png")

        # Измеряем время инференса
        inference_time = measure_inference_time(model, test_loader, device)
        print(f"Среднее время инференса: {inference_time:.2f} мс")

        # Сохраняем время инференса
        with open("results/cnn_inference_time.txt", "w") as f:
            f.write(f"{inference_time:.2f}")

        # Тестирование устойчивости к шуму
        print("Тестирование устойчивости к шуму...")

        # Выбираем целевой класс для тестирования с шумом
        # Мы используем класс "Car"
        target_class = "Car"

        snr_levels = [20, 15, 10, 5, 0]
        f1_scores_noise = []

        # Для каждого уровня SNR
        for snr in snr_levels:
            print(f"Тестирование с SNR = {snr} dB...")

            # Создаем датасет с шумом для целевого класса
            noisy_dataset = NoisyAudioDataset(test_dir, snr, target_class=target_class)
            noisy_loader = DataLoader(noisy_dataset, batch_size=BATCH_SIZE)

            # Оцениваем модель
            model.eval()
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in tqdm(noisy_loader, desc=f'Evaluating SNR={snr}'):
                    # Перемещаем данные на устройство
                    inputs = inputs.to(device)

                    # Forward pass
                    outputs = model(inputs)

                    # Получаем предсказания
                    _, preds = torch.max(outputs, 1)

                    # Сохраняем результаты
                    all_predictions.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Вычисляем F1-меру для целевого класса
            target_idx = CLASS_NAMES.index(target_class)

            # Создаем бинарные метки: 1 - целевой класс, 0 - другие классы
            binary_true = [1 if label == target_idx else 0 for label in all_labels]
            binary_pred = [1 if pred == target_idx else 0 for pred in all_predictions]

            f1 = f1_score(binary_true, binary_pred, pos_label=1)
            f1_scores_noise.append(f1)

            print(f"F1-мера для класса {target_class} при SNR = {snr} dB: {f1:.4f}")

        # Сохраняем результаты
        np.savez("results/cnn_noise_results.npz",
                 snr_levels=snr_levels,
                 f1_scores=f1_scores_noise)

        # Визуализируем зависимость от шума
        plot_noise_robustness(snr_levels, f1_scores_noise, target_class, "results/cnn_noise_robustness.png")

        # Сохраняем модель
        torch.save(model.state_dict(), "results/cnn_model.pth")

        print("Готово! Все результаты сохранены в папке 'results'")

    except Exception as e:
        print(f"Ошибка при обучении или оценке модели: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()