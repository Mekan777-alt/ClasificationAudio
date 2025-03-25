import os
import librosa
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
import numpy as np


# Функция для автоматического сбора данных
def get_dataset_paths(dataset_dir):
    audio_paths = []
    labels = []
    class_mapping = {}
    label_id = 0

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):  # Поддерживаемые аудиоформаты
                file_path = os.path.join(root, file)
                class_name = os.path.basename(root)

                # Если новый класс, добавляем в mapping
                if class_name not in class_mapping:
                    class_mapping[class_name] = label_id
                    label_id += 1

                audio_paths.append(file_path)
                labels.append(class_mapping[class_name])

    return audio_paths, labels, class_mapping


# Класс для модели трансформера
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(self.wav2vec.config.hidden_size, num_classes)

    def forward(self, input_audio):
        output = self.wav2vec(input_audio).last_hidden_state
        pooled_output = output.mean(dim=1)  # Пуллинг
        logits = self.classifier(pooled_output)
        return logits


# Класс для датасета
class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, processor, max_length=16000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.max_length = max_length  # Максимальная длина аудиофайлов

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        audio, _ = librosa.load(audio_path, sr=16000)

        # Нормализация
        audio = librosa.util.normalize(audio)

        # Обрезка или дополнение аудиофайла до max_length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant")

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(), label


# Подготовка данных
def prepare_data(dataset_dir):
    # Сбор всех путей и меток
    audio_paths, labels, class_mapping = get_dataset_paths(dataset_dir)

    # Разделение на обучающую и тестовую выборку
    train_paths, val_paths, train_labels, val_labels = train_test_split(audio_paths, labels, test_size=0.2)

    # Загрузка процессора Wav2Vec2
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    train_dataset = AudioDataset(train_paths, train_labels, processor)
    val_dataset = AudioDataset(val_paths, val_labels, processor)

    return train_dataset, val_dataset, class_mapping


# Функция для вычисления точности
def calculate_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).sum().item() / len(labels)


# Обучение модели
def train_model(train_dataset, val_dataset, num_classes, num_epochs=10, batch_size=4, lr=1e-5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Инициализация модели, функции потерь и оптимизатора
    model = AudioClassifier(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0

        # Тренировочный цикл
        for inputs, labels in train_loader:
            inputs = inputs.to(torch.float32)
            labels = labels.clone().detach().long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)

        # Средняя точность и потери на обучающей выборке
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_accuracy / len(train_loader)

        # Оценка на валидационной выборке
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(torch.float32)
                labels = labels.clone().detach().long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_accuracy += calculate_accuracy(outputs, labels)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_accuracy / len(val_loader)

        # Обновление learning rate через scheduler
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}"
        )

    return model


# Основной блок
if __name__ == "__main__":
    dataset_dir = "dataset"

    # Подготовка данных
    train_dataset, val_dataset, class_mapping = prepare_data(dataset_dir)
    print(f"Найдено классов: {len(class_mapping)}")
    print("Классы и их метки:", class_mapping)

    # Обучение модели
    num_classes = len(class_mapping)
    model = train_model(train_dataset, val_dataset, num_classes=num_classes)

    # Сохранение модели
    torch.save(model.state_dict(), "audio_classifier.pth")
    print("Модель сохранена в 'audio_classifier.pth'")
    result = """
    Epoch 10/10, Train Loss: 0.1646, Train Acc: 0.9473, Val Loss: 0.2946, Val Acc: 0.9060
    
    Метрики на 10-й эпохе
	1.	Train Loss: 0.1646
	•	Значение функции потерь на тренировочном наборе стало очень низким.
	•	Это говорит о том, что модель хорошо обучилась и правильно предсказывает метки на тренировочных данных.
	2.	Train Acc: 0.9473
	•	Точность на тренировочном наборе ~94.73%.
	•	Высокая точность показывает, что модель хорошо справляется с обучающими данными, но при этом не достигает 100%, что важно, чтобы избежать переобучения.
	3.	Val Loss: 0.2946
	•	Значение функции потерь на валидационном наборе немного выше, чем на тренировочном (0.2946 против 0.1646), но все еще довольно низкое.
	•	Это нормальное поведение, так как валидационные данные не используются для обучения модели.
	4.	Val Acc: 0.9060
	•	Точность на валидационном наборе ~90.60%.
	•	Это высокий показатель, который подтверждает, что модель хорошо обобщает свои знания на данных, которых она не видела.
	
	Обучение завершено успешно! Модель имеет высокую точность как на тренировочных данных (~94.73%), так и на валидации 
	(~90.60%). Ее можно использовать для классификации аудиоданных. Если хочешь улучшить точность, 
	можно увеличить объем данных или применить аугментацию.
"""