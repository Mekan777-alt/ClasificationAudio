import os
import shutil
import random
from sklearn.model_selection import train_test_split
import argparse


def split_dataset(source_dir, train_dir, test_dir, test_size=0.2, random_state=42):
    """
    Разделяет аудиоданные из source_dir на тренировочные и тестовые выборки
    и копирует их в соответствующие директории train_dir и test_dir.

    Args:
        source_dir: путь к исходному набору данных
        train_dir: путь к директории для тренировочных данных
        test_dir: путь к директории для тестовых данных
        test_size: доля тестовых данных (по умолчанию 0.2)
        random_state: seed для обеспечения воспроизводимости
    """
    # Создаем директории для тренировочных и тестовых данных, если их нет
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Проходим по всем классам в исходной директории
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)

        # Пропускаем, если это не директория
        if not os.path.isdir(class_dir):
            continue

        print(f"Обработка класса: {class_name}")

        # Создаем соответствующие директории в train и test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Получаем список всех аудиофайлов
        audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

        # Если файлов нет, пропускаем класс
        if not audio_files:
            print(f"Внимание: В классе {class_name} нет файлов .wav")
            continue

        # Разделяем файлы на тренировочные и тестовые
        train_files, test_files = train_test_split(
            audio_files, test_size=test_size, random_state=random_state
        )

        print(f"  Файлов всего: {len(audio_files)}")
        print(f"  Тренировочные: {len(train_files)}")
        print(f"  Тестовые: {len(test_files)}")

        # Копируем тренировочные файлы
        for filename in train_files:
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join(train_class_dir, filename)
            shutil.copy2(src_path, dst_path)

        # Копируем тестовые файлы
        for filename in test_files:
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join(test_class_dir, filename)
            shutil.copy2(src_path, dst_path)

    print("\nРазделение завершено!")
    print(f"Тренировочные данные сохранены в: {train_dir}")
    print(f"Тестовые данные сохранены в: {test_dir}")


def count_files(directory):
    """
    Подсчитывает количество файлов в каждом подкаталоге
    """
    total_count = 0
    class_counts = {}

    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)

        if os.path.isdir(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            count = len(files)
            class_counts[class_name] = count
            total_count += count

    return total_count, class_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Разделение набора аудиоданных на тренировочные и тестовые выборки')
    parser.add_argument('--source', default='dataset', help='Путь к исходному набору данных')
    parser.add_argument('--train', default='dataset_train', help='Путь для сохранения тренировочных данных')
    parser.add_argument('--test', default='dataset_test', help='Путь для сохранения тестовых данных')
    parser.add_argument('--test_size', type=float, default=0.2, help='Доля тестовых данных (по умолчанию 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Случайное начальное число для воспроизводимости')

    args = parser.parse_args()

    # Запускаем разделение
    split_dataset(args.source, args.train, args.test, args.test_size, args.seed)

    # Выводим статистику по разделенным данным
    print("\nСтатистика:")
    train_total, train_counts = count_files(args.train)
    test_total, test_counts = count_files(args.test)

    print(f"\nВсего файлов: {train_total + test_total}")
    print(f"Тренировочные файлы: {train_total} ({train_total / (train_total + test_total) * 100:.1f}%)")
    print(f"Тестовые файлы: {test_total} ({test_total / (train_total + test_total) * 100:.1f}%)")

    print("\nРаспределение по классам:")
    for class_name in sorted(train_counts.keys()):
        train_count = train_counts.get(class_name, 0)
        test_count = test_counts.get(class_name, 0)
        total = train_count + test_count

        print(f"  {class_name}: всего {total}, "
              f"тренировочные {train_count} ({train_count / total * 100:.1f}%), "
              f"тестовые {test_count} ({test_count / total * 100:.1f}%)")