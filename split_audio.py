import os
from pydub import AudioSegment
from pydub.utils import make_chunks

def split_audio_in_folder(folder_path):
    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print("Папка не найдена!")
        return

    # Получаем список всех .wav файлов в папке
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not wav_files:
        print("В папке нет файлов .wav")
        return

    # Обрабатываем каждый файл
    for wav_file in wav_files:
        file_path = os.path.join(folder_path, wav_file)
        file_name = os.path.splitext(wav_file)[0]

        # Создаём папку для каждого файла
        output_dir = os.path.join(folder_path, file_name)
        os.makedirs(output_dir, exist_ok=True)

        # Загружаем аудиофайл
        audio = AudioSegment.from_file(file_path)

        # Длина фрагмента в миллисекундах (4 секунды)
        chunk_length = 4 * 1000

        # Разбиваем на фрагменты
        chunks = make_chunks(audio, chunk_length)

        # Сохраняем каждый фрагмент в папке
        for i, chunk in enumerate(chunks):
            chunk_name = os.path.join(output_dir, f"{file_name}_part_{i + 1}.wav")
            chunk.export(chunk_name, format="wav")
            print(f"Сохранён: {chunk_name}")

    print("Разделение завершено для всех файлов!")

# Пример использования
split_audio_in_folder("Dataset Nov 2021")