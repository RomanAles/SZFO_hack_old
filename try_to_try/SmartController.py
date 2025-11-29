import re
import json
import vosk
import pyaudio
import noisereduce as nr
import numpy as np
from rapidfuzz import process

# Словари с командами
_label2id = {
    "отказ": 0, "отмена": 1, "подтверждение": 2, "начать осаживание": 3, "осадить на": 4,
    "продолжаем осаживание": 5, "зарядка тормозной магистрали": 6, "вышел из межвагонного пространства": 7,
    "продолжаем роспуск": 8, "растянуть автосцепки": 9, "протянуть на": 10, "отцепка": 11,
    "назад на башмак": 12, "захожу в межвагонное пространство": 13, "остановка": 14,
    "вперёд на башмак": 15, "сжать автосцепки": 16, "назад с башмака": 17, "тише": 18,
    "вперёд с башмака": 19, "прекратить зарядку тормозной магистрали": 20, "тормозить": 21, "отпустить": 22,
}

# Регулярные выражения для выделения количества вагонов (цифрами или словами)
wagon_count_pattern = re.compile(r"на (\d+|[а-яёА-ЯЁ\s]+)\s*вагон(?:а|ов)")

# Порог схожести
similarity_threshold = 80

# Словарь для чисел прописью
numbers_dict = {
    "ноль": 0, "один": 1, "два": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
    "девятьсот": 900
}

# Функция для преобразования чисел, написанных прописью
def russian_text_to_number(text):
    words = text.split()
    total = 0
    for word in words:
        if word in numbers_dict:
            total += numbers_dict[word]
    return total

# Функция для поиска наиболее похожих слов из словаря
def find_similar_words(recognized_word, vocabulary, top_n=5):
    matches = process.extract(recognized_word, vocabulary, limit=top_n)
    return matches

# Функция для перевода текста в label и attribute
def process_text(text):
    label = -1
    attribute = -1

    # Поиск наиболее похожих команд
    similar_commands = find_similar_words(text, _label2id.keys())

    if similar_commands:
        best_match = similar_commands[0]

        if best_match[1] >= similarity_threshold:
            label = _label2id[best_match[0]]

    # Проверка на наличие атрибутов
    wagon_count_match = re.search(wagon_count_pattern, text)

    if wagon_count_match:
        wagon_count_str = wagon_count_match.group(1).strip()

        if not wagon_count_str.isdigit():
            # Используем метод find_similar_words для распознавания числа
            similar_numbers = find_similar_words(wagon_count_str, numbers_dict.keys())

            if similar_numbers:
                best_number_match = similar_numbers[0]

                if best_number_match[1] >= similarity_threshold:
                    attribute = numbers_dict[best_number_match[0]]
        else:
            attribute = int(wagon_count_str)
    return {
        "text": text,
        "label": label,
        "attribute": attribute
    }

# Функция для распознавания речи с микрофона с шумоподавлением
def recognize_from_microphone():
    vosk.SetLogLevel(0)
    model = vosk.Model("vosk-model-small-ru-0.22")  # Укажите путь к вашей модели
    recognizer = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    stream.start_stream()
    print("Начинаю распознавание...")

    while True:
        data = stream.read(4000, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        reduced_noise = nr.reduce_noise(y=audio_data, sr=16000)
        reduced_noise_data = (reduced_noise * 32768.0).astype(np.int16).tobytes()

        if recognizer.AcceptWaveform(reduced_noise_data):
            result = recognizer.Result()
            text = json.loads(result)["text"]

            if text:  # Если текст распознан
                print(f"Распознано: {text}")
                processed = process_text(text)

                if processed['label'] == -1 and processed['attribute'] == -1:
                    print("Команда не распознана")
                else:
                    print(json.dumps(processed, ensure_ascii=False, indent=2))

# Запуск функции распознавания
if __name__ == "__main__":
    recognize_from_microphone()
