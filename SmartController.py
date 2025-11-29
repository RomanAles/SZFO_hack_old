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
similarity_threshold = 85

# Словарь для чисел прописью
numbers_dict = {
    "ноль": 0, "один": 1, "два": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
    "двадцать": 20, "двадцать один": 21, "двадцать два": 22, "двадцать три": 23,
    "двадцать четыре": 24, "двадцать пять": 25, "двадцать шесть": 26,
    "двадцать семь": 27, "двадцать восемь": 28, "двадцать девять": 29,
    "тридцать": 30, "тридцать один": 31, "тридцать два": 32, "тридцать три": 33,
    "тридцать четыре": 34, "тридцать пять": 35, "тридцать шесть": 36,
    "тридцать семь": 37, "тридцать восемь": 38, "тридцать девять": 39,
    "сорок": 40, "сорок один": 41, "сорок два": 42, "сорок три": 43,
    "сорок четыре": 44, "сорок пять": 45, "сорок шесть": 46,
    "сорок семь": 47, "сорок восемь": 48, "сорок девять": 49,
    "пятьдесят": 50, "пятьдесят один": 51, "пятьдесят два": 52, "пятьдесят три": 53,
    "пятьдесят четыре": 54, "пятьдесят пять": 55, "пятьдесят шесть": 56,
    "пятьдесят семь": 57, "пятьдесят восемь": 58, "пятьдесят девять": 59,
    "шестьдесят": 60, "шестьдесят один": 61, "шестьдесят два": 62, "шестьдесят три": 63,
    "шестьдесят четыре": 64, "шестьдесят пять": 65, "шестьдесят шесть": 66,
    "шестьдесят семь": 67, "шестьдесят восемь": 68, "шестьдесят девять": 69,
    "семьдесят": 70, "семьдесят один": 71, "семьдесят два": 72, "семьдесят три": 73,
    "семьдесят четыре": 74, "семьдесят пять": 75, "семьдесят шесть": 76,
    "семьдесят семь": 77, "семьдесят восемь": 78, "семьдесят девять": 79,
    "восемьдесят": 80, "восемьдесят один": 81, "восемьдесят два": 82, "восемьдесят три": 83,
    "восемьдесят четыре": 84, "восемьдесят пять": 85, "восемьдесят шесть": 86,
    "восемьдесят семь": 87, "восемьдесят восемь": 88, "восемьдесят девять": 89,
    "девяносто": 90, "девяносто один": 91, "девяносто два": 92, "девяносто три": 93,
    "девяносто четыре": 94, "девяносто пять": 95, "девяносто шесть": 96,
    "девяносто семь": 97, "девяносто восемь": 98, "девяносто девять": 99,
    "сто": 100
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
        if best_match[1] >= similarity_threshold: # Сохранение наиболее схожей команды
            label = _label2id[best_match[0]]

    # Проверка на наличие атрибутов
    wagon_count_match = re.search(wagon_count_pattern, text)
    if wagon_count_match:
        wagon_count_str = wagon_count_match.group(1).strip()
        if not wagon_count_str.isdigit(): # Если атрибут не содержит цифр в строке -> поиск похожих чисел
            # Используем метод find_similar_words для распознавания числа
            similar_numbers = find_similar_words(wagon_count_str, numbers_dict.keys())
            if similar_numbers:
                best_number_match = similar_numbers[0]
                if best_number_match[1] >= similarity_threshold: # Сохранение наиболее похожего числа
                    attribute = numbers_dict[best_number_match[0]]
        else: # Если атрибут содержит цифры в строке
            attribute = int(wagon_count_str)
    return {
        "text": text,
        "label": label,
        "attribute": attribute
    }

# Функция для распознавания речи с микрофона с шумоподавлением
def recognize_from_microphone():
    # Подключение модели vosk
    vosk.SetLogLevel(0)
    model = vosk.Model("vosk-model-small-ru-0.22")  # Укажите путь к вашей модели
    recognizer = vosk.KaldiRecognizer(model, 16000)

    # Подключение записи звука с микрофона
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    stream.start_stream()

    # Начало распознания услышанного текста
    print("Начинаю распознавание...")
    while True:
        # Считывание звука с микрофона
        data = stream.read(4000, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Использование шумоподавление на записанный звук с микрофона
        reduced_noise = nr.reduce_noise(y=audio_data, sr=16000)
        reduced_noise_data = (reduced_noise * 32768.0).astype(np.int16).tobytes()

        # Проверка записанного текста
        if recognizer.AcceptWaveform(reduced_noise_data):
            result = recognizer.Result()
            text = json.loads(result)["text"]
            if text:  # Если текст распознан
                print(f"Распознано: {text}")
                processed = process_text(text)
                if processed['label'] == -1 and processed['attribute'] == -1: # Если текст распознан неправильно
                    print("Команда не распознана")
                else: # Если текст распознан правильно
                    print(json.dumps(processed, ensure_ascii=False, indent=2))

# Запуск функции распознавания
if __name__ == "__main__":
    recognize_from_microphone()