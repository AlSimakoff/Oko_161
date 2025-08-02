Вот `README.md` для твоего Python-проекта по видеоаналитике транспортных средств с использованием YOLOv11, LPRNet и компьютерного зрения:

---

# 🚘 AVID — Automatic Vehicle Identification & Detection

**AVID** — это интеллектуальная система анализа видео, способная обнаруживать автомобили, распознавать номера, определять тип и цвет транспортных средств и вести журнал событий. Использует глубокие нейросети (YOLOv11, LPRNet), CV-пайплайн, SQLite и API-интеграции.

---

## 🔍 Основные возможности

* 🧠 Обнаружение объектов с помощью YOLOv11
* 🔢 Распознавание номерных знаков с LPRNet
* 🎨 Определение цвета автомобиля с помощью цветовой гистограммы и k-NN
* 🚛 Классификация транспорта: `car`, `truck`, `bus`
* 📦 Поддержка видеофайлов (по кадрам)
* 🧾 Сохранение результатов в локальную SQLite БД и внешний сервер (REST API)
* 📊 Визуализация объектов и ROI прямо на кадрах

---

## 📦 Стек технологий

* `Python 3.10+`
* `PyTorch` + `OpenCV` + `NumPy`
* `YOLOv11` (Ultralytics)
* `LPRNet` для распознавания номерных знаков
* `SQLite` для хранения логов
* `requests` для API-интеграции
* KNN-классификатор для цвета

---

## 🚀 Быстрый запуск

### 🔧 Установка зависимостей

```bash
pip install -r requirements.txt
```

### 🗃️ Подготовка данных

1. Подготовь видео (`*.mp4`/`*.h264`) в папке `data/video_test/`.
2. Убедись, что у тебя есть:

   * обученная модель YOLOv11 (`*.pt`)
   * обученная модель LPRNet (`*.pth`)
   * набор обучающих изображений по цветам (`colour_detection/training_dataset/`)

### ⚙️ Конфигурация

Настраивается в файле `settings.py`:

```python
DEVICE = 'cpu'  # или 'cuda'

# Пути к моделям и видео
FILE_PATH = "data/video_test/test_Blog.h264"
YOLO_MODEL_PATH = "detection_YOLOv11/southcity.pt"
LPR_MODEL_PATH = "lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth"

# БД и API
database_path = 'data/database/oko161.db'
server_url = 'http://localhost:5000/oko161'
name_company_object = 'BolshoiLog'
```

---

## ▶️ Запуск

```bash
python main.py
```

Или укажи свои параметры прямо в `main()`.

---

## 📊 Результаты

* 🚘 Обнаруженные машины отображаются с рамками:

  * Красный — `car`
  * Зелёный — `truck`
  * Синий — `bus`
* 📷 На номере отображается:

  * Номер
  * Цвет
  * Тип авто
* 🧾 Каждое распознавание сохраняется в:

  * локальную SQLite БД
  * внешний сервер через `POST`

---

## 📂 Структура проекта

```plaintext
├── main.py                # Точка входа
├── detect.py              # Главная логика анализа видео
├── client.py              # Работа с REST API
├── db.py                  # Работа с SQLite
├── settings.py            # Конфигурация
├── detection_YOLOv11/     # YOLO-модели и tracking
├── lpr_net/               # LPRNet модель и utils
├── colour_detection/      # Определение цвета (обучение + распознавание)
└── data/
    ├── video_test/        # Видео для анализа
    └── database/          # SQLite база
```

---

## 🧪 Пример записи в БД

```json
{
  "time": "2025-08-02 16:03:01",
  "license_number": "A123BC77",
  "color": "white",
  "type_auto": "car"
}
```

---

## 📡 REST API

Отправка данных:

```http
POST http://localhost:5000/oko161
Content-Type: application/json

{
  "time": "2025-08-02 16:03:01",
  "license_number": "A123BC77",
  "color": "white",
  "type_auto": "car",
  "table_name": "BolshoiLog"
}
```

Получение всех записей:

```http
GET http://localhost:5000/oko161?table=BolshoiLog
```

---

## 🔐 Примечания

* Поддерживает только российские номерные знаки (шаблон: `[A-Z][0-9]{3}[A-Z]{2}[0-9]{2,3}`)
* Область интереса задаётся вручную (`settings.DETECTION_AREA`)
* KNN-классификатор запускает обучение при первом вызове

---

## 📜 Лицензия

Проект распространяется под лицензией MIT.
YOLOv11 и LPRNet используются в исследовательских и учебных целях.

---


