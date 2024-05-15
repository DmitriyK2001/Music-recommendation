# Рекомендация музыки

Целью проекта состоит в создании системы музыкальных рекомедаций для пользователей стриминговых сервисов. Проект востребован, потому что рынок музыкальных сервисов растет с каждым годом.
# Структура проекта
```bash
.
├── Dockerfile
├── README.md
├── command.py
├── music
│   ├── 10000.txt
│   ├── __pycache__
│   │   ├── data_loader.cpython-311.pyc
│   │   ├── infer.cpython-311.pyc
│   │   ├── model.cpython-311.pyc
│   │   └── train.cpython-311.pyc
│   ├── data_loader.py
│   ├── infer.py
│   ├── model.py
│   ├── song_data.csv
│   └── train.py
├── poetry.lock
└── pyproject.toml
```
# Данные
Для решения задачи собираюсь использовать Million Song Dataset(http://millionsongdataset.com), как можно понять из названия, он содержит данные о миллионе треков, для сравнения, в Apple Music сейчас примерно 100 миллионов треков, в Spotify около 50, в Яндекс.Музыке 76, то есть датасет адекватен по размеру в сравнении с этими сервисами. Среди параметров трека его название, имя исполнителя, теги, продолжительность трека, список похожих исполнителей, год выпуска, громкость и т. д.
# Подход к моделированию
На данный момент используется библиотека scipy, используется метод матричной факторизации, разложение матрицы полезности на две матрицы, чтобы найти скрытые признаки объектов, как пользователей, так и песен. Чтобы выдать рекомендации для пользователя, соответсвующая ему строка умножается на матрицу с признаками песен, и определяются элементы из строки с максимальными оценками.
# Способ предсказания
Первым шагом идет обучение модели на нашем датасете. После обучения модель нужно будет обернуть в пайплайн, в нем необходимо провести предобработку треков с выделением необходимых признаков, работу модели на новых треках и вывод "похожих" треков. Очевидный вариант финального применения - использование в музыкальном сервисе.
