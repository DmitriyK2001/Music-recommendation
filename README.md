# Рекомендация музыки

Целью проекта состоит в создании системы музыкальных рекомедаций для пользователей стриминговых сервисов. Проект востребован, потому что рынок музыкальных сервисов растет с каждым годом.
# Структура проекта
.
├── Dockerfile
├── README.md
├── command.py
├── music
│   ├── infer.py
│   ├── model.py
│   └── train.py
├── poetry.lock
└── pyproject.toml

# Данные
Для решения задачи собираюсь использовать Million Song Dataset(http://millionsongdataset.com), как можно понять из названия, он содержит данные о миллионе треков, для сравнения, в Apple Music сейчас примерно 100 миллионов треков, в Spotify около 50, в Яндекс.Музыке 76, то есть датасет адекватен по размеру в сравнении с этими сервисами. Среди параметров трека его название, имя исполнителя, теги, продолжительность трека, список похожих исполнителей, год выпуска, громкость и т. д.
# Подход к моделированию
Используя библиотеку Pytorch, модель будет по признакам подбирать "похожие" треки. Под "похожими" понимаются близкие в признаковом пространстве. Предполагается, что это будет нейросеть, которая внешне работает аналогично kNN, но в то же время "под капотом" принципиально отличается от него, т.к. очевидно, что хранить в памяти и считать расстояние до миллиона точек для каждой точки невыполнимая задача.
# Способ предсказания
Первым шагом идет обучение модели на нашем датасете. После обучения модель нужно будет обернуть в пайплайн, в нем необходимо провести предобработку треков с выделением необходимых признаков, работу модели на новых треках и вывод "похожих" треков. Очевидный вариант финального применения - использование в музыкальном сервисе. 
