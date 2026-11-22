# Классификатор овощей (трек A)

Модель узнаёт на фото один из шести овощей: Capsicum, Carrot, Cucumber, Potato, Raddish, Squash.
Это сеть EfficientNet-B0, уже обученная на миллионе картинок ImageNet, которую мы доучили на датасете
[preethi866/vegetable-dataset](https://www.kaggle.com/datasets/preethi866/vegetable-dataset).

## Главный результат

Почти весь датасет (95%) — кадры 19 видео, а соседние кадры почти одинаковые. Поэтому проверяем
честно: делим данные целыми видео, и каждое фото проверяет модель, которая его видео не видела.

| Проверка | Accuracy | Macro-F1 |
| --- | --- | --- |
| Наша: по видео, все 3 879 фото | 0,937 | 0,920 |
| Автора датасета: его train → его test | 1,000 | 1,000 |

Accuracy — доля верных ответов. Macro-F1 — средняя оценка по классам, где каждый класс весит одинаково.

100% на тесте автора — не качество, а утечка: 717 из 799 его тестовых фото лежат и в обучающей части.

## Как запустить predict.py

```bash
pip install -r requirements.txt
python predict.py --input путь/к/папке --output preds.csv
```

* Работает на обычном процессоре, интернет не нужен. Веса лежат в `weights/model.pt` (16 МБ).
* Фото во вложенных папках тоже обрабатываются, файлы, которые не картинки, пропускаются.
* В `preds.csv` три колонки: `filename` — путь к фото внутри папки, `label` — класс,
  `confidence` — уверенность модели от 0 до 1.
* Если файл не читается, он получает `unknown` с уверенностью 0, а остальные фото обрабатываются дальше.
* Фото iPhone в формате HEIC читаются, если поставить `pip install pillow-heif`.
* Другие параметры: `--weights` — другой файл весов, `--batch` — сколько фото за раз (по умолчанию 16),
  `--threads` — сколько потоков процессора, `--benchmark N` — замерить время одной только сети,
  `--unknown-threshold` — порог уверенности для `unknown` (по умолчанию выключен).

**Время на одно фото** (процессор Intel Xeon 2,1 ГГц, 2 потока):

| Что | Время |
| --- | --- |
| Только сеть | 22 мс |
| Кадр 200×200 вместе с чтением файла | 32 мс |
| Фото 10 Мп вместе с чтением файла | около 100 мс |

## Что лежит в репозитории

| Файл | Зачем |
| --- | --- |
| `audit_dataset.py` | Проверка данных: размеры, копии, похожие снимки |
| `check_author_split.py` | Проверка разбиения автора: сколько его тестовых фото есть в обучении |
| `make_splits.py` | Наше разбиение на две половины по видео → `splits.csv` |
| `splits.csv`, `author_split.csv` | Наше разбиение и разбиение автора |
| `common.py` | Общий код: классы, чтение и подготовка фото, модель |
| `train.py` | Обучение и подсчёт метрик |
| `gradcam.py` | Картинки Grad-CAM: куда смотрит модель |
| `predict.py` | Ответы для папки с фото |
| `kaggle_train.ipynb` | Всё обучение одним ноутбуком на Kaggle |
| `weights/model.pt` | Веса готовой модели |

## Как повторить всё с нуля

### 1. Скачать данные

Скачайте архив со [страницы датасета](https://www.kaggle.com/datasets/preethi866/vegetable-dataset)
(кнопка Download) и распакуйте в папку `data_raw/`. Должно получиться так:

```
data_raw/Dataset/Dataset/<Класс>/*.jpg
data_raw/final_dataset/final_dataset/{train,val,test}/<Класс>/*.jpg
```

### 2. Проверить данные (около 2 минут на процессоре)

```bash
pip install -r requirements-train.txt
python audit_dataset.py --data data_raw --out audit_out
python check_author_split.py --inventory audit_out/inventory.csv
```

Первый скрипт пишет сводку в `audit_out/summary.txt`. Второй печатает, сколько тестовых фото автора
повторяются в его обучающей части.

### 3. Разделить данные

```bash
python make_splits.py --data data_raw --out splits.csv
```

Нужен scikit-learn 1.8 или новее: в старых версиях есть ошибка, из-за которой разбиение получается
другим. С новой версией файл совпадает с `splits.csv` из репозитория байт в байт (проверено на Windows
и Linux).

### 4. Обучить модель

Проще всего на Kaggle, там бесплатная видеокарта:

1. Импортируйте `kaggle_train.ipynb`.
2. Добавьте датасет: Input → Add Input.
3. Включите GPU T4 и интернет.
4. Нажмите Run All. Всё займёт около 20–25 минут.

Ноутбук сам скачивает этот репозиторий и запускает те же команды, что ниже. Чтобы обучить у себя
(лучше с видеокартой, на процессоре это намного дольше):

```bash
python train.py --data data_raw --splits splits.csv --test-fold 0 --out runs/fold0
python train.py --data data_raw --splits splits.csv --test-fold 1 --out runs/fold1
python train.py --data data_raw --splits author_split.csv --train-folds train --test-fold test --out runs/author
python train.py --data data_raw --splits splits.csv --test-fold none --out runs/final
python gradcam.py --data data_raw --runs runs --out runs/gradcam
```

* `fold0` и `fold1` — две модели: каждая учится на одной половине данных и проверяется на другой.
  Общий итог по всем фото считает ячейка «Итог по обоим фолдам» в ноутбуке.
* `author` — та же модель на разбиении автора, только для сравнения.
* `final` — модель на всех фото. Её файл `runs/final/model.pt` надо скопировать в `weights/model.pt`.

Как учим: сначала 2 эпохи (прохода по всем фото) учится только последний слой, потом 6 эпох вся сеть
маленькими шагами. Повторные запуски дают те же ответы на всех проверочных фото.

## Версии библиотек

* `predict.py`: в пустом окружении `pip install -r requirements.txt` поставил torch 2.15.0,
  torchvision 0.29.2 и Pillow 13.0.0 (Linux, Python 3.13), и скрипт отработал. На Linux pip вместе с
  torch скачивает библиотеки для видеокарты (около 5 ГБ); работать на процессоре это не мешает.
  Скачать меньше можно, если сначала поставить версию только для процессора:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
* Обучение шло на Kaggle (GPU Tesla T4) со scikit-learn 1.8 или новее. Точные версии остальных
  библиотек печатает первая ячейка `kaggle_train.ipynb`.
* Остальные скрипты проверены на процессоре с torch 2.15.0, torchvision 0.29.2, scikit-learn 1.9.1,
  numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.2.
