# Mediascope Hackathon Solution

Решение для кейса Mediascope на AI Business SPB Hackathon. Проект содержит Python-модель, которая принимает датасет с поисковыми запросами и возвращает тот же датасет с предсказанной разметкой по колонкам `TypeQuery`, `Title` и `ContentType`.

## Что делает решение

Основная точка входа находится в `solution.py`:

- класс `PredictionModel`
- публичный метод `predict(df: pd.DataFrame) -> pd.DataFrame`

Ожидаемый вход:

- `pd.DataFrame`
- обязательная колонка `QueryText`

Ожидаемый выход:

- `QueryText`
- `TypeQuery`
- `Title`
- `ContentType`

## Архитектура

Решение построено как каскадный пайплайн:

1. Классификация `TypeQuery`
2. Поиск `Title` через словарь франшиз, retrieval и embedding-based matching
3. Предсказание и калибровка `ContentType`
4. Агрегация сигналов в итоговый ответ

Ключевые файлы:

- `solution.py` — инференс и класс `PredictionModel`
- `train.py` — обучение и генерация артефактов
- `ct_classifier.py` — классификатор `ContentType`
- `title_retrieval.py` — retrieval для названий
- `franchise_dict.py` — словарь франшиз и матчинг
- `embeddings.py` — embedding / TF-IDF индекс
- `knowledge_graph.py` — граф знаний
- `aggregator.py` — объединение сигналов
- `scripts/submit.py` — сборка и отправка `bundle.zip`
- `scripts/eval_hidden_like.py` — локальная hidden-like проверка

## Требования

- Python `>= 3.11`

Основные библиотеки:

- `pandas`
- `numpy`
- `scikit-learn`
- `rapidfuzz`
- `pymorphy3`
- `pymorphy3-dicts-ru`
- `scipy`
- `requests`
- `python-dotenv`

## Установка

Вариант с `uv`:

```bash
uv sync
```

Вариант с `venv` и `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные

Если `train.csv` уже лежит в корне проекта, можно сразу запускать обучение.

Если нужно скачать тренировочные данные с сервера:

```bash
export API_KEY=your_key
python3 scripts/download_data.py
```

Скрипт положит данные в директорию `data/`.

## Обучение артефактов

Для генерации всех артефактов, которые использует `PredictionModel`, запусти:

```bash
python3 train.py
```

После успешного обучения в директории `artifacts/` появятся, в частности:

- `typequery_model.pkl`
- `ct_classifier.pkl`
- `embeddings.pkl`
- `franchise_dict.json`
- `knowledge_graph.json`
- `metadata.json`

## Использование модели

Минимальный пример:

```python
import pandas as pd

from solution import PredictionModel

df = pd.DataFrame(
    {
        "QueryText": [
            "гарри поттер смотреть",
            "скачать музыку бесплатно",
        ]
    }
)

model = PredictionModel()
pred = model.predict(df)
print(pred)
```

Ожидаемый результат — `DataFrame` с колонками:

- `QueryText`
- `TypeQuery`
- `Title`
- `ContentType`

## Локальная проверка

Для быстрой hidden-like оценки можно использовать:

```bash
python3 scripts/eval_hidden_like.py --split title_group_holdout --mode fast_proxy --validation-fraction 0.2 --max-valid-rows 200 --seed 42
```

Это полезно для локальной проверки того, как решение обобщается на unseen-title сценарии.

## Сборка архива

Для сборки сдаваемого архива:

```bash
python3 -c "from scripts.submit import build_bundle; build_bundle()"
```

Скрипт:

- проверит наличие обязательных артефактов
- провалится раньше времени, если `ct_classifier.pkl` устарел или несовместим
- соберёт `bundle.zip` в корне проекта

## Отправка решения

Если нужно отправить архив через API:

```bash
export API_KEY=your_key
python3 scripts/submit.py
```

Скрипт:

- соберёт `bundle.zip`
- отправит его на сервер
- будет опрашивать статус сабмита до завершения

## Примечания

- Submission-сборка по умолчанию использует режим без лемматизации, чтобы train и inference совпадали с окружением sandbox.
- Не включай `MEDIASCOPE_ENABLE_LEMMATIZATION=1` для leaderboard-сабмита, если не контролируешь то же окружение на стороне оценки.
- Перед финальной сборкой архива рекомендуется заново выполнить `python3 train.py`, чтобы артефакты точно соответствовали текущему коду.
- Основной сдаваемый интерфейс — это `PredictionModel` из `solution.py`.
- Проект ориентирован на офлайн-инференс по батчам входных запросов.
