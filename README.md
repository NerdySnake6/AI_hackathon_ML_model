# Классификатор поисковых медиазапросов

ML-пайплайн для классификации поисковых запросов по полям `TypeQuery`, `Title` и `ContentType`.

Этот репозиторий представляет собой портфолио-версию хакатонного проекта, выполненного по кейсу компании **Mediascope**. Решение обрабатывает шумные пользовательские запросы с опечатками, транслитерацией и неполным намерением и предсказывает:

- относится ли запрос к профессиональному видеоконтенту
- какое нормализованное название или франшиза упоминается в запросе
- какой тип контента соответствует запросу: `фильм`, `сериал`, `мультфильм`, `мультсериал` и другие категории

## Контекст проекта

Оригинальное решение разрабатывалось в команде из 3 человек в рамках хакатона.

Этот репозиторий отражает мой финальный вклад в проект: интеграцию всего пайплайна, доработку инференса, стабилизацию решения, подготовку артефактов модели и доведение проекта до итогового рабочего состояния.

Проект выполнялся в команде вместе с [snugforce-web](https://github.com/snugforce-web) и [tiver211](https://github.com/tiver211).

Часть ранних наработок и некоторых исходных идей опиралась на вклад моего сокомандника [snugforce-web](https://github.com/snugforce-web).

## Что делает проект

Основная точка входа находится в классе `PredictionModel` в [solution.py](/Users/nerdysnake6/Documents/%20ML_модель/solution.py).

Вход:

- `pandas.DataFrame`
- обязательная колонка: `QueryText`

Выход:

- `QueryText`
- `TypeQuery`
- `Title`
- `ContentType`

## Архитектура

Решение использует каскадный пайплайн:

1. классификация `TypeQuery`
2. поиск названия через словарь франшиз, retrieval и embedding-based matching
3. предсказание и калибровка `ContentType`
4. агрегация конкурирующих сигналов в финальный ответ

Ключевые файлы:

- [solution.py](/Users/nerdysnake6/Documents/%20ML_модель/solution.py): инференс и класс `PredictionModel`
- [train.py](/Users/nerdysnake6/Documents/%20ML_модель/train.py): обучение и генерация артефактов
- [typequery_classifier.py](/Users/nerdysnake6/Documents/%20ML_модель/typequery_classifier.py): модель `TypeQuery`
- [title_retrieval.py](/Users/nerdysnake6/Documents/%20ML_модель/title_retrieval.py): retrieval для названий
- [franchise_dict.py](/Users/nerdysnake6/Documents/%20ML_модель/franchise_dict.py): словарный матчинг
- [embeddings.py](/Users/nerdysnake6/Documents/%20ML_модель/embeddings.py): векторизация и индекс эмбеддингов
- [knowledge_graph.py](/Users/nerdysnake6/Documents/%20ML_модель/knowledge_graph.py): графовые признаки и матчинги
- [aggregator.py](/Users/nerdysnake6/Documents/%20ML_модель/aggregator.py): объединение сигналов
- [scripts/eval_hidden_like.py](/Users/nerdysnake6/Documents/%20ML_модель/scripts/eval_hidden_like.py): локальная hidden-like оценка

## Стек

- Python 3.11+
- pandas
- numpy
- scikit-learn
- rapidfuzz
- scipy
- pymorphy3

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

Для обучения ожидается локальный файл `train.csv` в корне проекта.

Исходный датасет не включен в эту портфолио-версию репозитория.

## Обучение артефактов

Чтобы собрать все артефакты, используемые инференс-пайплайном:

```bash
python3 train.py
```

После этого в директории `artifacts/` появятся, в частности:

- `typequery_model.pkl`
- `ct_classifier.pkl`
- `embeddings.pkl`
- `franchise_dict.json`
- `knowledge_graph.json`
- `metadata.json`

## Запуск инференса

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
predictions = model.predict(df)
print(predictions)
```

## Локальная оценка

Для быстрой hidden-like проверки:

```bash
python3 scripts/eval_hidden_like.py --split title_group_holdout --mode fast_proxy --validation-fraction 0.2 --max-valid-rows 200 --seed 42
```

## Примечания

- Репозиторий ориентирован на офлайн-инференс и локальную работу с артефактами.
- Сгенерированные артефакты и приватные датасеты намеренно исключены из репозитория.
- Перед пересборкой артефактов лучше заново запускать `python3 train.py` на актуальном коде.
