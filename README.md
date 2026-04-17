# Mediascope — AI Business SPB Hackathon 2026 — Solution

## Архитектура решения

Каскадно-параллельный пайплайн с последовательной фильтрацией и тремя ветками детекции:

```
QueryText → Preprocessing → TypeQuery Classifier
                                │
                          TypeQuery=1? ──No──→ TypeQuery=0, Title="", ContentType="прочее"
                                │ Yes
                    ┌───────────┼───────────┐
                    │           │           │
            Franchise Dict   Embeddings   Knowledge
            (fuzzy match)  (TF-IDF +     Graph
                           cosine sim)   (co-occurrence)
                    │           │           │
                    └───────────┼───────────┘
                                │
                          Meta-Aggregator
                                │
                    Title, ContentType, Confidence
```

## Компоненты

| Файл | Описание |
|------|----------|
| `solution.py` | Точка входа — класс `PredictionModel` |
| `preprocessing.py` | Нормализация, фичи, маркеры видео/не-видео |
| `typequery_classifier.py` | HistGradientBoosting для TypeQuery (F2-optimized) |
| `franchise_dict.py` | Словарь франшиз + fuzzy matching (rapidfuzz) |
| `embeddings.py` | TF-IDF + prototype vectors per franchise |
| `knowledge_graph.py` | Граф совместных упоминаний title↔токены |
| `aggregator.py` | Мета-агрегатор с кластеризацией и voting |
| `train.py` | Скрипт обучения — генерирует `artifacts/` |

## Артефакты

| Файл | Описание | Размер |
|------|----------|--------|
| `artifacts/typequery_model.pkl` | Классификатор TypeQuery | ~2 MB |
| `artifacts/franchise_dict.json` | Словарь ~4800 франшиз | ~1.4 MB |
| `artifacts/embeddings.pkl` | TF-IDF + прототипы | ~63 MB |
| `artifacts/knowledge_graph.json` | Граф знаний | ~10 MB |

## Установка и обучение

```bash
# Установка зависимостей
pip install pandas numpy scikit-learn rapidfuzz scipy requests python-dotenv

# Скачивание данных (нужен API_KEY)
cp .env.example .env
# вписать API_KEY
python scripts/download_data.py

# Обучение моделей
python train.py

# Тестирование
python solution.py
```

## Отправка

```bash
# Через скрипт (нужен API_KEY)
python scripts/submit.py

# Или загрузить bundle.zip на app.ai-business-spb.ru
```

## Метрики (на train.csv sample, n=200)

| Метрика | Значение |
|---------|----------|
| TypeQuery F2 | 0.97 |
| ContentType macro F1 | 0.77 |
| Title token F1 | 0.85 |
| **Combined** | **0.87** |
