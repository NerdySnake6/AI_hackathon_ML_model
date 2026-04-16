# ML-модель для разметки поисковых запросов

MVP-проект для разметки поисковых запросов пользователей по профессиональному видеоконтенту.

Система определяет:

- относится ли запрос к профессиональному видеоконтенту;
- является ли запрос мусорным или общим;
- тип контента: фильм, сериал, мультфильм, мультсериал;
- конкретный тайтл из каталога, если его можно найти.

Подробная архитектура описана в [docs/architecture.md](docs/architecture.md).

Сейчас retrieval в проекте гибридный (на базе `numpy` и `rapidfuzz`): сначала строятся lexical-кандидаты по алиасам, потом к ним добавляется vector signal. Кроме того, проект включает **двухкомпонентную систему фильтрации (ML + Rules)** для отсева нецелевого мусора (Domain Classification) на базе scikit-learn.

Архитектура системы полностью контейнеризована и готова к Production-развертыванию через Docker Compose с базой данных PostgreSQL.

## Быстрый запуск batch-режима

```bash
python3 scripts/run_batch.py \
  --input data/sample_queries.csv \
  --catalog data/sample_catalog.csv \
  --output outputs/predictions.csv
```

## Оценка качества

```bash
python3 scripts/evaluate.py \
  --input data/sample_labeled_queries.csv \
  --catalog data/sample_catalog.csv \
  --output outputs/metrics_report.json \
  --errors outputs/errors.csv
```

Скрипт считает:

- `video precision / recall / f1`;
- `content_type_accuracy`;
- `title_id_accuracy`;
- `end_to_end_accuracy`;
- `auto_accept coverage`;
- `auto_accept precision`.

Тесты запускаются без дополнительных тестовых зависимостей:

```bash
python3 -m unittest discover -s tests
```

## Запуск Production-среды и Базы Данных

Официальная архитектура работает на PostgreSQL и разворачивается одной командой с помощью Docker:

```bash
docker compose up -d --build
```

**Что происходит под капотом при старте контейнера:**
1. Инициализируется база данных PostgreSQL;
2. Накатываются миграции с помощью `alembic`;
3. В базу автоматически загружается подготовленный нами **отфильтрованный датасет из ~12,000 топовых фильмов** с русскими переводами (`data/imdb_catalog.csv`);
4. ML-классификатор обучается на **5,000 синтетических реалистичных запросах**, сгенерированных из этой базы (`data/imdb_labeled_queries.csv`);
5. Поднимается Uvicorn с FastAPI и веб-интерфейсом.

Весь процесс занимает около 15 секунд. После этого:
- **Веб-интерфейс**: `http://localhost:8000`
- **Документация API**: `http://localhost:8000/docs`

> **Важно для коллег:** Папка `data/raw/imdb/` (огромные архивы TSV) исключена из репозитория! Мы передаем уже обработанные легковесные файлы `imdb_catalog.csv` и `imdb_labeled_queries.csv`. Проект сразу готов к работе после `git clone`.

## Импорт IMDb

Теперь проект умеет импортировать IMDb bulk dumps напрямую в БД, без обязательного промежуточного CSV. Во время импорта IMDb-алиасы проходят quality filter: выкидываются `working/video/festival`-варианты, шумные `review/poster/...` атрибуты и слишком неоднозначные normalized aliases.

```bash
python3 scripts/import_imdb.py \
  --basics data/raw/imdb/title.basics.tsv.gz \
  --akas data/raw/imdb/title.akas.tsv.gz \
  --ratings data/raw/imdb/title.ratings.tsv.gz \
  --min-votes 500 \
  --max-titles 50000 \
  --max-titles-per-normalized-alias 1 \
  --import-db \
  --create-tables
```

CSV теперь опционален и нужен только как debug/export-снимок каталога:

```bash
python3 scripts/import_imdb.py \
  --basics data/raw/imdb/title.basics.tsv.gz \
  --akas data/raw/imdb/title.akas.tsv.gz \
  --ratings data/raw/imdb/title.ratings.tsv.gz \
  --output outputs/imdb_catalog.csv
```

Для первого smoke run лучше ограничить объем:

```bash
python3 scripts/import_imdb.py \
  --basics data/raw/imdb/title.basics.tsv.gz \
  --akas data/raw/imdb/title.akas.tsv.gz \
  --ratings data/raw/imdb/title.ratings.tsv.gz \
  --output outputs/imdb_catalog_sample.csv \
  --min-votes 500 \
  --max-titles 50000
```

Если нужен и CSV, и импорт в БД одновременно:

```bash
python3 scripts/import_imdb.py \
  --basics data/raw/imdb/title.basics.tsv.gz \
  --akas data/raw/imdb/title.akas.tsv.gz \
  --ratings data/raw/imdb/title.ratings.tsv.gz \
  --output outputs/imdb_catalog.csv \
  --import-db \
  --create-tables
```

После прямого импорта batch можно запускать уже от каталога в БД:

```bash
python3 scripts/run_batch.py \
  --input data/sample_queries.csv \
  --catalog-source db \
  --output outputs/predictions_from_db.csv
```

## Запуск API

```bash
uvicorn app.api:app --reload
```

Для запуска API от каталога в БД:

```bash
export QUERY_LABELER_CATALOG_SOURCE=db
uvicorn app.api:app --reload
```

После запуска браузерный интерфейс доступен по адресу:

```text
http://127.0.0.1:8000/
```

Файлы интерфейса лежат отдельно от Python-кода:

```text
app/static/index.html
app/static/styles.css
app/static/app.js
```

API:

```text
POST http://127.0.0.1:8000/label
```

Пример тела запроса:

```json
{
  "query_text": "1 сезон тьмы"
}
```

## CI/CD

В проект добавлены GitHub Actions workflows:

```text
.github/workflows/ci.yml
.github/workflows/cd.yml
```

CI запускается на `push` и `pull_request` в основные ветки и проверяет:

- установку зависимостей;
- компиляцию Python-файлов;
- `unittest`;
- offline evaluation;
- batch smoke test;
- генерацию Alembic SQL;
- импорт каталога в БД.

CD запускается вручную или по тегам `v*`: собирает Docker-образ, поднимает API-контейнер, проверяет `/health` и `/label`, затем сохраняет Docker image как artifact. Реального деплоя на сервер пока нет, потому что целевой инфраструктуры еще нет.

Локальная сборка Docker:

```bash
docker build -t query-labeler:local .
docker run --rm -p 8000:8000 query-labeler:local
```

## Текущий MVP

Сейчас реализован первый end-to-end каркас:

- Pydantic-схемы;
- нормализация запроса;
- обработка табуляции, переносов строк и лишних пробелов;
- генерация вариантов запроса;
- исправление неправильной клавиатурной раскладки;
- translit-алиасы каталога;
- fuzzy-поиск по каталогу через высокоскоростной C-движок `rapidfuzz`;
- lightweight hybrid retrieval: lexical + vector candidate search с матричными оптимизациями `numpy`;
- гибридный domain gate (ML scikit-learn + Rules) для интеллектуального отсечения мусора;
- generic/specific detector;
- candidate ranker;
- confidence policy;
- batch-скрипт и FastAPI endpoint;
- Production-ready Docker Compose и PostgreSQL.

## Следующие шаги (Задачи на завтрашний день хакатона)

1. **Подключить боевые данные или расширить датасет**: Обогатить текущие 12,000 популярных IMDB-фильмов специфичным локальным контентом (блогеры, передачи, локальные проекты), как только они станут доступны от заказчика.
2. **Переход на Deep Learning / Embeddings**: Заменить текущий быстрый TF-IDF (Scikit-Learn Logistic Regression) на нейронные эмбеддинги (например, `rubert-tiny2` через `SentenceTransformers`). Это позволит модели понимать смысл сленга или опечаток ("кинчик", "глянуть"), не опираясь строго на токены на уровне букв.
3. **Перенести Vector Retrieval в PostgreSQL**: Подключить `pgvector` в нашу текущую БД, чтобы поиск `TitleCandidate` происходил непосредственно средствами СУБД, снимая нагрузку с оперативной памяти Python.
4. **Аналитика и Дашборды**: Развернуть легкую витрину (например, Grafana или Streamlit) для демонстрации распределения трафика (Сколько мусора отсек ML? Какие жанры ищут чаще?).
