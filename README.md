# ML-модель для разметки поисковых запросов

MVP-проект для разметки поисковых запросов пользователей по профессиональному видеоконтенту.

Система определяет:

- относится ли запрос к профессиональному видеоконтенту;
- является ли запрос мусорным или общим;
- тип контента: фильм, сериал, мультфильм, мультсериал;
- конкретный тайтл из каталога, если его можно найти.

Подробная архитектура описана в [docs/architecture.md](docs/architecture.md).

Сейчас retrieval в проекте уже гибридный: сначала строятся lexical-кандидаты по алиасам, потом к ним добавляется vector signal через легкий локальный encoder. Это промежуточный шаг между полностью rule-based MVP и production-схемой с PostgreSQL + `pgvector`.

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

## База данных

Для MVP по умолчанию используется SQLite:

```text
sqlite:///outputs/query_labeler.db
```

Для нормальной версии проекта основной выбор — PostgreSQL: там удобно хранить каталог, алиасы, запросы, предсказания, очередь проверки и версии модели. ClickHouse лучше добавить позже для аналитики по большому потоку событий, а MongoDB здесь не дает явного преимущества, потому что данные хорошо ложатся в реляционную схему.

Подключение к PostgreSQL задается переменной окружения:

```bash
export QUERY_LABELER_DATABASE_URL="postgresql+psycopg://user:password@localhost:5432/query_labeler"
```

Локальный импорт каталога в БД:

```bash
python3 scripts/import_catalog.py \
  --catalog data/sample_catalog.csv \
  --create-tables
```

Для production вместо `--create-tables` лучше использовать Alembic:

```bash
alembic upgrade head
```

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
- fuzzy-поиск по каталогу через стандартный `difflib`;
- lightweight hybrid retrieval: lexical + vector candidate search;
- rule-based domain gate для отсечения мусорных запросов;
- generic/specific detector;
- candidate ranker;
- confidence policy;
- batch-скрипт;
- FastAPI endpoint;
- браузерный demo UI;
- SQLAlchemy/Alembic-заготовка.

## Следующие шаги

1. Подключить реальные данные хакатона.
2. Перенести vector retrieval в PostgreSQL + `pgvector`.
3. Добавить `RapidFuzz` для более сильного fuzzy-поиска.
4. Обучить ML-классификатор `video / non-video`.
5. Расширить offline-метрики на реальных данных.
