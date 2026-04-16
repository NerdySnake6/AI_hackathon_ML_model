# Архитектура ML-системы разметки поисковых запросов

## Цель

Система размечает пользовательские поисковые запросы и определяет:

- относится ли запрос к профессиональному видеоконтенту;
- какой тип контента ищет пользователь: фильм, сериал, мультфильм, мультсериал или общий видеозапрос;
- какое конкретное название ищет пользователь, если оно есть в каталоге.

Пример результата:

```json
{
  "query": "1 сезон тьмы",
  "is_prof_video": true,
  "content_type": "series",
  "title": "Тьма",
  "title_id": "title_001",
  "confidence": 0.91,
  "decision": "review"
}
```

## Ключевая идея

Это не одна модель, которая угадывает название из воздуха. Это гибридная система:

```text
raw query
  -> validation
  -> typo-tolerant normalization
  -> query understanding: video / non-video / uncertain / generic
  -> title mention extraction
  -> hybrid retrieval: lexical + vector candidate search
  -> candidate ranking
  -> type resolving
  -> confidence policy
  -> auto_accept / review / manual_required / non_video / generic_video
```

Каталог тайтлов остается обязательным компонентом, но он больше не является единственным источником истины для определения `video/non_video`. Каталог нужен для привязки к локальному `title_id`, а понимание того, похож ли запрос на видеоконтент, должно происходить до финального entity linking.

## Идеальный пайплайн

Для production-версии правильная логика выглядит так:

```text
raw query
  -> validation
  -> normalization and typo recovery
  -> query understanding model
       - is_prof_video
       - is_generic
       - content_type
       - title_mention
       - confidence
  -> hybrid retrieval
       - lexical retrieval: aliases, fuzzy, BM25, trigram
       - vector retrieval: embeddings, nearest neighbors
  -> reranker
       - lexical score
       - vector score
       - metadata score
       - type hints
       - year match
  -> decision policy
       - auto_accept
       - review
       - manual_required
       - needs_catalog_enrichment
       - non_video
```

Ключевой принцип: отсутствие тайтла в каталоге не должно автоматически переводить запрос в `non_video`. Если `query understanding` считает, что это похоже на название видеоконтента, но retrieval не нашел надежного `title_id`, запрос должен попадать в `manual_required` или `needs_catalog_enrichment`.

## Основные компоненты

### 1. Validation

Входные запросы проходят базовую проверку:

- строка не пустая;
- длина не превышает лимит;
- batch-запрос не превышает лимит;
- лишние пробелы обрезаются.

Начальные лимиты:

```text
MAX_QUERY_LENGTH = 300
MAX_BATCH_SIZE = 500
MAX_CANDIDATES = 5
```

### 2. Typo-tolerant preprocessing

Пользователи часто пишут с ошибками, промахиваются по клавишам, смешивают раскладки и используют транслит. Поэтому система хранит несколько вариантов запроса:

- нормализованный запрос;
- очищенный запрос для поиска тайтла;
- compact-вариант без пробелов;
- вариант после исправления клавиатурной раскладки;
- грубые падежные варианты;
- исходный латинский вариант для поиска по translit-алиасам.

Примеры:

```text
"Тьма\t1 сезон"          -> "тьма 1 сезон"
"интерстелар онлайн"     -> "интерстелар"
"bynthcntkkfh"           -> "интерстеллар"
"машаи медведь"          -> "машаимедведь" как compact-вариант
"slovo pacana"           -> поиск по translit-алиасу каталога
```

### 3. Catalog

Каталог хранится локально и может быть загружен из CSV или БД. В production он может синхронизироваться из внешнего источника, но inference не должен зависеть от чужого API в реальном времени.

Минимальная структура:

```text
catalog_titles
- id
- canonical_title
- content_type
- year
- popularity
- external_source
- external_id

title_aliases
- id
- title_id
- alias
- normalized_alias
```

Архитектурно источник каталога прячется за repository-интерфейсом:

```text
CatalogRepository
  -> CsvCatalogRepository
  -> PostgresCatalogRepository
  -> ExternalApiCatalogRepository, optional
```

### 4. Domain gate

Перед определением типа контента система должна понять, относится ли запрос к профессиональному видеоконтенту. Это защищает от мусорных запросов:

```text
"10 троллейбус ижевск"       -> non_video
"купить мультиметр"          -> non_video
"кухня ремонт"               -> non_video
"кухня 3 сезон"              -> prof_video
"смотреть фильмы 2025"       -> prof_video, generic
"тьма 1 сезон"               -> prof_video, specific
```

В MVP используется rule-based gate:

- видео-интенты: `смотреть`, `онлайн`, `фильм`, `сериал`, `сезон`, `серия`;
- hard negative слова: `купить`, `ремонт`, `маршрут`, `расписание`, `погода`, `рецепт`;
- сигнал из каталога: найден похожий тайтл;
- confidence threshold.

Важный open-world случай: отсутствие тайтла в каталоге не равно `non_video`. Короткие запросы, похожие на название, но не найденные в каталоге, должны уходить в `manual_required` как `unknown_title_candidate`. Например:

```text
"пылающий берег" -> manual_required, possible out-of-catalog title
```

Для точного определения типа такого запроса нужны либо расширенный каталог, либо обученная ML-модель по исторической разметке, либо внешний поисковый/справочный источник, синхронизированный в локальный каталог.

Позже этот блок заменяется или усиливается ML-классификатором: `char n-gram TF-IDF + LogisticRegression`, `CatBoost` или transformer.

### 5. Generic/specific detector

Система отличает запросы конкретного тайтла от общих запросов:

```text
"смотреть фильмы 2025"       -> generic_video, title_id = null
"лучшие сериалы про любовь"  -> generic_video, title_id = null
"тьма 1 сезон"               -> specific, title_id = title_001
```

Если запрос generic, поиск конкретного `title_id` не навязывается.

### 6. Candidate search

Поиск кандидатов должен быть гибридным:

- точное совпадение по нормализованным алиасам;
- сравнение compact-вариантов без пробелов;
- fuzzy-сравнение через `difflib` в MVP;
- `RapidFuzz` как рекомендуемое усиление;
- translit-алиасы каталога;
- исправление неправильной раскладки;
- dense embeddings для vector retrieval.

Dense retrieval не заменяет текстовые поля каталога. Правильная схема такая:

```text
catalog title
  -> canonical_title
  -> aliases
  -> metadata
  -> embedding

user query
  -> normalization
  -> title_mention extraction
  -> encoder
  -> nearest-neighbor search
```

Для большого каталога этот блок можно развернуть так:

- lexical index: OpenSearch / Elasticsearch / PostgreSQL trigram;
- vector index: `pgvector` в PostgreSQL или отдельный ANN-индекс;
- reranker поверх top-k кандидатов.

Важно: векторный поиск не требует decoder. Нужен encoder, который переводит названия и запросы в общее векторное пространство. Обратное восстановление текста из вектора здесь не является основной задачей.

### 7. Candidate ranker

Ranker повышает или понижает score кандидата:

- `сезон` или `серия` усиливает сериалы;
- `мультфильм`, `мультик`, `мультсериал` усиливает анимационный контент;
- год в запросе усиливает совпадение по году;
- hard negative слова снижают уверенность;
- точное совпадение алиаса повышает уверенность.

### 8. Decision policy

Система не должна слепо принимать все ответы. Итоговое решение зависит от confidence:

```text
domain = non_video
  -> non_video

generic = true
  -> generic_video

title_confidence >= AUTO_ACCEPT_THRESHOLD
  -> auto_accept

REVIEW_THRESHOLD <= title_confidence < AUTO_ACCEPT_THRESHOLD
  -> review

title_confidence < REVIEW_THRESHOLD
  -> manual_required
```

Начальные пороги:

```text
AUTO_ACCEPT_THRESHOLD = 0.92
REVIEW_THRESHOLD = 0.65
```

### 9. Human-in-the-loop

Запросы с неуверенным результатом попадают в очередь проверки. Оператор видит:

- исходный запрос;
- предсказание модели;
- confidence;
- top-N кандидатов;
- причины решения;
- быстрые действия: подтвердить, исправить, выбрать другой тайтл, отметить как non-video.

Исправления операторов становятся новыми обучающими примерами.

### 10. DB и миграции

Для серьезного проекта закладывается SQLAlchemy + Alembic. MVP может работать с CSV или локальным SQLite, но основная production-БД для этого проекта — PostgreSQL.

Почему PostgreSQL:

- данные имеют явные связи: запросы, тайтлы, алиасы, предсказания, кандидаты, очередь проверки;
- нужны транзакции и воспроизводимая история операторских исправлений;
- удобно поддерживать миграции через Alembic;
- SQL-запросы хорошо подходят для аудита качества и выгрузок датасетов;
- `pgvector` позволяет хранить embeddings рядом с каталогом и prediction metadata.

ClickHouse лучше рассматривать как дополнительный аналитический слой для большого потока событий:

- агрегировать частоты запросов;
- смотреть drift и долю low-confidence запросов;
- строить отчеты по auto_accept/review/manual.

MongoDB здесь не является первым выбором: структура данных уже достаточно табличная, а связи и миграции важнее гибкости документов.

Подключение к PostgreSQL задается через `QUERY_LABELER_DATABASE_URL`:

```text
postgresql+psycopg://user:password@host:5432/query_labeler
```

Локальная разработка может идти на SQLite:

```text
sqlite:///outputs/query_labeler.db
```

Минимальные таблицы:

- `search_queries`;
- `catalog_titles`;
- `title_aliases`;
- `title_embeddings`;
- `predictions`;
- `prediction_candidates`;
- `review_queue`;
- `model_versions`.

Минимальная схема для hybrid retrieval:

```text
catalog_titles
- id
- canonical_title
- content_type
- year
- popularity

title_aliases
- id
- title_id
- alias
- normalized_alias

title_embeddings
- id
- title_id
- alias_id, nullable
- encoder_name
- embedding vector
- created_at
```

Практически это значит следующее:

- текст и алиасы в БД хранить все равно нужно;
- embeddings хранятся как дополнительный индекс поиска;
- retrieval берет top-k из lexical search и top-k из vector search;
- потом объединенный список кандидатов идет в reranker.

### 11. API и batch-режим

Обязательные режимы:

```text
GET /
POST /label
POST /label_batch
GET /health

python scripts/run_batch.py --input data/sample_queries.csv --output outputs/predictions.csv
```

Batch-режим нужен для хакатона и offline-метрик. API нужен для браузерного демо и будущей интеграции. В MVP интерфейс встроен в FastAPI, чтобы не зависеть от дополнительных UI-фреймворков. Streamlit можно добавить позже как альтернативную оболочку.

### 12. MVP-стек

```text
Python
Pydantic
SQLAlchemy
Alembic
FastAPI
difflib baseline
RapidFuzz optional
CSV input/output
```

### 13. Текущий каркас

В текущем репозитории уже есть базовый offline-каркас hybrid retrieval:

- легкий локальный hashing encoder;
- precomputed alias embeddings;
- объединение lexical search и vector similarity;
- текущий pipeline уже умеет подмешивать vector signal до ranker.

Это не production-quality semantic search, но это правильная архитектурная ступень к `pgvector` и обучаемому encoder.

### 14. Production-расширение

```text
Postgres / ClickHouse
OpenSearch / pgvector
обучаемый ranker
ML classifier для domain gate
операторский интерфейс
мониторинг качества
регулярное переобучение
MLflow / DVC
n8n / Airflow / Prefect как optional orchestration layer
```
