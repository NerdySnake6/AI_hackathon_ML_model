"""Application settings and rule dictionaries for the query labeling MVP."""

from __future__ import annotations

MAX_QUERY_LENGTH = 300
MAX_BATCH_SIZE = 500
MAX_CANDIDATES = 5
VECTOR_EMBEDDING_DIMENSION = 256
SEARCH_TOKEN_MIN_LENGTH = 3
SEARCH_PREFIX_LENGTH = 4
SEARCH_MAX_ALIAS_CANDIDATES = 400

AUTO_ACCEPT_THRESHOLD = 0.92
REVIEW_THRESHOLD = 0.65
DOMAIN_VIDEO_THRESHOLD = 0.55
HYBRID_VECTOR_THRESHOLD = 0.42
HYBRID_LEXICAL_WEIGHT = 0.7
HYBRID_VECTOR_WEIGHT = 0.25

DEFAULT_CATALOG_PATH = "data/sample_catalog.csv"

VIDEO_INTENT_WORDS = {
    "смотреть",
    "посмотреть",
    "онлайн",
    "online",
    "бесплатно",
    "фильм",
    "фильмы",
    "кино",
    "сериал",
    "сериалы",
    "сезон",
    "серия",
    "серии",
    "трейлер",
}

GENERIC_VIDEO_WORDS = {
    "фильмы",
    "сериалы",
    "мультфильмы",
    "мультики",
    "комедии",
    "боевики",
    "ужасы",
    "мелодрамы",
    "аниме",
    "новинки",
    "лучшие",
    "подборка",
}

TITLE_NOISE_WORDS = VIDEO_INTENT_WORDS | {
    "hd",
    "fullhd",
    "1080",
    "720",
    "качество",
    "все",
    "новый",
    "новые",
    "русском",
    "языке",
    "озвучка",
}

HARD_NEGATIVE_WORDS = {
    "автобус",
    "троллейбус",
    "маршрут",
    "расписание",
    "остановка",
    "купить",
    "цена",
    "ремонт",
    "рецепт",
    "погода",
    "курс",
    "доллар",
    "вакансии",
    "работа",
    "адрес",
    "карта",
    "номер",
    "паспорт",
    "значение",
    "слова",
    "мультиметр",
}

FILM_WORDS = {"фильм", "фильмы", "кино"}
SERIES_WORDS = {"сериал", "сериалы", "сезон", "серия", "серии"}
CARTOON_WORDS = {"мультфильм", "мультфильмы", "мультик", "мультики"}
ANIMATED_SERIES_WORDS = {"мультсериал", "мультсериалы", "аниме"}
