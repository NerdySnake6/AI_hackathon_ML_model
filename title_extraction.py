"""
Title extraction: when no franchise match is found, extract a candidate
title from the query by removing stop-words.
"""

from preprocessing import normalize

# Words to remove when extracting title candidates
TITLE_STOP_WORDS = {
    # Action verbs
    'смотреть', 'скачать', 'онлайн', 'бесплатно', 'загрузить', 'закачать',
    'посмотреть', 'загруз', 'посмотр',
    # Quality/format
    'hd', 'full', 'hdrip', 'bdrip', 'webrip', 'dvdrip', 'hdtv',
    '1080', '720', '480', '2160', '4k', '蓝光',
    'качестве', 'качество', 'хорошем', 'отличном', 'без рекламы',
    'русский', 'русская', 'русское', 'русско', 'оригинал', 'озвучка',
    'озвучк', 'переозвучка', 'перевод', 'субтитр', 'дубляж', 'линейка',
    # Time indicators
    'год', 'года', '2020', '2021', '2022', '2023', '2024', '2025', '2026',
    '2019', '2018', '2017', '2016', '2015',
    # Generic content descriptors
    'фильм', 'фильмы', 'кино', 'картина',
    'сериал', 'сериалы', 'дорам', 'аниме',
    'мультфильм', 'мультфильмы', 'мультсериал', 'мультсериалы',
    'мульт', 'анимацион', 'cartoon', 'аним',
    # Season/episode
    'сезон', 'серия', 'серии', 'эпизод', 'часть', 'части',
    'season', 'episode', 'eps',
    # Platforms
    'kinopoisk', 'кинопоиск', 'ivi', 'окко', 'more tv', 'moretv',
    'wink', 'premier', 'start', 'lordfilm', 'лордфильм',
    'lostfilm', 'лостфильм', 'anime', 'watch', 'online', 'stream',
    # Other stop words
    'все', 'весь', 'всё', 'полностью', 'подряд', 'без остановки',
    'торрент', '磁力', 'на', 'телефон', 'компьютер', 'андроид',
    'iphone', 'айфон', 'пк', 'ПК',
    'и', 'в', 'на', 'с', 'по', 'к', 'у', 'о', 'от', 'до', 'для',
    'а', 'но', 'или', 'же', 'бы', 'ли',
    'new', 'новый', 'новинка', 'новинки',
    'best', 'лучший', 'лучшие', 'топ',
    # Common words that appear in queries but aren't titles
    'сериалов', 'фильмов', 'мультфильмов', 'мультсериалов',
    'анимеш', 'анимешник', 'манга', 'ранобэ',
    'малышей', 'детей', 'детский', 'взрослых', 'подростк',
    'тьмы', 'тьма',  # common words that are also titles
    'для', 'про', 'о', 'как', 'где', 'когда', 'почему', 'зачем',
    'английский', 'американский', 'российский', 'турецкий', 'корейский',
    'китайский', 'японский', 'французский', 'немецкий',
    'субтитр', 'озвучк', 'перевод', 'дубляж', 'режиссёр',
    'актер', 'актёр', 'рол', 'главн',
    'все', 'весь', 'всё', 'кажд', 'люб',
    'скачать', 'загруз', 'бесплатн', 'без', 'смс', 'регистраци',
    'торрент', 'магнит', 'ссылк',
    # Additional common noise
    'очень', 'хороший', 'крутой', 'интересный', 'самый', 'какой', 'который',
    'типо', 'типа', 'вроде', 'просто', 'прям', 'вообще', 'чисто',
    'смотреть', 'посмотреть', 'глянуть', 'видео', 'ролик', 'плеер',
    'кинотеатр', 'сайт', 'страница', 'официальный', 'бесплатный',
    'hd', 'fullhd', 'qhd', 'uhd', '4k', '1080p', '720p',
}

TITLE_FUNCTION_WORDS = {'и', 'из', 'в', 'во', 'на', 'по', 'о', 'с', 'к', 'у'}


def _is_year_token(token: str) -> bool:
    """Return True when the token looks like a release year."""
    return len(token) == 4 and token.isdigit() and 1950 <= int(token) <= 2035


def _trim_noise(words: list[str], query: str) -> list[str]:
    """Trim title-external noise while preserving meaningful inner tokens."""
    left = 0
    right = len(words) - 1

    while left <= right and words[left] in TITLE_STOP_WORDS:
        left += 1
    while right >= left and words[right] in TITLE_STOP_WORDS:
        right -= 1

    trimmed = words[left:right + 1]
    if not trimmed:
        return []

    while len(trimmed) > 1 and _is_year_token(trimmed[-1]):
        trimmed.pop()
    while len(trimmed) > 1 and _is_year_token(trimmed[0]):
        trimmed.pop(0)

    has_season_context = any(token in query for token in ('сезон', 'season', 'серия', 'эпизод', 'episode'))
    while len(trimmed) > 1 and has_season_context and trimmed[-1].isdigit():
        trimmed.pop()

    return trimmed


def extract_title_candidate(query: str) -> str:
    """
    Extract a potential title from a query by removing stop-words.
    Returns the remaining text or empty string.

    Examples:
        "белая королева сериал 2013 смотреть" → "белая королева"
        "рик и морти 7 сезон" → "рик морти"
        "смотреть фильмы 2025 онлайн" → "" (all stop words)
    """
    norm = normalize(query)
    words = norm.split()

    trimmed = _trim_noise(words, norm)
    if trimmed:
        kept = [w for w in trimmed if len(w) >= 2 or w in TITLE_FUNCTION_WORDS or w.isdigit()]
        meaningful = [w for w in kept if w not in TITLE_STOP_WORDS]
        if meaningful:
            return ' '.join(kept)

    # Fallback: aggressive filtering for generic queries
    kept = [
        w for w in words
        if w not in TITLE_STOP_WORDS and (len(w) >= 2 or w.isdigit())
    ]

    # Need at least 1 meaningful word
    if not kept:
        return ''

    return ' '.join(kept)
