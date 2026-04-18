"""
Text preprocessing module: normalization, transliteration correction,
typo fixing, feature extraction for the TypeQuery classifier.

Lemmatization is intentionally disabled by default to match the evaluation
sandbox, where `pymorphy3` is unavailable. It can be re-enabled locally for
experiments via the `MEDIASCOPE_ENABLE_LEMMATIZATION=1` environment variable.
"""
import json
import os
import re
from collections import Counter

try:
    import pymorphy3

    _morph = pymorphy3.MorphAnalyzer()
    HAS_PYMORPHY = True
except ImportError:
    _morph = None
    HAS_PYMORPHY = False
except Exception:
    _morph = None
    HAS_PYMORPHY = False

_lemma_cache = {}
LEMMATIZATION_ENV_VAR = "MEDIASCOPE_ENABLE_LEMMATIZATION"
_LEMMATIZATION_ENABLED = (
    HAS_PYMORPHY
    and os.environ.get(LEMMATIZATION_ENV_VAR, "0").strip().lower() in {"1", "true", "yes", "on"}
)

# ---------- transliteration map ----------
# (defined below, after detect_translit)

# Common typo dictionary (built from training data, saved/loaded)
_typo_dict: dict = {}


def is_lemmatization_enabled() -> bool:
    """Return whether lemmatization is enabled for the current process."""
    return _LEMMATIZATION_ENABLED


def build_typo_dict(queries: list[str], canonical_titles: list[str], threshold: int = 2) -> dict:
    """Build a typo correction dictionary from frequent query→title mappings."""
    from rapidfuzz import process, fuzz
    typos = {}
    # Count query patterns that map to a known title
    q_counts: Counter = Counter()
    for q in queries:
        q_norm = normalize(q)
        if len(q_norm) > 3:
            q_counts[q_norm] += 1

    # For common misspellings, find nearest canonical title
    for q_norm, cnt in q_counts.items():
        if cnt < 2:
            continue
        # Check if it's close to a canonical title but not exact
        best = process.extractOne(q_norm, canonical_titles, scorer=fuzz.ratio, score_cutoff=70)
        if best:
            match_title, score, _ = best
            if q_norm.lower() != match_title.lower():
                typos[q_norm] = match_title
    return typos


def save_artifacts(typo_dict: dict, path: str = "artifacts/typo_dict.json"):
    """Persist typo-correction artifacts to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(typo_dict, f, ensure_ascii=False, indent=2)


def load_artifacts(path: str = "artifacts/typo_dict.json"):
    """Load typo-correction artifacts into the module cache."""
    global _typo_dict
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            _typo_dict = json.load(f)
    return _typo_dict


def normalize(text: str, use_lemmatization: bool = False) -> str:
    """Lower, strip special chars, collapse whitespace, and optionally lemmatize."""
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    # Keep Russian + Latin letters, digits, spaces
    t = re.sub(r'[^a-zа-яё0-9\s]', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    
    if use_lemmatization and is_lemmatization_enabled():
        words = t.split()
        lemmas = []
        for w in words:
            if w not in _lemma_cache:
                _lemma_cache[w] = _morph.parse(w)[0].normal_form
            lemmas.append(_lemma_cache[w])
        return " ".join(lemmas)
    
    return t


def detect_translit(text: str) -> bool:
    """Heuristic: significant fraction of Latin chars in otherwise Cyrillic context."""
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    return latin > 0 and cyrillic == 0  # fully latin


# Extended multi-char transliteration (order matters — longest first)
TRANSLIT_MULTI = {
    'shch': 'щ', 'ch': 'ч', 'sh': 'ш', 'yo': 'ё', 'zh': 'ж',
    'kh': 'х', 'ts': 'ц', 'yu': 'ю', 'ya': 'я', 'ie': 'е',
    'ii': 'ий', 'yj': 'ый', 'iy': 'ий',
}

TRANSLIT_SINGLE = {
    'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф',
    'g': 'г', 'h': 'х', 'i': 'и', 'j': 'й', 'k': 'к', 'l': 'л',
    'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р',
    's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
    'y': 'ы', 'z': 'з',
    'ё': 'ё',
}


def transliterate(text: str) -> str:
    """Convert Latin transliteration to Cyrillic.

    Handles multi-char sequences first (sh, ch, ya...), then single chars.
    """
    text = text.lower().strip()
    result = text

    # Pass 1: multi-char replacements
    for latin, cyr in sorted(TRANSLIT_MULTI.items(), key=lambda x: -len(x[0])):
        result = result.replace(latin, '<<' + cyr + '>>')

    # Pass 2: single char replacements (skip chars inside <<>> markers)
    chars = list(result)
    in_marker = False
    out = []
    i = 0
    while i < len(chars):
        if i + 1 < len(chars) and chars[i] == '<' and chars[i+1] == '<':
            # Find closing >>
            end = result.index('>>', i)
            out.append(result[i+2:end])
            i = end + 2
            continue
        ch = chars[i]
        if ch in TRANSLIT_SINGLE:
            out.append(TRANSLIT_SINGLE[ch])
        else:
            out.append(ch)
        i += 1

    return ''.join(out)


def fix_typos(text: str) -> str:
    """Correct common typos using pre-built dictionary."""
    if text in _typo_dict:
        return _typo_dict[text]
    return text


# ---------- Hand-crafted features ----------
# Markers of video-content intent
VIDEO_MARKERS = [
    'смотреть', 'скачать', 'онлайн', 'бесплатно', 'торрент', 'hd', 'full hd',
    '1080', '720', '480', '2160', 'качестве', 'фильм', 'сериал', 'аниме',
    'дорама', 'мультфильм', 'мультсериал', 'озвучк', 'сезон', 'серия',
    'серии', 'эпизод', 'часть', 'трейлер', 'kinopoisk', 'кинопоиск',
    'ivi', 'окко', 'more tv', 'moretv', 'wink', ' PREMIER', 'premier',
    'start', 'start.ru', '阿姆', 'lordfilm', 'лордфильм', 'lostfilm',
    'лостфильм', 'newfilm', 'anime', 'watch', 'online', 'stream',
]

# Markers that suggest non-video intent (aggressive list)
NON_VIDEO_MARKERS = [
    # Shopping / commerce
    'купить', 'цена', 'заказать', 'доставк', 'магазин', 'отзыв',
    'заказ', 'оптом', 'распродаж', 'скидк', 'акци', 'стоимость',
    'продаж', 'купл', 'продам',
    # Downloads / hacking / mobile
    'скачать apk', 'скачать мод', 'взлом', 'взломать', 'хак', 'cheat',
    'hack', 'apk', 'mod ', 'кряк', 'crack', 'keygen', 'serial',
    'активатор', 'лицензионный ключ', 'крякнут', 'ватсап', 'whatsapp',
    'телеграм', 'telegram', 'инстаграм', 'instagram', 'тикток', 'tiktok',
    # Food / recipes / health
    'рецепт', 'приготов', 'калорий', 'состав', 'блюдо', 'суп',
    'салат', 'выпечк', 'тесто', 'нарезк', 'борщ', 'котлет', 'плов',
    'мясо', 'куриц', 'рыб', 'торт', 'пирог', 'каш', 'варен', 'жарен',
    # Tech / weather / transport
    'погод', 'прогноз', 'температур', 'осадк',
    'расписани', 'автобус', 'троллейбус', 'маршрут', 'остановк',
    'транспорт', 'поезд', 'электричк', 'билет',
    # Images / media
    'картинк', 'фото', 'обои', 'скачать песн', 'скачать музык',
    'скачать трек', 'реферат', 'курсов', 'диплом', 'котик', 'котят',
    'щен', 'собак', 'животн',
    # Games
    'скачать игру', 'играть онлайн', ' игр', ' game', ' games',
    'прохожден', 'гайд', 'гайды', 'стрим', 'twitch',
    # Drivers / software
    'скачать драйвер', 'скачать обновлени', 'скачать программ',
    'установить программ', 'настройк', 'инструкци',
    # Other non-video
    'аптека', 'больница', 'врач', 'лекарств', 'симптом', 'лечени',
    'форум', 'отзывы', 'аналог', 'аналоги',
    'торрент клиент', 'торрент програм', 'торрент сайт',
    'промокод', 'бонус', 'регистраци', 'войти', 'авторизац',
    # Mobile phone / tech
    'iphone', 'samsung', 'xiaomi', 'huawei', 'смартфон',
    'драйвер', 'обновление windows', 'скачать windows',
    # Banking / finance
    'банк', 'кредит', 'ипотек', 'вклад', 'карт', 'оплат',
    # Education
    'урок', 'задач', 'домашн', 'школ', 'универ', 'учеб',
    # Real estate
    'квартир', 'аренд', 'снять', 'ипотек', 'недвиж',
]

# Season/episode patterns
SEASON_PATTERN = re.compile(r'(\d+)\s*(?:сезон|season|s\b)', re.IGNORECASE)
EPISODE_PATTERN = re.compile(r'(\d+)\s*(?:серия|эпизод|episode|eps\b)', re.IGNORECASE)
YEAR_PATTERN = re.compile(r'\b(19[5-9]\d|20[0-2]\d)\b')


def extract_features(text: str) -> dict:
    """Extract hand-crafted features from a query string."""
    norm = normalize(text)
    features = {}

    # Basic string stats
    features['length'] = len(text)
    features['word_count'] = len(norm.split())
    features['has_digits'] = int(bool(re.search(r'\d', text)))
    features['digit_count'] = len(re.findall(r'\d', text))

    # Video markers
    lower = norm.lower()
    features['video_marker_count'] = sum(1 for m in VIDEO_MARKERS if m in lower)
    features['has_video_marker'] = int(features['video_marker_count'] > 0)

    # Non-video markers
    features['non_video_marker_count'] = sum(1 for m in NON_VIDEO_MARKERS if m in lower)
    features['has_non_video_marker'] = int(features['non_video_marker_count'] > 0)

    # Season / episode info
    season_match = SEASON_PATTERN.search(norm)
    episode_match = EPISODE_PATTERN.search(norm)
    year_match = YEAR_PATTERN.search(norm)
    features['has_season'] = int(bool(season_match))
    features['has_episode'] = int(bool(episode_match))
    features['has_year'] = int(bool(year_match))
    features['season_number'] = int(season_match.group(1)) if season_match else 0
    features['episode_number'] = int(episode_match.group(1)) if episode_match else 0

    # Transliteration detection
    features['is_translit'] = int(detect_translit(norm))

    # Specific content-type hints
    features['hint_film'] = int(any(m in lower for m in ['фильм', 'фильма', 'картина', 'кино']))
    features['hint_series'] = int(any(m in lower for m in ['сериал', 'сериала', 'дорам', 'сезон', 'серия']))
    features['hint_anime'] = int(any(m in lower for m in ['аниме', 'онгоинг', 'манга', 'ранобэ']))
    features['hint_cartoon'] = int(any(m in lower for m in ['мульт', 'cartoon', 'анимацион']))

    return features


def features_to_vector(features: dict) -> list:
    """Convert feature dict to ordered list for classifier input."""
    keys = [
        'length', 'word_count', 'has_digits', 'digit_count',
        'video_marker_count', 'has_video_marker',
        'non_video_marker_count', 'has_non_video_marker',
        'has_season', 'has_episode', 'has_year',
        'season_number', 'episode_number',
        'is_translit',
        'hint_film', 'hint_series', 'hint_anime', 'hint_cartoon',
    ]
    return [float(features.get(k, 0)) for k in keys]


FEATURE_KEYS = [
    'length', 'word_count', 'has_digits', 'digit_count',
    'video_marker_count', 'has_video_marker',
    'non_video_marker_count', 'has_non_video_marker',
    'has_season', 'has_episode', 'has_year',
    'season_number', 'episode_number',
    'is_translit',
    'hint_film', 'hint_series', 'hint_anime', 'hint_cartoon',
]
