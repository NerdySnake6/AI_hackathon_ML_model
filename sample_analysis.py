"""Error analysis on sample of 200 queries (no train.csv needed for this)."""
import os
os.chdir(r'E:\h_t2')

import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, f1_score

# We'll generate a representative sample by re-reading from full_preds if available,
# or just run a fresh prediction on downloaded data
try:
    df = pd.read_csv('data/train.csv')
    print(f'Loaded {len(df)} rows from data/train.csv')
except FileNotFoundError:
    # Try downloading
    print('train.csv not found in data/')
    print('Run: python scripts/download_data.py (needs API_KEY in .env)')
    print('')
    print('For now, running analysis on a synthetic sample...')

    # Generate representative queries to test
    from solution import PredictionModel
    
    # Manually crafted test queries covering error patterns
    test_queries = [
        # False positive candidates (should be 0)
        ("купить iphone 15 pro цена", 0, "", ""),
        ("рецепт борща с фото", 0, "", ""),
        ("скачать взлом ватсап", 0, "", ""),
        ("погода москва завтра", 0, "", ""),
        ("расписание поездов казань", 0, "", ""),
        ("картинки котиков обои", 0, "", ""),
        
        # TypeQuery=1, known title
        ("рик и морти 7 сезон", 1, "рик и морти", "мультсериал"),
        ("смотреть чебурашка онлайн", 1, "чебурашка", "фильм"),
        ("зимородок 1 серия", 1, "зимородок", "сериал"),
        ("слово пацана кровь на асфальте", 1, "слово пацана кровь на асфальте", "сериал"),
        
        # TypeQuery=1, no specific title (generic)
        ("смотреть фильмы 2025", 1, "", "фильм"),
        ("новинки сериалов 2024", 1, "", "сериал"),
        ("мультфильмы для малышей", 1, "", "мультфильм"),
        
        # TypeQuery=1, new content (not in train)
        ("1 сезон тьмы", 1, "тьма", "сериал"),
        ("английский сериал slow horses", 1, "тихие воды", "сериал"),
        
        # Edge cases: typos/translit
        ("rik i morti", 1, "рик и морти", "мультсериал"),
        ("гарри потер и философский камень", 1, "гарри поттер", "фильм"),
    ]
    
    queries = [t[0] for t in test_queries]
    df = pd.DataFrame({'QueryText': queries, 'TypeQuery': [t[1] for t in test_queries],
                        'Title': [t[2] for t in test_queries], 'ContentType': [t[3] for t in test_queries]})
    print(f'Testing on {len(df)} representative queries')

model = PredictionModel()
preds = model.predict(df[['QueryText']])

print('\n========== PREDICTIONS ==========')
for i in range(len(df)):
    q = df.iloc[i]['QueryText'][:55]
    tq_t = df.iloc[i]['TypeQuery']
    tq_p = preds.iloc[i]['TypeQuery']
    t_t = str(df.iloc[i].get('Title', ''))
    t_p = str(preds.iloc[i]['Title'])
    ct_t = str(df.iloc[i].get('ContentType', ''))
    ct_p = preds.iloc[i]['ContentType']
    
    tq_ok = '✓' if tq_t == tq_p else '✗'
    t_ok = '✓' if t_t == t_p else '✗'
    ct_ok = '✓' if ct_t == ct_p else '✗'
    
    print(f'[{tq_ok}][{ct_ok}][{t_ok}] {q}')
    print(f'     TQ: {tq_t}→{tq_p}  CT: "{ct_t}"→"{ct_p}"  Title: "{t_t}"→"{t_p}"')
    print()
