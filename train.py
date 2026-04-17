"""
Training script: build all model artifacts from train.csv.
Run with: python train.py
"""
import sys
import os
import json

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import build_typo_dict, save_artifacts as save_preprocessing
from franchise_dict import build_franchise_dict, save_artifacts as save_franchise
from typequery_classifier import train_typequery_classifier, save_artifacts as save_typequery
from embeddings import build_embedding_index, save_artifacts as save_embeddings
from knowledge_graph import build_knowledge_graph, save_artifacts as save_knowledge_graph
from ct_classifier import train_ct_classifier, save_artifacts as save_ct
from supplemental_titles import load_kinopoisk_top250, merge_title_dicts


def main():
    """Train all artifacts used by the offline submission bundle."""
    # Load training data
    print("Loading train.csv...")
    df = pd.read_csv('train.csv')
    print(f"  Shape: {df.shape}")
    print(f"  TypeQuery distribution:\n{df['TypeQuery'].value_counts().to_dict()}")

    # Create artifacts directory
    os.makedirs('artifacts', exist_ok=True)

    # 1. Build franchise dictionary
    print("\n[1/5] Building franchise dictionary...")
    franchise_dict = build_franchise_dict(df)
    supplemental_titles = load_kinopoisk_top250('kinopoisk-top250.csv')
    if supplemental_titles:
        franchise_dict = merge_title_dicts(franchise_dict, supplemental_titles)
    save_franchise(franchise_dict)
    print(f"  {len(franchise_dict)} canonical titles")

    # 2. Build typo dictionary
    print("\n[2/5] Building typo dictionary...")
    canonical_titles = list(franchise_dict.keys())
    queries = df['QueryText'].tolist()
    typo_dict = build_typo_dict(queries, canonical_titles)
    save_preprocessing(typo_dict)
    print(f"  {len(typo_dict)} typo corrections")

    # 3. Train TypeQuery classifier
    print("\n[3/5] Training TypeQuery classifier...")
    model, tfidf, threshold = train_typequery_classifier(df)
    save_typequery(model, tfidf, threshold)
    print("  Done")

    # 4. Build embedding index
    print("\n[4/5] Building embedding index...")
    vectorizer, prototypes, prototype_info = build_embedding_index(df)
    save_embeddings(vectorizer, prototypes, prototype_info)
    print(f"  {len(prototypes)} franchise prototypes")

    # 5. Build knowledge graph
    print("\n[5/6] Building knowledge graph...")
    edges, node_types = build_knowledge_graph(df)
    # Build content type map
    content_type_map = {}
    for title, data in franchise_dict.items():
        content_type_map[title] = data['contentType']
    save_knowledge_graph(edges, node_types, content_type_map)
    print(f"  {len(edges)} nodes")

    # 6. Train ContentType classifier
    print("\n[6/6] Training ContentType classifier...")
    ct_model, ct_tfidf = train_ct_classifier(df)
    save_ct(ct_model, ct_tfidf)
    print("  Done")

    # Save metadata
    metadata = {
        'n_train': len(df),
        'n_titles': len(franchise_dict),
        'n_supplemental_titles': len(supplemental_titles),
        'typequery_threshold': float(threshold),
    }
    with open('artifacts/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nAll artifacts saved to artifacts/")
    print("Ready to submit solution.py!")


if __name__ == '__main__':
    main()
