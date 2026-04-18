"""
Training script: build all model artifacts from train.csv.
Run with: python train.py
"""
import sys
import os
import json

import pandas as pd
from sklearn import __version__ as sklearn_version

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from ct_classifier import (
    CT_ARTIFACT_FORMAT_VERSION,
    save_artifacts as save_ct,
    train_ct_classifier,
)
from franchise_dict import build_franchise_dict, save_artifacts as save_franchise
from embeddings import build_embedding_index, save_artifacts as save_embeddings
from knowledge_graph import build_knowledge_graph, save_artifacts as save_knowledge_graph
from preprocessing import (
    HAS_PYMORPHY,
    build_typo_dict,
    is_lemmatization_enabled,
    save_artifacts as save_preprocessing,
)
from supplemental_titles import load_kinopoisk_top250, load_enriched_dataset, merge_title_dicts
from typequery_classifier import train_typequery_classifier, save_artifacts as save_typequery


def main():
    """Train all artifacts used by the offline submission bundle."""
    print(
        "Preprocessing mode:",
        f"lemmatization_enabled={is_lemmatization_enabled()}",
        f"(pymorphy3_available={HAS_PYMORPHY})",
    )

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
    
    # Merge supplemental sources
    kp_titles = load_kinopoisk_top250('kinopoisk-top250.csv')
    if kp_titles:
        franchise_dict = merge_title_dicts(franchise_dict, kp_titles)

    enriched_titles = load_enriched_dataset('artifacts/enriched_films.csv')
    if enriched_titles:
        franchise_dict = merge_title_dicts(franchise_dict, enriched_titles)
        
    save_franchise(franchise_dict)
    
    if is_lemmatization_enabled():
        print("  Expanding dictionary with lemmatized variants...")
        from scripts.expand_variants import expand_variants

        expand_variants()
    else:
        print("  Lemmatization disabled; skipping variant expansion.")
    
    # Re-load the expanded dict for subsequent steps (like embeddings)
    from franchise_dict import load_artifacts as load_franchise
    artifacts = load_franchise('artifacts/franchise_dict.json')
    if 'franchise_dict' in artifacts:
        franchise_dict = artifacts['franchise_dict']
    else:
        franchise_dict = artifacts
    
    print(f"  {len(franchise_dict)} canonical titles (expanded)")

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
    vectorizer, prototypes, prototype_info = build_embedding_index(df, franchise_dict=franchise_dict)
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
    ct_model, ct_tfidf = train_ct_classifier(df, franchise_dict=franchise_dict)
    save_ct(ct_model, ct_tfidf)
    print("  Done")

    # Save metadata
    metadata = {
        'n_train': len(df),
        'n_titles': len(franchise_dict),
        'n_supplemental_kp': len(kp_titles) if 'kp_titles' in locals() else 0,
        'n_supplemental_enriched': len(enriched_titles) if 'enriched_titles' in locals() else 0,
        'lemmatization_enabled': is_lemmatization_enabled(),
        'pymorphy3_available_local': HAS_PYMORPHY,
        'typequery_threshold': float(threshold),
        'ct_model_class': type(ct_model).__name__ if ct_model is not None else None,
        'ct_artifact_format_version': CT_ARTIFACT_FORMAT_VERSION,
        'sklearn_version': sklearn_version,
    }
    with open('artifacts/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nAll artifacts saved to artifacts/")
    print("Ready to submit solution.py!")


if __name__ == '__main__':
    main()
