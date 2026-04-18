"""Expand franchise variants with lemmatized forms when available."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing import normalize


def expand_variants():
    """Augment stored franchise variants with lemmatized normalized variants."""
    from franchise_dict import save_artifacts, load_artifacts

    # Handle both old (plain dict) and new (artifacts dict) formats
    data_loaded = load_artifacts('artifacts/franchise_dict.json')
    if 'franchise_dict' in data_loaded:
        d = data_loaded['franchise_dict']
    else:
        d = data_loaded

    print(f"Expanding variants for {len(d)} titles...")

    for title, data in d.items():
        variants = set(data.get('variants', []))
        new_variants = set()

        for v in variants:
            v_lemma = normalize(v, use_lemmatization=True)
            if v_lemma:
                new_variants.add(v_lemma)

        data['variants'] = list(variants | new_variants)

    # Use the new save_artifacts that pre-computes everything
    save_artifacts(d, 'artifacts/franchise_dict.json')

    print("Done! Dictionary expanded with lemmatized variants.")


if __name__ == "__main__":
    expand_variants()
