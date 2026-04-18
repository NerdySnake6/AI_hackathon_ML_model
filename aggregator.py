"""
Meta-aggregator module: combines predictions from three detection branches
(fuzzy dictionary, embeddings, knowledge graph) and resolves conflicts.
"""
from collections import Counter
from rapidfuzz import fuzz


def aggregate_predictions(
    franchise_match: tuple,       # (canonical_title, matched_span, contentType, confidence, match_type)
    embedding_matches: list,       # [(title, contentType, confidence), ...]
    graph_matches: list,           # [(title, contentType, confidence), ...]
) -> dict:
    """
    Aggregate predictions from three branches.

    Returns dict with: title, contentType, confidence, agreement
    """
    candidates = []

    # Branch 1: Franchise dictionary (highest priority)
    fm_title, fm_span, fm_ct, fm_conf, fm_type = franchise_match
    if fm_title:
        candidates.append({
            'title': fm_title,
            'contentType': fm_ct,
            'confidence': fm_conf,
            'branch': 'franchise',
            'weight': 2.0 if fm_type == 'exact' else 1.8 if fm_type == 'lemma_match' else 1.6 if fm_type == 'lemma_overlap' else 1.2,
        })

    # Branch 2: Embedding matches
    for title, ct, conf in embedding_matches[:3]:
        candidates.append({
            'title': title,
            'contentType': ct,
            'confidence': conf,
            'branch': 'embedding',
            'weight': 1.0,
        })

    # Branch 3: Knowledge graph matches
    for title, ct, conf in graph_matches[:3]:
        candidates.append({
            'title': title,
            'contentType': ct,
            'confidence': conf,
            'branch': 'graph',
            'weight': 0.7,
        })

    if not candidates:
        return {
            'title': '',
            'contentType': '',
            'confidence': 0.0,
            'agreement': 0.0,
        }

    # Score each candidate: confidence * weight
    for c in candidates:
        c['score'] = c['confidence'] * c['weight']

    # Cluster candidates by title similarity
    title_clusters = []
    used = set()
    for i, c1 in enumerate(candidates):
        if i in used:
            continue
        cluster = [c1]
        used.add(i)
        for j, c2 in enumerate(candidates):
            if j in used:
                continue
            # Check if titles are similar
            sim = fuzz.ratio(c1['title'], c2['title'])
            if sim >= 70 or c1['title'] in c2['title'] or c2['title'] in c1['title']:
                cluster.append(c2)
                used.add(j)
        title_clusters.append(cluster)

    # Score each cluster
    best_cluster = None
    best_score = 0
    for cluster in title_clusters:
        total = sum(c['score'] for c in cluster)
        # Bonus for multi-branch agreement
        branches = set(c['branch'] for c in cluster)
        branch_bonus = len(branches) * 0.1
        total_with_bonus = total + branch_bonus
        if total_with_bonus > best_score:
            best_score = total_with_bonus
            best_cluster = cluster

    if not best_cluster:
        return {
            'title': '',
            'contentType': '',
            'confidence': 0.0,
            'agreement': 0.0,
        }

    # Only accept if best cluster has meaningful score
    if best_score < 0.15:
        return {
            'title': '',
            'contentType': '',
            'confidence': 0.0,
            'agreement': 0.0,
        }

    # Pick canonical title (highest-weighted match in best cluster)
    canonical = max(best_cluster, key=lambda c: c['score'])

    # Determine contentType by weighted vote
    ct_votes = Counter()
    for c in best_cluster:
        if c['contentType']:
            ct_votes[c['contentType']] += c['score']

    content_type = ct_votes.most_common(1)[0][0] if ct_votes else ''

    # Compute agreement: fraction of branches that agree
    branches_in_cluster = set(c['branch'] for c in best_cluster)
    total_branches = 3
    agreement = len(branches_in_cluster) / total_branches

    # Final confidence: weighted score normalized
    max_possible = sum(c['weight'] for c in candidates)
    final_confidence = min(best_score / max(max_possible, 1.0), 1.0)

    return {
        'title': canonical['title'],
        'contentType': content_type,
        'confidence': round(final_confidence, 4),
        'agreement': round(agreement, 4),
    }
