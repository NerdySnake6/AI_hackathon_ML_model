"""
Knowledge graph module: build a lightweight co-occurrence graph of
franchises, content types, and common query terms. Used as an additional
detection branch alongside fuzzy matching and embeddings.
"""
import os
import json
import re
from collections import defaultdict, Counter
from preprocessing import normalize


def build_knowledge_graph(df, top_n_titles: int = 500):
    """
    Build a co-occurrence graph from training data.
    Nodes: franchise titles + common query tokens
    Edges: weighted co-occurrence counts
    """
    labeled = df[df['Title'].notna()].copy()
    labeled['norm_title'] = labeled['Title'].str.strip().str.lower()

    # Select top titles by frequency
    title_counts = labeled['norm_title'].value_counts()
    top_titles = set(title_counts.head(top_n_titles).index)

    # Build node→edges mapping
    edges = defaultdict(lambda: defaultdict(int))
    node_types = {}  # 'title' or 'token'

    # Register title nodes
    for title in top_titles:
        node_types[title] = 'title'

    # Build co-occurrence from queries
    for _, row in labeled.iterrows():
        norm_title = row['norm_title']
        if norm_title not in top_titles:
            continue
        norm_query = normalize(row['QueryText'])
        tokens = set(norm_query.split())

        for token in tokens:
            if len(token) < 3:
                continue
            node_types[token] = 'token'
            edges[norm_title][token] += 1
            edges[token][norm_title] += 1

    # Build title→title edges from shared tokens
    title_to_tokens = defaultdict(set)
    for _, row in labeled.iterrows():
        norm_title = row['norm_title']
        if norm_title not in top_titles:
            continue
        norm_query = normalize(row['QueryText'])
        tokens = set(norm_query.split())
        title_to_tokens[norm_title] |= tokens

    titles_list = list(top_titles)
    for i in range(len(titles_list)):
        for j in range(i + 1, len(titles_list)):
            t1, t2 = titles_list[i], titles_list[j]
            shared = title_to_tokens[t1] & title_to_tokens[t2]
            if shared:
                weight = len(shared)
                edges[t1][t2] += weight
                edges[t2][t1] += weight

    return dict(edges), node_types


def save_artifacts(edges, node_types, content_type_map, path: str = "artifacts/knowledge_graph.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert nested defaultdict to regular dict
    edges_serializable = {k: dict(v) for k, v in edges.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'edges': edges_serializable,
            'node_types': node_types,
            'content_type_map': content_type_map,
        }, f, ensure_ascii=False, indent=2)


def load_artifacts(path: str = "artifacts/knowledge_graph.json"):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Rebuild nested dicts
        edges = {k: defaultdict(int, v) for k, v in data['edges'].items()}
        return edges, data['node_types'], data['content_type_map']
    return {}, {}, {}


class KnowledgeGraphMatcher:
    """Match queries using knowledge graph traversal."""

    def __init__(self, edges, node_types, content_type_map):
        self.edges = edges
        self.node_types = node_types
        self.content_type_map = content_type_map

    def match(self, query: str, max_depth: int = 2) -> list:
        """
        Find franchise titles connected to query tokens via graph edges.
        Returns list of (title, contentType, confidence).
        """
        from preprocessing import normalize
        norm = normalize(query)
        tokens = set(norm.split())

        # Score candidates by edge weights
        candidates = Counter()
        matched_tokens = set()

        for token in tokens:
            if token in self.edges:
                for neighbor, weight in self.edges[token].items():
                    if self.node_types.get(neighbor) == 'title':
                        candidates[neighbor] += weight
                        matched_tokens.add(token)

        # BFS expansion for connected titles
        if max_depth > 1:
            for token in tokens:
                if token in self.edges:
                    for neighbor, weight in self.edges[token].items():
                        if self.node_types.get(neighbor) == 'token' and neighbor not in tokens:
                            # Follow to titles
                            if neighbor in self.edges:
                                for title, w2 in self.edges[neighbor].items():
                                    if self.node_types.get(title) == 'title':
                                        candidates[title] += w2 * 0.5  # decay for 2-hop

        if not candidates:
            return []

        # Normalize scores
        max_score = max(candidates.values())
        results = []
        for title, score in candidates.most_common(5):
            conf = score / max_score * 0.7
            ct = self.content_type_map.get(title, '')
            results.append((title, ct, conf))

        return results
