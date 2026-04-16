import csv
import gzip
import random
from pathlib import Path

RAW_DIR = Path("data/raw/imdb")
CATALOG_OUT = Path("data/imdb_catalog.csv")
LABELED_OUT = Path("data/imdb_labeled_queries.csv")

# Filter for titles with > 15,000 votes to capture ~20k of the most popular global titles
MIN_VOTES = 15000

def run():
    print("Pass 1: Extracting top popular tconsts from ratings...")
    top_tconsts = {}
    rating_path = RAW_DIR / "title.ratings.tsv.gz"
    if not rating_path.exists():
        print(f"Error: {rating_path} not found.")
        return

    with gzip.open(rating_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) # skip header
        for row in reader:
            if len(row) < 3: continue
            try:
                num_votes = int(row[2])
                if num_votes >= MIN_VOTES:
                    top_tconsts[row[0]] = float(row[1]) # map tconst -> averageRating
            except ValueError:
                pass

    print(f"  Found {len(top_tconsts)} top titles based on votes.")

    print("Pass 2: Extracting basic metadata for top titles...")
    catalog_data = {}
    valid_types = {
        "movie": "film",
        "tvMovie": "film",
        "tvSeries": "series",
        "tvMiniSeries": "series"
    }
    basics_path = RAW_DIR / "title.basics.tsv.gz"
    
    with gzip.open(basics_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            if len(row) < 6: continue
            tconst, title_type, primary_title, original_title, is_adult, start_year = row[:6]
            if tconst in top_tconsts and title_type in valid_types:
                catalog_data[tconst] = {
                    "id": tconst,
                    "title": primary_title,
                    "original_title": original_title,
                    "type": valid_types[title_type],
                    "year": start_year if start_year != "\\N" else ""
                }
    
    print(f"  Filtered down to {len(catalog_data)} valid movies/series.")

    print("Pass 3: Extracting Russian localized titles...")
    akas_path = RAW_DIR / "title.akas.tsv.gz"
    ru_count = 0
    with gzip.open(akas_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            if len(row) < 4: continue
            title_id, ordering, title, region = row[:4]
            if region == "RU" and title_id in catalog_data:
                catalog_data[title_id]["title"] = title
                ru_count += 1

    print(f"  Applied {ru_count} Russian localizations.")

    print("Pass 4: Writing imdb_catalog.csv...")
    with open(CATALOG_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title_id", "canonical_title", "content_type", "year", "popularity", "aliases", "external_source", "external_id"])
        for tconst, meta in catalog_data.items():
            writer.writerow([
                meta["id"],
                meta["title"],
                meta["type"],
                meta["year"],
                round(top_tconsts[tconst] / 10.0, 3),
                meta["original_title"],
                "imdb",
                meta["id"]
            ])
            
    print("Pass 5: Generating synthetic labeled queries for ML training...")
    queries = []
    titles = list(catalog_data.values())
    random.shuffle(titles)
    
    # Postive queries (3,000 cases)
    for i in range(3000):
        t = random.choice(titles)
        ru = t["title"]
        orig = t["original_title"]
        title_picked = orig if random.random() > 0.8 else ru
        
        if t["type"] == "series":
            templates = [
                f"{title_picked} смотреть онлайн",
                f"сериал {title_picked}",
                f"{title_picked} 1 сезон",
                f"{title_picked} все серии подряд"
            ]
        else:
            templates = [
                f"{title_picked} смотреть",
                f"{title_picked} онлайн в хорошем качестве",
                f"фильм {title_picked}",
                f"{title_picked} бесплатно",
                f"скачать {title_picked} 2024"
            ]
            
        queries.append([f"imdb_pos_{i}", random.choice(templates), "true", "prof_video", t["type"], "auto_accept", "synthetic"])

    # Negative queries (1,500 cases)
    neg_templates = [
        "купить билет на {}",
        "саундтрек из {}",
        "актеры фильма {}",
        "рецензия {}",
        "{} отзывы критиков",
        "карта кинотеатров",
        "расписание электричек москва",
        "как починить кран",
        "погода на завтра спб"
    ]
    for i in range(1500):
        t = random.choice(titles)
        title_picked = t["title"]
        temp = random.choice(neg_templates)
        query = temp.format(title_picked) if "{}" in temp else temp
        queries.append([f"imdb_neg_{i}", query, "false", "non_video", "", "non_video", "synthetic"])

    # Generic Queries (500 cases)
    generic_templates = [
        "лучшие фильмы {year} года",
        "смешные комедии",
        "боевики смотреть",
        "сериалы онлайн бесплатно",
        "тупейшие хорроры"
    ]
    for i in range(500):
        q = random.choice(generic_templates).format(year=random.randint(1990, 2024))
        queries.append([f"imdb_gen_{i}", q, "true", "prof_video", "generic", "generic_video", "synthetic"])

    # Shuffle the dataset so ML doesn't see all positives first
    random.shuffle(queries)

    with open(LABELED_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_text", "is_prof_video", "domain_label", "content_type", "decision", "data_source"])
        writer.writerows(queries)
        
    print(f"  Generated robust dataset of {len(queries)} samples.")
    print("SUCCESS: Full IMDB Knowledge Extracted!")

if __name__ == "__main__":
    run()
