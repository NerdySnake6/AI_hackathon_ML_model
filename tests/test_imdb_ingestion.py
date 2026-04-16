"""Tests for IMDb bulk dump conversion into the local catalog schema."""

from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from app.ingestion.imdb import ImdbImportConfig, load_imdb_catalog


class ImdbIngestionTestCase(unittest.TestCase):
    """Validate the IMDb ingestion pipeline on compact synthetic fixtures."""

    def test_converts_selected_titles_and_aliases(self) -> None:
        """Keep only supported titles and collect useful aliases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            basics_path = Path(temp_dir) / "title.basics.tsv.gz"
            akas_path = Path(temp_dir) / "title.akas.tsv.gz"
            ratings_path = Path(temp_dir) / "title.ratings.tsv.gz"

            self.write_tsv_gz(
                basics_path,
                fieldnames=[
                    "tconst",
                    "titleType",
                    "primaryTitle",
                    "originalTitle",
                    "isAdult",
                    "startYear",
                    "genres",
                ],
                rows=[
                    {
                        "tconst": "tt001",
                        "titleType": "movie",
                        "primaryTitle": "Interstellar",
                        "originalTitle": "Interstellar",
                        "isAdult": "0",
                        "startYear": "2014",
                        "genres": "Adventure,Sci-Fi",
                    },
                    {
                        "tconst": "tt002",
                        "titleType": "tvSeries",
                        "primaryTitle": "Masha and the Bear",
                        "originalTitle": "Masha and the Bear",
                        "isAdult": "0",
                        "startYear": "2009",
                        "genres": "Animation,Comedy",
                    },
                    {
                        "tconst": "tt003",
                        "titleType": "movie",
                        "primaryTitle": "Nightfall",
                        "originalTitle": "Nightfall",
                        "isAdult": "0",
                        "startYear": "1956",
                        "genres": "Crime,Drama",
                    },
                    {
                        "tconst": "tt004",
                        "titleType": "short",
                        "primaryTitle": "Skip Me",
                        "originalTitle": "Skip Me",
                        "isAdult": "0",
                        "startYear": "2020",
                        "genres": "Comedy",
                    },
                ],
            )
            self.write_tsv_gz(
                akas_path,
                fieldnames=[
                    "titleId",
                    "ordering",
                    "title",
                    "region",
                    "language",
                    "types",
                    "attributes",
                    "isOriginalTitle",
                ],
                rows=[
                    {
                        "titleId": "tt001",
                        "ordering": "1",
                        "title": "Интерстеллар",
                        "region": "RU",
                        "language": "ru",
                        "types": "\\N",
                        "attributes": "\\N",
                        "isOriginalTitle": "0",
                    },
                    {
                        "titleId": "tt002",
                        "ordering": "1",
                        "title": "Маша и Медведь",
                        "region": "RU",
                        "language": "ru",
                        "types": "\\N",
                        "attributes": "\\N",
                        "isOriginalTitle": "0",
                    },
                    {
                        "titleId": "tt001",
                        "ordering": "2",
                        "title": "Ignored French Alias",
                        "region": "FR",
                        "language": "fr",
                        "types": "\\N",
                        "attributes": "\\N",
                        "isOriginalTitle": "0",
                    },
                    {
                        "titleId": "tt001",
                        "ordering": "3",
                        "title": "Interstellar Working Title",
                        "region": "US",
                        "language": "en",
                        "types": "working",
                        "attributes": "\\N",
                        "isOriginalTitle": "0",
                    },
                    {
                        "titleId": "tt002",
                        "ordering": "2",
                        "title": "Masha Review Title",
                        "region": "RU",
                        "language": "ru",
                        "types": "imdbDisplay",
                        "attributes": "review title",
                        "isOriginalTitle": "0",
                    },
                    {
                        "titleId": "tt001",
                        "ordering": "4",
                        "title": "Сумерки",
                        "region": "RU",
                        "language": "ru",
                        "types": "imdbDisplay",
                        "attributes": "\\N",
                        "isOriginalTitle": "0",
                    },
                    {
                        "titleId": "tt003",
                        "ordering": "1",
                        "title": "Сумерки",
                        "region": "RU",
                        "language": "ru",
                        "types": "imdbDisplay",
                        "attributes": "\\N",
                        "isOriginalTitle": "0",
                    },
                ],
            )
            self.write_tsv_gz(
                ratings_path,
                fieldnames=["tconst", "averageRating", "numVotes"],
                rows=[
                    {"tconst": "tt001", "averageRating": "8.7", "numVotes": "1000"},
                    {"tconst": "tt002", "averageRating": "8.1", "numVotes": "800"},
                    {"tconst": "tt003", "averageRating": "6.5", "numVotes": "700"},
                    {"tconst": "tt004", "averageRating": "6.5", "numVotes": "1500"},
                ],
            )

            catalog = load_imdb_catalog(
                basics_path=basics_path,
                akas_path=akas_path,
                ratings_path=ratings_path,
                config=ImdbImportConfig(min_votes=500),
            )

        by_id = {title.title_id: title for title in catalog}
        self.assertEqual(set(by_id), {"tt001", "tt002", "tt003"})
        self.assertEqual(by_id["tt001"].content_type.value, "film")
        self.assertEqual(by_id["tt002"].content_type.value, "animated_series")
        self.assertIn("Интерстеллар", by_id["tt001"].aliases)
        self.assertIn("Маша и Медведь", by_id["tt002"].aliases)
        self.assertNotIn("Ignored French Alias", by_id["tt001"].aliases)
        self.assertNotIn("Interstellar Working Title", by_id["tt001"].aliases)
        self.assertNotIn("Masha Review Title", by_id["tt002"].aliases)
        self.assertIn("Сумерки", by_id["tt001"].aliases)
        self.assertNotIn("Сумерки", by_id["tt003"].aliases)
        self.assertGreater(by_id["tt001"].popularity, by_id["tt002"].popularity)

    @staticmethod
    def write_tsv_gz(
        path: Path,
        fieldnames: list[str],
        rows: list[dict[str, str]],
    ) -> None:
        """Write a compact gzipped TSV fixture."""
        with gzip.open(path, "wt", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
