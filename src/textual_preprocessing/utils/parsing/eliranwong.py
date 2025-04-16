import pandas as pd
import re
from typing import Iterable, List, Optional, TypeVar, Dict
from utils.parsing._parse import Parser, Document

class LXXSweteParser(Parser):
    def __init__(self, combine: bool = False):
        """
        Initializes the parser with the option to combine verses.

        Parameters:
        ----------
        combine: bool
            If True, combine all "Ecc" verses into a single document with newline-separated text.
        """
        self.combine = combine

    def parse_file(self, files: Dict[str, str], combine: bool = False) -> Iterable[Document]:
        """
        Parses the LXX Swete files into documents.

        Parameters:
        ----------
        files: Dict[str, str]
            A dictionary containing paths for the two files:
            - "versification": path to 00-Swete_versification.csv
            - "text": path to 01-Swete_word_with_punctuations.csv
        combine: bool
            If True, combine all "Ecc" verses into a single document with newline-separated text.

        Yields:
        ----------
        Document objects for each "book" verse or a combined file for each "book".
        """
        # ensure combined
        combine = self.combine  

        # Step 1: Load and sort data for consistency
        versification = pd.read_csv(files["versification"], sep=r"\s+", header=None, names=["word_id", "verse"])
        text_data = pd.read_csv(files["text"], sep=r"\s+", header=None, names=["word_id", "word"])

        # Step 2: Extract book name from verse column
        versification['book'] = versification['verse'].str.extract(r"^([0-9]?[A-Za-z]+)")

        # Filter out rows where 'book' is NaN (invalid entries)
        versification = versification.dropna(subset=['book'])

        # Sort versification by word_id to ensure correct ordering
        versification = versification.sort_values("word_id").reset_index(drop=True)

        # Step 3: Assign each word to its corresponding verse
        verse_ranges = []
        num_verses = len(versification)
        for i in range(num_verses):
            current = versification.iloc[i]
            start_id = current["word_id"]
            verse = current["verse"]
            book = current["book"]

            if i + 1 < num_verses:
                end_id = versification.iloc[i + 1]["word_id"]
            else:
                end_id = text_data["word_id"].max() + 1  # ensure inclusion of final words

            word_ids = text_data[(text_data["word_id"] >= start_id) & (text_data["word_id"] < end_id)]["word_id"]
            verse_ranges.extend([(word_id, verse, book) for word_id in word_ids])

        verse_df = pd.DataFrame(verse_ranges, columns=["word_id", "verse", "book"])

        # Step 4: Merge with text data
        merged = pd.merge(text_data, verse_df, on="word_id", how="inner")

        # Step 5: Ensure verses are sorted properly by numeric order
        def parse_verse_id(verse: str) -> tuple[int, int]:
            # e.g., 'Ecc.1.1' -> (1, 1)
            parts = verse.split(".")
            try:
                chapter = int(parts[1])
                verse_num = int(parts[2])
            except (IndexError, ValueError):
                chapter = 0
                verse_num = 0
            return (chapter, verse_num)

        merged["chapter"], merged["verse_num"] = zip(*merged["verse"].map(parse_verse_id))

        # Step 6: Group by book
        grouped_books = merged.groupby("book")

        for book, book_data in grouped_books:
            # Sort by chapter, verse number, and word_id
            book_data = book_data.sort_values(["chapter", "verse_num", "word_id"])

            # Generate verse identifiers
            book_data["verse_key"] = book_data["book"] + "." + book_data["chapter"].astype(str) + "." + book_data["verse_num"].astype(str)

            # Group by full verse key and concatenate words
            grouped_verses = (
                book_data.groupby("verse_key")["word"]
                         .apply(lambda words: " ".join(map(str, words)))
                         .reset_index()
            )

            # Step 7: Combine or yield results for each book
            if combine:
                combined_text = "\n".join(grouped_verses.sort_values("verse_key")["word"])
                yield Document(
                    id=re.sub(r"\b0\.0\b", "", book),
                    title=re.sub(r"\b0\.0\b", "", book),
                    author="Unknown",
                    text=combined_text.strip("\n"),
                )
            else:
                for _, row in grouped_verses.iterrows():
                    yield Document(
                        id=row["verse_key"],
                        title=row["verse_key"],
                        author="Unknown",
                        text=row["word"].strip(),
                    )
