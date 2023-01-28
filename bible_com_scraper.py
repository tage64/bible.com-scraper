#!/usr/bin/python3

import re
from textwrap import TextWrapper
import math
import os
import asyncio
from typing import *

import bs4
import httpx
import progressbar  # type: ignore

NUM_REGEX: re.Pattern = re.compile("[1-9][0-9]*")


class Book(NamedTuple):
    "Information about a bible book."
    code: str  # 3-letter code for the book.
    idx: int  # 1-based index of the book in the bible.
    name: str  # The name of the book.
    # A list of chapters, which are lists of verses. Note that some verses can be missing, hence None.
    chapters: list[list[str | None]]


async def get_bible(bible_id: int, book_code: str | None = None) -> list[Book]:
    """Get all books in the bible (in order).
    Or if `book_code` is set to a 3-letter code, only fetch that book.
    """
    async with httpx.AsyncClient() as http_client:
        book_names = await get_book_names(bible_id, http_client)
        if book_code is not None:
            book_code = book_code.upper()
            book_names = {book_code: book_names[book_code]}
        print(f"Downloading {len(book_names)} books.")

        with progressbar.ProgressBar(
            min_value=0,
            max_value=len(book_names),
            left_justify=False,
            widgets=[
                progressbar.SimpleProgress(format="%(value_s)s/%(max_value_s)s books "),
                progressbar.Bar(marker="=", left="[", right="]"),
            ],
        ) as pro_bar:

            async def get_book(book_code: str, book_idx: int, book_name: str) -> Book:
                book = Book(
                    book_code,
                    book_idx,
                    book_name,
                    await get_chapters(book_code, bible_id, http_client),
                )
                pro_bar.increment()
                return book

            return await asyncio.gather(
                *map(lambda x: get_book(x[0], x[1][0], x[1][1]), book_names.items())
            )


async def get_chapters(
    book_code: str, bible_id: int, http_client: httpx.AsyncClient
) -> list[list[str | None]]:
    "Get all chapters (which are lists of verses) for a specific book."
    no_chapters: int = await get_no_chapters(book_code, bible_id, http_client)
    return await asyncio.gather(
        *map(
            lambda i: get_verses(book_code, i, bible_id, http_client),
            range(no_chapters),
        )
    )


async def get_book_names(
    bible_id: int, http_client: httpx.AsyncClient
) -> dict[str, Tuple[int, str]]:
    """Get an ordered dictionary with 3-letter book codes as keys
    and tuples of 1-based book indexes and book names as values.
    """
    resp = await http_client.get(
        f"https://www.bible.com/json/bible/books/{bible_id}?filter="
    )
    resp.raise_for_status()
    data = resp.json()
    return {x["usfm"]: (i, x["human"]) for (i, x) in enumerate(data["items"], start=1)}


async def get_no_chapters(
    book_code: str, bible_id: int, http_client: httpx.AsyncClient
) -> int:
    "Get the number of chapters of a book (represented by a 3-letter code)."
    resp = await http_client.get(
        f"https://www.bible.com/json/bible/books/{bible_id}/{book_code}/chapters"
    )
    resp.raise_for_status()
    data = resp.json()
    return len(data["items"])


async def get_verses(
    book_code: str, chapter_idx: int, bible_id: int, http_client: httpx.AsyncClient
) -> list[str | None]:
    """Get a list of all verses in a chapter.
    Some verses might be missing and is then set to `None`.
    """
    url = f"https://www.bible.com/en-GB/bible/{bible_id}/{book_code}.{chapter_idx + 1}.KJV/"
    page = await http_client.get(url)
    page.raise_for_status()  # Raise an exception if the request failed with a bad status code.
    soup = bs4.BeautifulSoup(page.content, "html.parser")
    chapter_find = soup.find(class_=f"chapter ch{chapter_idx + 1}")
    assert isinstance(chapter_find, bs4.Tag)
    no_ver = len(chapter_find.find_all(class_="label", text=NUM_REGEX)) - 1

    verses: list[str | None] = []
    for i in range(no_ver):
        verse = soup.find(class_=re.compile(f"verse v{i + 1}"))
        if verse is None:
            verses.append(None)
        else:
            assert isinstance(verse, bs4.Tag), f"Was type: {type(verse)}, {verse}"
            content: str = "".join((x.text for x in verse.find_all(class_="content")))
            verses.append(content)
    return verses


async def store_bible(
        bible_id: int, output_dir: str, book_code: str|None=None, max_line_len: int | None = None
) -> None:
    """Fetch the bible and save it in `output_dir` with one sub-directory per book
    and one file per chapter.
    The argument `book` can be set to a 3-letter code to specify a certain book to download.
    """
    text_wrapper: TextWrapper | None = TextWrapper() if max_line_len else None
    books: list[Book] = await get_bible(bible_id, book_code)
    for book in books:
        # Make a directory for the book.
        book_dir: str = os.path.join(output_dir, f"{str(book.idx).rjust(2, '0')}_{book.name}")
        os.makedirs(book_dir, exist_ok=True)

        # Compute the length of the numbers for the chapters.
        chapter_num_len: int = math.ceil(math.log10(len(book.chapters)))

        for j, chapter in enumerate(book.chapters):
            # Compute the length of the numbers for the verses.
            verse_num_len: int = math.ceil(math.log10(len(chapter)))
            if text_wrapper is not None:
                text_wrapper.subsequent_indent = " " * (verse_num_len + 2)

            with open(
                os.path.join(book_dir, f"{str(j + 1).rjust(chapter_num_len, '0')}.txt"),
                "w",
            ) as file:
                for k, verse in enumerate(chapter):
                    if verse is None:
                        continue
                    verse_text: str = (
                        str(k + 1).rjust(verse_num_len, " ") + ". " + verse
                    )
                    if text_wrapper is not None:
                        verse_text = "\n".join(text_wrapper.wrap(verse_text))
                    file.write(verse_text + "\n")


def main():
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "bible_id",
        type=int,
        help="The bible.com specific ID for the bible. Can be found in the URL of the bible version.",
    )
    argparser.add_argument("output_dir", help="The base dir for the output.")
    argparser.add_argument("book", nargs="?", help="Only download this book. Should be the 3-letter code for the book.")
    line_len_group = argparser.add_mutually_exclusive_group()
    line_len_group.add_argument(
        "-l",
        "--line-length",
        type=int,
        default=100,
        help="Lines longers than this will be wrapped.",
    )
    line_len_group.add_argument(
        "--no-wrap-lines", action="store_true", help="Don't wrap long lines."
    )
    args = argparser.parse_args()
    asyncio.get_event_loop().run_until_complete(
        store_bible(
            args.bible_id,
            args.output_dir,
            max_line_len=args.line_length if not args.no_wrap_lines else None,
            book_code=args.book,
        )
    )


if __name__ == "__main__":
    main()
