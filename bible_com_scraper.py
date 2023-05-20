#!/usr/bin/python3

import re
from textwrap import TextWrapper
import math
import os
import asyncio
import itertools
from enum import Enum
from typing import *

import bs4
import httpx
import progressbar  # type: ignore

# Number of tries if http connection times out.
HTTP_TRIES: int = 3

NUM_REGEX: re.Pattern = re.compile("[1-9][0-9]*")
CHAPTER_CLASS_REGEX: re.Pattern = re.compile("ChapterContent_chapter.*")
VERSE_CLASS_REGEX: re.Pattern = re.compile("ChapterContent_verse.*")
HEADING_CLASS_REGEX: re.Pattern = re.compile("ChapterContent_heading.*")
CONTENT_CLASS_REGEX: re.Pattern = re.compile("ChapterContent_content.*")
NOTE_CLASS_REGEX: re.Pattern = re.compile("ChapterContent_note.*")
VERSE_USFM_REGEX: re.Pattern = re.compile(
    "([A-Za-z0-9]+)\.([1-9][0-9]*)\.([1-9][0-9]*)"
)


class Note(NamedTuple):
    "A note in a verse."
    text: str


class Verse(NamedTuple):
    "A verse is a tuple of its number and content."
    number: int
    content: list[str | Note]


class Heading(NamedTuple):
    "A heading in a chapter."
    text: str


# A chapter is a list of headings and verses. If an element is `None`, that means that the verse is missing.
Chapter = list[Heading | Verse]


class Book(NamedTuple):
    "Information about a bible book."
    code: str  # 3-letter code for the book.
    idx: int  # 1-based index of the book in the bible.
    name: str  # The name of the book.
    chapters: list[Chapter]


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
) -> list[Chapter]:
    "Get all chapters (which are lists of verses) for a specific book."
    no_chapters: int = await get_no_chapters(book_code, bible_id, http_client)
    return await asyncio.gather(
        *map(
            lambda i: get_chapter(book_code, i, bible_id, http_client),
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


async def get_chapter(
    book_code: str, chapter_idx: int, bible_id: int, http_client: httpx.AsyncClient
) -> Chapter:
    "Retrieve the content of a chapter."
    url = f"https://www.bible.com/bible/{bible_id}/{book_code}.{chapter_idx+1}.KJV"
    tries: int = 1
    while True:
        try:
            page = await http_client.get(url)
            break
        except httpx.ReadTimeout:
            assert tries < HTTP_TRIES, f"Http timeout with {tries} tries."
            tries += 1
            continue
    page.raise_for_status()  # Raise an exception if the request failed with a bad status code.
    soup = bs4.BeautifulSoup(page.content, "html.parser")
    chapter_find = soup.find(class_=CHAPTER_CLASS_REGEX)
    assert isinstance(chapter_find, bs4.Tag)

    # This is a list of HTML tags corresponding to verses.
    # There might be zero to many number of verse-tags per verse.
    verse_and_heading_tags: List = chapter_find.find_all(
        class_=[VERSE_CLASS_REGEX, HEADING_CLASS_REGEX]
    )

    chapter: Chapter = Chapter([])
    for tag in verse_and_heading_tags:
        assert isinstance(tag, bs4.Tag)
        if isinstance(tag["class"], list):
            tag_class: str = tag["class"][0]
        else:
            tag_class = tag["class"]
        if HEADING_CLASS_REGEX.fullmatch(tag_class):
            assert tag.string is not None
            if chapter != [] and isinstance(chapter[-1], Heading):
                # Multiple headings without text between are concatinated.
                chapter[-1] = Heading(chapter[-1].text + " " + tag.string)
            else:
                chapter.append(Heading(tag.string))
        elif VERSE_CLASS_REGEX.fullmatch(tag_class):
            usfm = tag["data-usfm"]
            assert isinstance(usfm, str)
            usfm_match = VERSE_USFM_REGEX.fullmatch(usfm)
            assert isinstance(
                usfm_match, re.Match
            ), f"Couldn't match usfm code: {usfm}."
            assert (
                usfm_match[1] == book_code
            ), f"Wrong book code for verse {usfm_match.string}, expected {book_code}."
            assert usfm_match[2] == str(
                chapter_idx + 1
            ), f"Wrong chapter index for verse. {usfm_match.string}, expected {chapter_idx + 1}."
            verse_no: int = int(usfm_match[3])
            content_and_note_tags: list = tag.find_all(
                class_=[CONTENT_CLASS_REGEX, NOTE_CLASS_REGEX]
            )
            verse_content: list[str | Note] = []
            for inner_tag in content_and_note_tags:
                assert isinstance(inner_tag, bs4.Tag)
                text: str = "".join(inner_tag.stripped_strings).strip()
                if not text:
                    continue
                if isinstance(inner_tag["class"], list):
                    inner_tag_class: str = inner_tag["class"][0]
                else:
                    inner_tag_class = inner_tag["class"]
                if CONTENT_CLASS_REGEX.fullmatch(inner_tag_class):
                    if verse_content and isinstance(verse_content[-1], str):
                        verse_content[-1] += text
                    else:
                        verse_content.append(text)
                else:
                    verse_content.append(Note(text))
            if not verse_content:
                continue
            if (
                chapter != []
                and isinstance(chapter[-1], Verse)
                and chapter[-1].number == verse_no
            ):
                # Two adjacent verse tags with the same number aare concatinated.
                chapter[-1] = Verse(verse_no, chapter[-1].content + verse_content)
            else:
                chapter.append(Verse(verse_no, verse_content))
        else:
            assert False, f"Bad verse item class: {tag['class']}"
    return chapter


# Warning to the reader: The following function is a bit messy and poorly documented.
# It does the job of writing the bible to pretty looking text files.

# A TextBlock is either just a string, or it is a prefix
# followed by a list of TextBlocks. The intention is to print the prefix followed by each
# block indented as much as the prefix is wide.
# This type is only used in the following function, but needs to be toplevel to to make mypy happy.
TextBlock = str | Tuple[str, list["TextBlock"]]


async def store_bible(
    bible_id: int,
    output_dir: str,
    book_code: str | None = None,
    max_line_len: int | None = None,
    include_notes: bool = False,
) -> None:
    """Fetch the bible and save it in `output_dir` with one sub-directory per book
    and one file per chapter.
    The argument `book` can be set to a 3-letter code to specify a certain book to download.
    """

    def text_block_to_str(text: TextBlock) -> str:
        def text_block_to_lines(
            text: TextBlock, max_line_len: int | None
        ) -> Iterable[str]:
            "Convert a TextBlock to a string, see the comment for TextBlock for more info."
            if isinstance(text, str):
                return text.splitlines()
            (prefix, items) = text
            indent: int = len(prefix)
            inner_line_len: int | None = (
                max_line_len - indent if max_line_len is not None else None
            )
            assert (
                inner_line_len is None or inner_line_len > 0
            ), f"Lines are too short with prefix: {prefix}"
            lines: Iterable[str] = (
                line
                for item in items
                for line in text_block_to_lines(item, inner_line_len)
            )
            if inner_line_len is not None:
                text_wrapper: TextWrapper = TextWrapper(
                    width=inner_line_len, replace_whitespace=False
                )
                lines = (
                    wrapped_line
                    for line in lines
                    for wrapped_line in text_wrapper.wrap(line)
                )
            return (
                x + y
                for (x, y) in itertools.zip_longest(
                    [prefix], lines, fillvalue=" " * indent
                )
            )

        return "\n".join(text_block_to_lines(text, max_line_len)) + "\n"

    books: list[Book] = await get_bible(bible_id, book_code)
    for book in books:
        # Make a directory for the book.
        book_dir: str = os.path.join(
            output_dir, f"{str(book.idx).rjust(2, '0')}_{book.name}"
        )
        os.makedirs(book_dir, exist_ok=True)

        # Compute the length of the numbers for the chapters.
        chapter_num_len: int = math.ceil(math.log10(len(book.chapters)))

        for j, chapter in enumerate(book.chapters):
            # Compute the length of the numbers for the verses.
            verse_num_len: int = math.ceil(
                math.log10(max((x.number for x in chapter if isinstance(x, Verse))))
            )

            with open(
                os.path.join(book_dir, f"{str(j + 1).rjust(chapter_num_len, '0')}.txt"),
                "w",
            ) as file:
                notes: list[Note] = []
                for item in chapter:
                    if isinstance(item, Verse):
                        prefix: str = (
                            " " + str(item.number).rjust(verse_num_len, " ") + ". "
                        )
                        text_blocks: list[str] = []
                        prev_segment_type = None
                        for segment in item.content:
                            if isinstance(segment, str):
                                match prev_segment_type:
                                    case None | "text":
                                        text_blocks.append(segment)
                                    case "note":
                                        text_blocks[-1] += " " + segment
                                prev_segment_type = "text"
                            else:
                                if not include_notes:
                                    continue
                                notes.append(segment)
                                reference_text: str = f"[{len(notes)}]"
                                match prev_segment_type:
                                    case None:
                                        text_blocks.append(reference_text)
                                    case "text" | "note":
                                        text_blocks[-1] += " " + reference_text
                                prev_segment_type = "note"
                        file.write(text_block_to_str((prefix, text_blocks)))  # type: ignore
                    else:
                        file.write(text_block_to_str(("# ", [item.text])))
                for i, note in enumerate(notes):
                    file.write(text_block_to_str((f"[{i+1}] ", [note.text])))


def main():
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "bible_id",
        type=int,
        help="The bible.com specific ID for the bible. Can be found in the URL of the bible version.",
    )
    argparser.add_argument("output_dir", help="The base dir for the output.")
    argparser.add_argument(
        "book",
        nargs="?",
        help="Only download this book. Should be the 3-letter code for the book.",
    )
    argparser.add_argument(
        "--notes", action="store_true", help="Include notes for each chapter."
    )
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
            include_notes=args.notes,
        )
    )


if __name__ == "__main__":
    main()
