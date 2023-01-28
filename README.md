# bible.com-scraper
Download any bible from bible.com.

This is a single python script for downloading any bible from [bible.com][1] and store it as plain text.
It can also be used as a Python library.

## Prerequisits

You need [Python 3.11][2] or later and [Python Poetry][3].
Please consult your distro's package manager or follow the links to get them installed.

## Running

First, clone this repo and cd into it:
```Bash
git clone https://github.com/tage64/bible.com-scraper
cd bible.com-scraper
```

Then, you must install the virtual environment and launch the Poetry shell:
```Bash
$ poetry install
$ poetry shell
```

You should now be able to run the script as such:
```Bash
$ bible_com_scraper <BIBLE_ID> <OUTPUT_DIR>
```

Where `<BIBLE_ID>` is the id of the bible version you want to download.
To find it out, search up your bible on [bible.com][1] and inspect the URL.
For example, if the url is:
```
https://www.bible.com/versions/841-esp-la-sankta-biblio-1926-esperanto-londona-biblio
```
Then the bible id is 841.

[1]: https://bible.com
[2]: https://www.python.org
[3]: https://python-poetry.org
