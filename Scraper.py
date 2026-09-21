from bs4 import BeautifulSoup
import requests
import re
import sqlite3
import math
import csv
from collections import Counter

# STOPWORDS list — generated with AI assistance as documented in the development section. Covers standard English function words, website
# UI boilerplate, publication metadata, and generic journalistic filler.
# Words in this list are removed before TF-IDF encoding because they are unlikely to carry any political signal.
STOPWORDS = (
    # Standard English function words
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "it's", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "same", "so", "than",
    "too", "very", "just", "as", "if", "then", "because", "while", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "once", "here", "there", "any", "also", "even", "still",
    "now", "however", "although", "though", "since", "unless", "whether",
    "until", "therefore", "thus", "hence", "indeed", "yet", "nor", "either",
    "neither", "according", "regarding", "following", "including", "within",
    "without", "along", "across", "behind", "beyond", "per", "plus", "via",

    # Website UI / navigation boilerplate
    "cookie", "cookies", "banner", "consent", "accept", "decline", "dismiss",
    "close", "popup", "modal", "overlay", "gdpr", "ccpa", "privacy",
    "policy", "terms", "conditions", "disclaimer", "notice", "alert",
    "notification", "notifications", "subscribe", "subscription", "newsletter",
    "signup", "sign", "login", "log", "register", "account", "username",
    "password", "email", "submit", "button", "click", "tap", "swipe",
    "scroll", "menu", "navigation", "nav", "sidebar", "footer", "header",
    "homepage", "home", "page", "pages", "site", "website", "web", "link",
    "links", "url", "href", "search", "browse", "filter", "sort", "share",
    "tweet", "post", "comment", "comments", "reply", "replies", "load",
    "loading", "reload", "refresh", "continue", "next", "previous", "back",
    "skip", "read", "more", "less", "show", "hide", "expand", "collapse",
    "toggle", "select", "option", "options", "settings", "preferences",
    "manage", "update", "save", "cancel", "confirm", "ok", "yes", "no",

    # Publication / article metadata boilerplate
    "published", "updated", "posted", "written", "reported", "edited",
    "author", "authors", "writer", "reporter", "journalist", "editor",
    "staff", "contributor", "correspondent", "bureau", "desk", "section",
    "category", "categories", "tag", "tags", "topic", "topics", "label",
    "date", "time", "ago", "today", "yesterday", "week", "month", "year",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "am", "pm", "est", "gmt", "utc", "bst", "edt", "pst",
    "minute", "minutes", "hour", "hours", "days", "weeks", "months", "years",
    "edition", "print", "digital", "online", "breaking", "exclusive",
    "developing", "live", "watch", "video", "audio", "podcast", "gallery",
    "photo", "photos", "image", "images", "graphic", "chart", "map",
    "infographic", "caption", "credit", "source", "sources", "via",
    "advertisement", "advertise", "advertising", "sponsored", "promotion",
    "related", "recommended", "trending", "popular", "featured", "latest",
    "recent", "top", "most", "read", "viewed", "shared", "liked",

    # Generic journalistic filler
    "said", "says", "told", "added", "noted", "stated", "explained",
    "according", "confirmed", "reported", "announced", "revealed", "claimed",
    "suggested", "indicated", "argued", "believed", "thought", "felt",
    "knew", "went", "came", "got", "made", "took", "put", "set", "let",
    "keep", "kept", "given", "give", "gives", "giving", "see", "seen",
    "look", "looked", "looking", "looks", "seem", "seemed", "seems",
    "appear", "appeared", "appears", "become", "became", "becomes",
    "remain", "remained", "remains", "include", "included", "includes",
    "involve", "involved", "involves", "use", "used", "using", "uses",
    "make", "making", "makes", "help", "helped", "helping", "helps",
    "want", "wanted", "wants", "need", "needed", "needs", "try", "tried",
    "trying", "tries", "call", "called", "calling", "calls", "ask", "asked",
    "come", "coming", "comes", "go", "going", "goes", "gone", "get",
    "getting", "gets", "know", "knowing", "knows", "think", "thinking",
    "thinks", "thought", "work", "working", "works", "worked", "start",
    "started", "starting", "starts", "end", "ended", "ending", "ends",
    "mean", "means", "meaning", "meant", "find", "finds", "finding",
    "found", "show", "shows", "showing", "showed", "shown", "turn",
    "turns", "turned", "turning", "move", "moves", "moved", "moving",
    "place", "places", "placed", "placing", "point", "points", "pointed",
    "play", "plays", "played", "playing", "run", "runs", "ran", "running",
    "lead", "leads", "led", "leading", "hold", "holds", "held", "holding",
    "based", "base", "number", "numbers", "part", "parts", "way", "ways",
    "thing", "things", "people", "person", "man", "woman", "men", "women",
    "time", "times", "place", "day", "days", "year", "new", "old",
    "large", "small", "big", "long", "short", "high", "low", "good",
    "bad", "great", "little", "own", "right", "early", "last", "next",
    "first", "second", "third", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million",
    "billion", "percent", "per", "cent", "much", "many", "well", "often",
    "never", "always", "already", "almost", "around", "away", "back",
    "ever", "together", "likely", "recently", "currently", "previously",
    "formally", "officially", "largely", "widely", "increasingly", "highly",
    "nearly", "particularly", "generally", "simply", "especially", "also",
    "s", "t", "re", "ve", "ll", "d", "m", "wasn", "isn", "aren", "didn",
    "doesn", "hadn", "hasn", "haven", "wouldn", "couldn", "shouldn",
    "won", "don", "can't", "won't", "isn't", "aren't", "didn't",
    "doesn't", "hadn't", "hasn't", "haven't", "wouldn't", "couldn't",
    "shouldn't", "wasn't", "weren't", "not", "n't",
)


def scraper(url):
    # SCRAPER

    # requests.get fetches the raw HTML from the URL
    page_to_scrape = requests.get(url)
    soup = BeautifulSoup(page_to_scrape.text, "html.parser")
    # get_text() strips all HTML tags and returns only visible text
    text = soup.get_text()

    # DATA CLEANER

    # convert everything to lowercase for consistent tokenisation
    text = text.lower()
    # collapse all whitespace i
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # remove any non-ASCII characters 
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # remove all punctuation so words are stored without trailing commas etc
    text = re.sub(r'[^\w\s]', '', text)

    # UNIGRAMS with stopword filtering

    # split on whitespace, filter out short words and stopwords
    words = [s.strip() for s in text.split() if len(s.strip()) > 2 and s.strip() not in STOPWORDS]
    # total_words is calculated here before bigrams are added so the TF denominator reflects only real word count not bigram count
    total_words = len(words)

    # BIGRAMS

    # join adjacent word pairs with underscore 
    bigrams = [words[i] + "_" + words[i+1] for i in range(len(words) - 1)]

    sentences = words + bigrams
    return sentences, total_words


def ensure_schema(database):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    # SQLite does not enforce foreign keys by default, this turns it on
    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS TBLarticles (
        artid       INTEGER PRIMARY KEY,
        title       TEXT,
        url         TEXT,
        bias_rating REAL,
        content     TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS TBLdictionary (
        word_id INTEGER PRIMARY KEY,
        word    TEXT NOT NULL UNIQUE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS TBLdocumentfrequency (
        word_id INTEGER PRIMARY KEY,
        df      REAL NOT NULL,
        FOREIGN KEY (word_id) REFERENCES TBLdictionary(word_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS TBLtermfrequency (
        artid   INTEGER NOT NULL,
        word_id INTEGER NOT NULL,
        tf      REAL NOT NULL,
        tf_idf  REAL,
        PRIMARY KEY (artid, word_id),
        FOREIGN KEY (artid) REFERENCES TBLarticles(artid)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
        FOREIGN KEY (word_id) REFERENCES TBLdictionary(word_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );
    """)

    connection.commit()
    connection.close()


def encoder(sentences, total_words, database, url=None, title=None, bias_rating=None):

    # if no words get scraped then nothing gets added to database
    if not sentences:
        return None

    ensure_schema(database)

    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    # ensures that foreign keys are enabled
    cursor.execute("PRAGMA foreign_keys = ON;")

    # re-concatenates word list to store as metadata
    content = " ".join(sentences)

    cursor.execute(
        "INSERT INTO TBLarticles (title, url, bias_rating, content) VALUES (?, ?, ?, ?);",
        (title, url, bias_rating, content)
    )
    # lastrowid gives the auto assigned primary key
    artid = cursor.lastrowid

    counts = Counter(sentences)
    # unique_words gets used later for DF calculation
    unique_words = set(counts.keys())

    for word, count in counts.items():
        # check if word is already in dictionary
        cursor.execute(
            "SELECT word_id FROM TBLdictionary WHERE word = ?;",
            (word,)
        )
        row = cursor.fetchone()

        if row is None:
            # if the word is new add it to the dictionary
            cursor.execute(
                "INSERT INTO TBLdictionary (word) VALUES (?);",
                (word,)
            )
            word_id = cursor.lastrowid
        else:
            # if word already exists get its ID
            word_id = row[0]

        # TF = number of times word appears / total words in article
        tf = count / total_words

        cursor.execute(
            "INSERT INTO TBLtermfrequency (artid, word_id, tf, tf_idf) VALUES (?, ?, ?, NULL);",
            (artid, word_id, tf)
        )

    # DF loop is separate from TF loop because document frequency should
    # only increment by 1 per article regardless of how many times
    # the word appears within that article
    for word in unique_words:
        cursor.execute(
            "SELECT word_id FROM TBLdictionary WHERE word = ?;",
            (word,)
        )
        row = cursor.fetchone()
        if row is None:
            continue
        word_id = row[0]

        # check if this word already has a DF entry
        cursor.execute(
            "SELECT df FROM TBLdocumentfrequency WHERE word_id = ?;",
            (word_id,)
        )
        df_row = cursor.fetchone()

        if df_row is None:
            # first article containing this word
            cursor.execute(
                "INSERT INTO TBLdocumentfrequency (word_id, df) VALUES (?, 1);",
                (word_id,)
            )
        else:
            # word has appeared in previous articles, increment count
            cursor.execute(
                "UPDATE TBLdocumentfrequency SET df = df + 1 WHERE word_id = ?;",
                (word_id,)
            )

    connection.commit()
    connection.close()
    return artid


def update_tfidf(database, artid):

    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # count total articles in database, needed for IDF calculation
    cursor.execute("SELECT COUNT(*) FROM TBLarticles;")
    N = cursor.fetchone()[0]

    # can't calculate IDF with no articles
    if N == 0:
        connection.close()
        return

    # fetch TF and DF values for every word in this article using a JOIN
    # more efficient than querying each table separately per word
    cursor.execute("""
        SELECT tf.word_id, tf.tf, df.df
        FROM TBLtermfrequency tf
        JOIN TBLdocumentfrequency df ON df.word_id = tf.word_id
        WHERE tf.artid = ?;
    """, (artid,))
    rows = cursor.fetchall()

    for word_id, tf, df in rows:
        # smoothed IDF formula from scikit-learn TfidfTransformer docs
        # +1 inside the log prevents division by zero when df equals N
        # +1 outside the log ensures IDF never goes negative
        idf = math.log((N + 1) / (df + 1)) + 1.0
        tf_idf = tf * idf

        cursor.execute("""
            UPDATE TBLtermfrequency
            SET tf_idf = ?
            WHERE artid = ? AND word_id = ?;
        """, (tf_idf, artid, word_id))

    connection.commit()
    connection.close()


def parse_bias(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        parsed = float(s)
        # reject scores outside valid range rather than clipping
        # so bad data doesn't get treated as extreme left or right
        if parsed < 0 or parsed > 100:
            return None
        return parsed
    except ValueError:
        return None


def import_csv_to_db(csv_path, database):
    ensure_schema(database)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        # check the CSV has headers at all
        if reader.fieldnames is None:
            print("Error: CSV file is empty or has no headers")
            return

        fieldnames = [name.lower().strip() for name in reader.fieldnames]

        if "url" not in fieldnames:
            print(f"Error: could not find a URL column in CSV headers: {reader.fieldnames}")
            return

        processed = 0
        skipped = 0

        for row in reader:
            # normalise keys to lowercase so column name capitalisation doesnt matter
            row_lower = {k.lower().strip(): v for k, v in row.items()}

            title = row_lower.get("title", "").strip() or None
            url = row_lower.get("url", "").strip()
            bias = parse_bias(row_lower.get("bias rating", ""))

            # skip rows with no URL as there is nothing to scrape
            if not url:
                skipped += 1
                continue

            try:
                sentences, total_words = scraper(url)
                artid = encoder(
                    sentences, total_words, database,
                    url=url, title=title, bias_rating=bias
                )
                if artid is not None:
                    update_tfidf(database, artid)
                    processed += 1
                    display_title = title if title else "[No Title]"
                    print(f"Article {processed} processed | {display_title}")
                else:
                    skipped += 1
            except Exception as e:
                print(f"[SKIP] {url} | {e}")
                skipped += 1

        print(f"Done. Processed={processed}, Skipped={skipped}")


if __name__ == "__main__":
    csv_file = input("Enter path to CSV (title,url,bias): ").strip()
    db_file = "tf_idf.db"

    import_csv_to_db(csv_file, db_file)
    print("Database build complete.")