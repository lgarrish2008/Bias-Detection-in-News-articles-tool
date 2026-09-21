import time
import re
import random
import urllib.parse
from urllib.parse import urlparse

import requests
import feedparser
import pandas as pd


# =========================
# SETTINGS
# =========================
OUTPUT_FILE = "political_opinion_urls.csv"

TARGET_PER_BUCKET = 270        # Target for this number per bucket of sources
MAX_PASSES = 10                 # How many times to loop through all sources
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_FEEDS = 0.7       # prevent getting blocked from servers


# Per-source caps (buckets with less get more)
def max_for_bucket(bucket: str) -> int:
    if bucket in {"C", "CR"}:
        return 140
    if bucket == "R":
        return 120
    return 80


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7",
}


# =========================
# POLITICS/OPINION FILTERING
# =========================
POLITICS_TAGS = {
    "politics", "u.s. politics", "us politics", "government", "elections",
    "white house", "congress", "parliament", "policy", "campaign", "democracy",
    "international relations", "foreign policy", "national security", "defense",
    "immigration", "economy", "tax", "trade", "uk politics", "political"
}

OPINION_TAGS = {
    "opinion", "comment", "commentary", "editorial", "op-ed", "analysis",
    "column", "columns", "letters", "viewpoint", "views"
}

URL_CUES = [
    "/politics", "/political", "/us/politics", "/world/politics",
    "/uk/politics", "/election", "/elections", "/government", "/policy",
    "/opinion", "/comment", "/commentisfree", "/editorial", "/op-ed",
    "/columns", "/column", "/analysis", "/views", "/viewpoint",
    "/whitehouse", "/congress", "/senate", "/house", "/campaigns",
    "/nation/politics", "/news/politics", "/politics-and-policy",
]

TEXT_CUES = [
    "election", "elections", "vote", "voting", "poll", "ballot",
    "parliament", "mp ", "tory", "conservative", "labour", "lib dem", "liberal democrat",
    "democrat", "republican", "white house", "congress", "senate", "house of representatives",
    "government", "minister", "prime minister", "policy", "bill", "law", "legislation",
    "supreme court", "court", "judge", "constitutional",
    "nato", "un ", "ukraine", "russia", "china", "israel", "gaza", "iran",
    "immigration", "border", "asylum", "tax", "inflation", "budget", "spending",
    "opinion", "editorial", "column", "commentary", "analysis"
]

EXCLUDE_CUES = [
    "/sport", "/sports", "/football", "/cricket", "/nba", "/nfl",
    "/entertainment", "/movies", "/tv", "/music", "/celebrity",
    "/travel", "/food", "/recipes", "/style", "/fashion", "/culture",
    "/tech/review", "/reviews", "/gaming", "/games",
]


# =========================
# HARD BLOCK: NO GOOGLE DOMAINS
# =========================
BLOCKED_HOST_SUFFIXES = (
    "google.com",
    "news.google.com",
    "consent.google.com",
    "policies.google.com",
    "accounts.google.com",
    "googleusercontent.com",
)

def is_blocked_domain(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower()
        return any(host == s or host.endswith("." + s) for s in BLOCKED_HOST_SUFFIXES)
    except Exception:
        return False


# =========================
# FEED HELPERS
# =========================
def parse_feed(url: str):
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return feedparser.parse(r.content)


def _entry_tags(entry) -> set[str]:
    tags = set()
    for t in entry.get("tags", []) or []:
        term = (t.get("term") or "").strip().lower()
        if term:
            tags.add(term)
    return tags


def is_politics_or_opinion(entry) -> bool:
    link = (entry.get("link") or "").lower()
    title = (entry.get("title") or "").lower()
    summary = (entry.get("summary") or entry.get("description") or "").lower()

    for bad in EXCLUDE_CUES:
        if bad in link:
            return False

    tags = _entry_tags(entry)
    if tags:
        if any(t in POLITICS_TAGS for t in tags):
            return True
        if any(t in OPINION_TAGS for t in tags):
            return True

    if any(cue in link for cue in URL_CUES):
        return True

    text = f"{title} {summary}"
    if any(k in text for k in TEXT_CUES):
        return True

    return False


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    url = re.sub(r"#.*$", "", url)
    url = re.sub(r"([?&])(utm_[^=&]+|fbclid|gclid|mc_cid|mc_eid)=[^&]+", r"\1", url)
    url = re.sub(r"[?&]$", "", url)
    return url


def pick_best_entry_link(entry) -> str:
    """
    Prefer non-google alternate links when available.
    """
    candidates = []
    main = entry.get("link")
    if main:
        candidates.append(main)

    for l in entry.get("links", []) or []:
        href = l.get("href")
        if not href:
            continue
        if not is_blocked_domain(href):
            return href
        candidates.append(href)

    return candidates[0] if candidates else ""


def clean_link(raw_link: str) -> str:
    link = normalize_url(raw_link)
    if not link:
        return ""
    if is_blocked_domain(link):
        return ""
    return link


# =========================
# SOURCES (DIRECT RSS/ATOM ONLY) - NO GOOGLE RSS
# =========================
# Bucket codes:
# L  = Left
# CL = Lean Left
# C  = Center
# CR = Lean Right
# R  = Right

FEEDS = {
    # -------------------------
    # LEFT (L)
    # -------------------------
    "Guardian_US_L":               ("L", "https://www.theguardian.com/us/rss"),
    "Guardian_Politics_UK_L":      ("L", "https://www.theguardian.com/politics/rss"),
    "Guardian_Opinion_L":          ("L", "https://www.theguardian.com/commentisfree/rss"),
    "MotherJones_L":               ("L", "https://www.motherjones.com/feed/"),
    "DemocracyNow_L":              ("L", "https://www.democracynow.org/democracynow.rss"),
    "TheNation_L":                 ("L", "https://www.thenation.com/feed/"),
    "Jacobin_L":                   ("L", "https://jacobin.com/feed"),
    "Truthout_L":                  ("L", "https://truthout.org/feed/"),
    "CommonDreams_L":              ("L", "https://www.commondreams.org/feeds/feed.rss"),
    "Intercept_Politics_L":        ("L", "https://theintercept.com/politics/feed/"),
    "Vox_Politics_L":              ("L", "https://www.vox.com/rss/politics/index.xml"),
    "Slate_Politics_L":            ("L", "https://slate.com/feeds/politics.rss"),
    "HuffPost_Politics_L":         ("L", "https://www.huffpost.com/section/politics/feed"),
    "Salon_NewsPolitics_L":        ("L", "http://www.salon.com/category/news/feed/rss/"),
    "Alternet_L":                  ("L", "https://www.alternet.org/feed/"),
    "RawStory_L":                  ("L", "http://www.rawstory.com/feeds/feed.rss"),
    "TalkingPointsMemo_L":         ("L", "https://talkingpointsmemo.com/feed"),
    "NewStatesman_L":              ("L", "https://www.newstatesman.com/feed"),
    "NovaraMedia_L":               ("L", "https://novaramedia.com/feed/"),
    "OpenDemocracy_L":             ("L", "https://www.opendemocracy.net/en/feed/"),
    "CurrentAffairs_L":            ("L", "https://www.currentaffairs.org/feed"),
    "Prospect_UK_L":               ("L", "https://www.prospectmagazine.co.uk/feed"),
    "TheAmericanProspect_L":       ("L", "https://prospect.org/feeds/feed.rss"),
    "Dissent_L":                   ("L", "https://www.dissentmagazine.org/feed"),
    "NewInternationalist_L":       ("L", "https://newint.org/feed"),
    "PinkNews_L":                  ("L", "https://www.thepinknews.com/feed/"),

    # -------------------------
    # LEAN LEFT (CL)
    # -------------------------
    "CNN_Politics_CL":             ("CL", "http://rss.cnn.com/rss/cnn_allpolitics.rss"),
    "NPR_Politics_CL":             ("CL", "https://feeds.npr.org/1014/rss.xml"),
    "Politico_Politics_CL":        ("CL", "https://www.politico.com/rss/politics08.xml"),
    "Axios_CL":                    ("CL", "https://www.axios.com/feeds/feed.rss"),
    "NBC_Politics_CL":             ("CL", "http://feeds.nbcnews.com/feeds/nbcpolitics"),
    "MSNBC_CL":                    ("CL", "https://feeds.nbcnews.com/msnbc/public/news"),
    "ABC_Politics_CL":             ("CL", "https://feeds.abcnews.com/abcnews/politicsheadlines"),
    "NYT_Politics_CL":             ("CL", "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml"),
    "NYT_Opinion_CL":              ("CL", "https://rss.nytimes.com/services/xml/rss/nyt/Opinion.xml"),
    "WaPo_Politics_CL":            ("CL", "http://feeds.washingtonpost.com/rss/politics"),
    "PBS_Politics_CL":             ("CL", "https://www.pbs.org/newshour/feeds/rss/politics"),
    "Time_Politics_CL":            ("CL", "https://time.com/section/politics/feed/"),
    "TheAtlantic_All_CL":          ("CL", "https://www.theatlantic.com/feed/all/"),
    "NewYorker_News_CL":           ("CL", "https://www.newyorker.com/feed/news"),
    "ForeignPolicy_CL":            ("CL", "https://foreignpolicy.com/feed/"),
    "Lawfare_CL":                  ("CL", "https://www.lawfaremedia.org/rss.xml"),
    "JustSecurity_CL":             ("CL", "https://www.justsecurity.org/feed/"),
    "Brookings_CL":                ("CL", "https://www.brookings.edu/feed/"),
    "PewResearch_CL":              ("CL", "https://www.pewresearch.org/feed/"),
    "ChathamHouse_CL":             ("CL", "https://www.chathamhouse.org/rss"),
    "RUSI_CL":                     ("CL", "https://www.rusi.org/rss.xml"),
    "RAND_CL":                     ("CL", "https://www.rand.org/rss.html"),
    "SkyNews_Politics_CL":         ("CL", "http://feeds.skynews.com/feeds/rss/politics.xml"),
    "Channel4News_CL":             ("CL", "https://www.channel4.com/news/feed"),
    "Independent_UK_Politics_CL":  ("CL", "https://www.independent.co.uk/news/uk/politics/rss"),

    # -------------------------
    # CENTER (C)
    # -------------------------
    "BBC_Politics_C":              ("C", "https://feeds.bbci.co.uk/news/politics/rss.xml"),
    "BBC_US_C":                    ("C", "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml"),
    "BBC_World_C":                 ("C", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    "AP_Top_C":                    ("C", "https://apnews.com/hub/ap-top-news?output=rss"),
    "AP_Politics_C":               ("C", "https://apnews.com/hub/politics?output=rss"),
    "USA Today_Washington_C":      ("C", "https://rssfeeds.usatoday.com/UsatodaycomWashington-TopStories"),
    "CBS_Politics_C":              ("C", "https://www.cbsnews.com/latest/rss/politics/"),
    "TheHill_C":                   ("C", "https://thehill.com/rss/syndicator/19110"),
    "Politifact_C":                ("C", "https://www.politifact.com/rss/all/"),
    "FactCheck_C":                 ("C", "https://www.factcheck.org/feed/"),
    "ProPublica_C":                ("C", "https://www.propublica.org/feeds/propublica/main"),
    "CSMonitor_C":                 ("C", "https://www.csmonitor.com/rss"),
    "AlJazeera_All_C":             ("C", "https://www.aljazeera.com/xml/rss/all.xml"),
    "DW_All_C":                    ("C", "https://rss.dw.com/rdf/rss-en-all"),
    "France24_All_C":              ("C", "https://www.france24.com/en/rss"),
    "Euronews_World_C":            ("C", "https://www.euronews.com/rss?level=theme&name=news"),
    "ReutersAgency_Politics_C":    ("C", "https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best"),
    "FT_News_C":                   ("C", "https://www.ft.com/news-feed?format=rss"),
    "Economist_Feed_C":            ("C", "https://www.economist.com/rss"),

    # -------------------------
    # LEAN RIGHT (CR)
    # -------------------------
    "NationalReview_CR":           ("CR", "https://www.nationalreview.com/feed/"),
    "TheDispatch_CR":              ("CR", "https://thedispatch.com/feed/"),
    "Reason_CR":                   ("CR", "https://reason.com/latest/feed/"),
    "RealClearPolitics_CR":        ("CR", "https://www.realclearpolitics.com/rss"),
    "WashingtonExaminer_CR":       ("CR", "https://www.washingtonexaminer.com/feed"),
    "WashingtonTimes_Politics_CR": ("CR", "https://www.washingtontimes.com/rss/headlines/news/politics/"),
    "NYPost_Politics_CR":          ("CR", "https://nypost.com/politics/feed/"),
    "Telegraph_UK_CR":             ("CR", "https://www.telegraph.co.uk/rss.xml"),
    "Spectator_UK_CR":             ("CR", "https://www.spectator.co.uk/feed/"),
    "UnHerd_CR":                   ("CR", "https://unherd.com/feed/"),
    "Spiked_CR":                   ("CR", "https://www.spiked-online.com/feed/"),
    "DailySignal_CR":              ("CR", "https://www.dailysignal.com/feed/"),
    "FreeBeacon_CR":               ("CR", "https://freebeacon.com/feed/"),
    "CityJournal_CR":              ("CR", "https://www.city-journal.org/rss.xml"),
    "Commentary_CR":               ("CR", "https://www.commentary.org/feed/"),

    # -------------------------
    # RIGHT (R)
    # -------------------------
    "Fox_Politics_R":              ("R", "http://feeds.foxnews.com/foxnews/politics"),
    "Fox_Opinion_R":               ("R", "http://feeds.foxnews.com/foxnews/opinion"),
    "DailyWire_R":                 ("R", "https://www.dailywire.com/feeds/rss.xml"),
    "Breitbart_R":                 ("R", "https://www.breitbart.com/feed/"),
    "Newsmax_Politics_R":          ("R", "https://www.newsmax.com/rss/Politics/1"),
    "DailyCaller_R":               ("R", "https://dailycaller.com/feed/"),
    "Federalist_R":                ("R", "https://thefederalist.com/feed/"),
    "Townhall_R":                  ("R", "https://townhall.com/rss/"),
    "PJMedia_R":                   ("R", "https://pjmedia.com/feed"),
    "TheBlaze_R":                  ("R", "https://www.theblaze.com/feeds/feed.rss"),
    "GatewayPundit_R":             ("R", "https://www.thegatewaypundit.com/feed/"),
    "AmericanConservative_R":      ("R", "https://www.theamericanconservative.com/feed/"),
    "AmericanThinker_R":           ("R", "https://www.americanthinker.com/rss.xml"),
    "LifeSiteNews_R":              ("R", "https://www.lifesitenews.com/feed/"),
    "Cato_R":                      ("R", "https://www.cato.org/rss.xml"),
    "Mises_R":                     ("R", "https://mises.org/rss.xml"),
    "Heritage_R":                  ("R", "https://www.heritage.org/rss"),
}


# =========================
# COLLECTION
# =========================
def main():
    buckets = ["L", "CL", "C", "CR", "R"]
    bucket_counts = {b: 0 for b in buckets}

    seen = set()
    rows = []

    sources = list(FEEDS.items())

    def all_done() -> bool:
        return all(bucket_counts[b] >= TARGET_PER_BUCKET for b in buckets)

    def bucket_deficit(bucket: str) -> int:
        return max(0, TARGET_PER_BUCKET - bucket_counts[bucket])

    for p in range(1, MAX_PASSES + 1):
        if all_done():
            break

        print(f"\n=== PASS {p}/{MAX_PASSES} ===")
        progress_before = bucket_counts.copy()

        # Shuffle sources each pass to avoid always hitting the same few first
        random.shuffle(sources)

        # Sort sources so we prioritise buckets with biggest deficit
        sources_sorted = sorted(
            sources,
            key=lambda kv: bucket_deficit(kv[1][0]),
            reverse=True
        )

        for source, (bucket, feed_url) in sources_sorted:
            if all_done():
                break
            if bucket_counts[bucket] >= TARGET_PER_BUCKET:
                continue

            print(f"\nFetching from {source} [{bucket}]")
            try:
                feed = parse_feed(feed_url)
            except Exception as e:
                print(f"  ERROR fetching/parsing: {e}")
                time.sleep(SLEEP_BETWEEN_FEEDS)
                continue

            entries = getattr(feed, "entries", []) or []
            print(f"  Entries found: {len(entries)}")

            kept = 0
            per_source_cap = max_for_bucket(bucket)

            for entry in entries:
                if kept >= per_source_cap:
                    break
                if bucket_counts[bucket] >= TARGET_PER_BUCKET:
                    break

                raw = pick_best_entry_link(entry)
                link = clean_link(raw)

                if not link:
                    continue
                if link in seen:
                    continue

                if not is_politics_or_opinion(entry):
                    continue

                title = (entry.get("title") or "").strip()
                published = (entry.get("published") or entry.get("updated") or "").strip()

                seen.add(link)
                rows.append({
                    "bucket": bucket,
                    "source": source,
                    "title": title,
                    "url": link,
                    "published": published
                })
                kept += 1
                bucket_counts[bucket] += 1

            print(f"  Kept (politics/opinion): {kept}")
            print(f"  Bucket totals: {bucket_counts}")

            time.sleep(SLEEP_BETWEEN_FEEDS)

        if bucket_counts == progress_before:
            print("\nNo progress this pass. Feeds are exhausted or filter is too strict.")
            break

    df = pd.DataFrame(rows)

    # Trim to exact target per bucket (if overshot)
    if not df.empty:
        trimmed = []
        for b in ["L", "CL", "C", "CR", "R"]:
            sub = df[df["bucket"] == b].head(TARGET_PER_BUCKET)
            trimmed.append(sub)
        df = pd.concat(trimmed, ignore_index=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(df)} political/opinion URLs to {OUTPUT_FILE}")
    print("Final bucket counts:")
    if not df.empty:
        print(df["bucket"].value_counts().to_dict())
    else:
        print(bucket_counts)


if __name__ == "__main__":
    main()