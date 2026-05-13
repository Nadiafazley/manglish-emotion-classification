import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "threads.csv"
OUTPUT_CSV = BASE_DIR / "Threads_Manglish_Clean_Labeled.csv"
OUTPUT_JSON = BASE_DIR / "Threads_Manglish_Clean_Labeled.json"


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ROMAN_LANGUAGE_RE = re.compile(
    "["
    "\u0900-\u097F"  # Devanagari
    "\u0980-\u09FF"  # Bengali
    "\u0B80-\u0BFF"  # Tamil
    "\u0E00-\u0E7F"  # Thai
    "\u3040-\u30FF"  # Japanese
    "\u3400-\u9FFF"  # Chinese
    "\uAC00-\uD7AF"  # Korean
    "\u0600-\u06FF"  # Arabic/Jawi
    "]"
)
ACCENTED_LATIN_RE = re.compile(r"[À-ÖØ-öø-ÿĀ-ž]")
MOJIBAKE_EMOJI_RE = re.compile(r"(?:ðŸ\S*|â[€œ˜™\S]*)")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
SENTENCE_SPLIT_RE = re.compile(r"[.!?;|\n]+")


MALAY_MANGlish_TERMS = {
    "aku", "akak", "abang", "adik", "awak", "bahagia", "baju", "banyak",
    "baru", "benci", "benda", "betul", "bila", "boleh", "bosan", "bro",
    "budak", "bukan", "cinta", "cuma", "dah", "dalam", "dekat", "dengan",
    "dia", "doakan", "gembira", "geram", "gila", "hensem", "ibu", "jangan",
    "jatuh", "je", "jerawat", "jom", "jujur", "kak", "kalau", "kan",
    "kecewa", "kenapa", "kerja", "kita", "kot", "kurang", "la", "lagi",
    "lah", "lelaki", "makan", "makin", "malu", "mana", "marah", "macam",
    "menangis", "menjadi", "nak", "ngam", "ni", "orang", "pakai", "paling",
    "perempuan", "pun", "pulak", "rindu", "risau", "rumah", "sakit", "sama",
    "sangat", "saya", "sebab", "sedih", "selamat", "seronok", "siapa",
    "sikit", "sis", "tak", "takut", "tapi", "teruk", "tidak", "tidur",
    "tu", "we", "weh", "woi", "ya", "yang", "yee", "yg",
}

ENGLISH_MARKERS = {
    "i", "me", "my", "you", "your", "we", "our", "they", "their", "the",
    "a", "an", "and", "but", "for", "with", "to", "from", "in", "on", "is",
    "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "can", "could", "should", "not", "dont", "don't",
}

NORMALIZATION_MAP = {
    "xde": "takde",
    "takde": "tak ada",
    "tdk": "tidak",
    "taknak": "tak nak",
    "dgn": "dengan",
    "yg": "yang",
    "sy": "saya",
    "aq": "aku",
    "ak": "aku",
    "u": "you",
    "ur": "your",
    "dont": "don't",
    "cant": "can't",
    "im": "i'm",
}

EMOTION_KEYWORDS = {
    "joy": {
        "happy", "happier", "happiness", "gembira", "bahagia", "seronok",
        "best", "excited", "love", "menjadi", "syukur", "alhamdulillah",
        "cantik", "comel", "hebat", "bagus", "win", "menang", "hijau",
        "haha", "hahaha", "lol",
    },
    "sadness": {
        "sad", "sedih", "kecewa", "down", "cry", "crying", "menangis",
        "rindu", "sakit", "penat", "tired", "hurt", "hancur", "sunyi",
        "lonely", "hilang", "miss", "missing", "teruk",
    },
    "anger": {
        "angry", "marah", "benci", "geram", "annoyed", "bengang", "fckin",
        "fuck", "fucking", "bodoh", "stupid", "hate", "cancel", "sampah",
        "vandalism",
    },
    "fear": {
        "takut", "scared", "fear", "worry", "worried", "risau", "cemas",
        "anxiety", "anxious", "panic", "seram", "trauma",
    },
    "surprise": {
        "wow", "damn", "omg", "terkejut", "surprise", "tak sangka",
    },
    "love": {
        "sayang", "love", "cinta", "nikah", "married", "kahwin", "ibu",
        "home", "warm", "dear", "heart",
    },
    "disgust": {
        "jijik", "geli", "gross", "disgusting", "raba", "creepy", "cringe",
    },
}

NEGATIVE_PHRASES = {
    "not happy": "sadness",
    "tak happy": "sadness",
    "tak suka": "anger",
    "don't like": "anger",
    "dont like": "anger",
}


def read_threads(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["text"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    df["text"] = df["text"].astype(str).str.strip()
    return df


def clean_text(text: str, normalize: bool = True) -> str:
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = text.lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if normalize:
        for source, target in NORMALIZATION_MAP.items():
            text = re.sub(rf"\b{re.escape(source)}\b", target, text)

    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def has_code_mixed_sentence(text: str) -> bool:
    for sentence in SENTENCE_SPLIT_RE.split(text):
        sentence_tokens = set(tokenize(clean_text(sentence, normalize=False)))
        if sentence_tokens & MALAY_MANGlish_TERMS and sentence_tokens & ENGLISH_MARKERS:
            return True
    return False


def is_manglish_or_malay(text: str, tokens: list[str]) -> tuple[bool, bool, int]:
    if not tokens or NON_ROMAN_LANGUAGE_RE.search(text):
        return False, False, 0

    text_without_mojibake_emoji = MOJIBAKE_EMOJI_RE.sub("", text)
    if ACCENTED_LATIN_RE.search(text_without_mojibake_emoji):
        return False, False, 0

    token_set = set(tokens)
    manglish_hits = token_set & MALAY_MANGlish_TERMS
    english_hits = token_set & ENGLISH_MARKERS
    code_mixed = bool(manglish_hits and english_hits and has_code_mixed_sentence(text))

    # Keep only true code-mixed rows: at least one Malay/Manglish marker and
    # at least one English marker in the same sentence/segment.
    keep = code_mixed
    return keep, code_mixed, len(manglish_hits)


def label_emotion(cleaned: str, tokens: list[str]) -> tuple[str, str, str]:
    scores = Counter()
    token_set = set(tokens)

    for phrase, emotion in NEGATIVE_PHRASES.items():
        if phrase in cleaned:
            scores[emotion] += 2

    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if " " in keyword:
                if keyword in cleaned:
                    scores[emotion] += 2
            elif keyword in token_set:
                scores[emotion] += 1

    if not scores:
        return "neutral", "", json.dumps({}, sort_keys=True)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    primary = ranked[0][0]
    secondary = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0 else ""
    return primary, secondary, json.dumps(dict(sorted(scores.items())), sort_keys=True)


def preprocess() -> pd.DataFrame:
    df = read_threads(SOURCE_FILE)
    df = df[df["text"].str.strip().ne("")]
    df = df.drop_duplicates(subset=["text"], keep="first").copy()

    df["language_check_text"] = df["text"].apply(lambda value: clean_text(value, normalize=False))
    df["language_check_tokens"] = df["language_check_text"].apply(tokenize)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip().ne("")]
    df["tokens"] = df["clean_text"].apply(tokenize)

    manglish_info = df.apply(
        lambda row: is_manglish_or_malay(row["text"], row["language_check_tokens"]),
        axis=1,
        result_type="expand",
    )
    manglish_info.columns = ["is_manglish_data", "code_mixed", "manglish_score"]
    df = pd.concat([df, manglish_info], axis=1)
    df = df[df["is_manglish_data"]].copy()

    labels = df.apply(
        lambda row: label_emotion(row["clean_text"], row["tokens"]),
        axis=1,
        result_type="expand",
    )
    labels.columns = ["emotion_label", "secondary_emotion", "emotion_scores"]
    df = pd.concat([df, labels], axis=1)

    df["tokens"] = df["tokens"].apply(lambda values: " ".join(values))
    df = df[
        [
            "text",
            "clean_text",
            "tokens",
            "is_manglish_data",
            "code_mixed",
            "manglish_score",
            "emotion_label",
            "secondary_emotion",
            "emotion_scores",
        ]
    ].reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    json_records = df.copy()
    json_records["tokens"] = json_records["tokens"].apply(lambda value: value.split())
    json_records["emotion_scores"] = json_records["emotion_scores"].apply(json.loads)
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(json_records.to_dict(orient="records"), file, ensure_ascii=False, indent=2)

    return df


if __name__ == "__main__":
    result = preprocess()
    print(f"Preprocessing complete: {len(result)} English-Malay code-mixed rows saved.")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")
    print("Emotion distribution:")
    print(result["emotion_label"].value_counts().to_string())
