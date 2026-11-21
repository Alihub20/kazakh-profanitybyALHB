import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
MENTION_RE = re.compile(r"@\w+")

MAX_CHARS = 2000  # long comments are cut (profanity is almost always visible early)


def remove_personal_info(text: str) -> str:
    # Remove links, e-mails, phone numbers and @usernames.
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    # Normalization applied to every comment before it becomes model input.
    t = str(text).lower()[:MAX_CHARS]
    t = t.replace("ё", "е")
    t = re.sub(r"[*@#$%]+", "*", t)          # masking symbols -> one '*'
    t = re.sub(r"\d", "0", t)                 # all digits -> 0
    t = re.sub(r"(\w)\1{2,}", r"\1", t)       # бооооқ -> боқ
    t = re.sub(r"[^\w\s*]", " ", t)           # punctuation -> space (keep '*')
    t = re.sub(r"_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def dedup_key(text: str) -> str:
    # Aggressive key used to detect near-duplicates (only letters).
    return re.sub(r"[^a-zа-яәіңғүұқөһ]", "", normalize(text))
