import re
from textprep import normalize

PATTERNS = {
    "sik":        r"\bс[іи]к(?:ті|ти|тім|тим|тің|тір|тир|ейін|ейин|кен|іп|ип|у)?\b",
    "sigeyin":    r"\bс[іи]гей[іи]н\b",
    "qotaq":      r"\b[қк]ота[қк]\w*",
    "boq":        r"\bбоқ\w*|\bбок(?:тай|ты|тан)\b",
    "kot":        r"\bкөт(?:ің|іңді|іңе|ке|тен|ім|ін|іне)?\b",
    "am":         r"\bам(?:ың|ыңды|ыңа|ыңның|ыңнан|ын|ына)\b",
    "zhezokshe":  r"\b[жз]ез[өо]кше\w*",
    "curse_auz":  r"\b[әа]кең(?:н[іи]ң)?\s+аузын\b|\bаузын\s+[ұу]райын\b",
    "it_balasy":  r"\bиттің\s+баласы\b",
    "latin":      r"\b(?:sik(?:ti|tim|tin|eyin)?|sigeyin|qotaq|kotak|boq|zhezokshe|jezokshe|amyndy)\b",
    "masked":     r"\bс\*+[кг]\w*|\b[қк]\*+та[қк]\w*|\b[қк]о\*+\w*|\bб\*+[қк]\w*|\bбо\*+\w*|\bк\*+т[іi]\w*",
}
COMPILED = {name: re.compile(p) for name, p in PATTERNS.items()}


def lexicon_hits(text: str) -> list:
    t = normalize(text)
    return [name for name, rx in COMPILED.items() if rx.search(t)]


def lexicon_label(text: str) -> int:
    return int(len(lexicon_hits(text)) > 0)
