from __future__ import annotations

try:
	from extractor_defaults_core import DEPARTMENT_SUFFIXES
except ImportError:  # pragma: no cover
	from src.extractor_defaults_core import DEPARTMENT_SUFFIXES

SPACY_SIGNAL_DEPT_PATTERN = rf"([가-힣A-Za-z0-9\u0600-\u06FF][가-힣A-Za-z0-9\u0600-\u06FF\s\-]{{0,40}}{DEPARTMENT_SUFFIXES})"
SPACY_SIGNAL_DEPARTMENT_SUFFIX_PATTERN_JA = r"(部|チーム|本部|センター)$"
SPACY_SIGNAL_JA_PERSON_PREFIXES = ["までの", "からの", "の"]
SPACY_SIGNAL_PERSON_TIME_SUFFIX_PATTERN = r"(년|월|일|시|분|초)$"
SPACY_SIGNAL_LATIN_TIME_SUFFIX_PATTERN = r"(day|month|year|hour|minute|second)$"
