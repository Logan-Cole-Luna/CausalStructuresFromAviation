"""
Traditional NLP methods for causal chain extraction from aviation accident narratives.

Method 1: Rule-based pattern matching using regex.
Method 2: spaCy dependency parsing.
"""
import re
from typing import List, Optional

# Relative pronouns / conjunctions that make poor standalone spans
_RELATIVE_WORDS = {'which', 'that', 'this', 'these', 'those', 'it', 'he', 'she',
                   'they', 'who', 'whom', 'where', 'when', 'what', 'there'}

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Causal connective patterns
# ---------------------------------------------------------------------------

# Forward patterns: "CAUSE <connective> EFFECT"
CAUSAL_FORWARD = [
    "caused",
    "resulted in",
    "led to",
    "triggered",
    "produced",
]

# Backward patterns: "EFFECT <connective> CAUSE"
CAUSAL_BACKWARD = [
    "due to",
    "because of",
    "caused by",
    "attributed to",
    "as a result of",
    "resulting from",
    "stemmed from",
    "contributed to",
]

# Pre-compile patterns (longer phrases first to avoid partial matches)
_ALL_PATTERNS = sorted(CAUSAL_FORWARD + CAUSAL_BACKWARD, key=len, reverse=True)
_PATTERN_RE = {
    p: re.compile(r'\b' + re.escape(p) + r'\b', re.IGNORECASE)
    for p in _ALL_PATTERNS
}

# Sentence splitter
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation."""
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _trim_span(span: str, max_chars: int = 200) -> str:
    """
    Trim a span to max_chars, then find the nearest clause boundary
    ('. ' or ', ') as the final trim point.
    """
    span = span.strip()
    if len(span) <= max_chars:
        return span
    span = span[:max_chars]
    # Find last clause boundary within the truncated span
    last_period = span.rfind('. ')
    last_comma = span.rfind(', ')
    boundary = max(last_period, last_comma)
    if boundary > 0:
        span = span[:boundary]
    return span.strip()


def _trim_span_from_end(span: str, max_chars: int = 200) -> str:
    """
    Take the last max_chars of span, then find the nearest clause boundary
    from the start as the final trim point (for 'last N words' style).
    """
    span = span.strip()
    if len(span) <= max_chars:
        return span
    span = span[-max_chars:]
    # Find first clause boundary from start
    first_period = span.find('. ')
    first_comma = span.find(', ')
    candidates = [x for x in [first_period, first_comma] if x >= 0]
    if candidates:
        boundary = min(candidates) + 2  # skip past the delimiter
        span = span[boundary:]
    return span.strip()


# ---------------------------------------------------------------------------
# Method 1: Rule-based pattern matching
# ---------------------------------------------------------------------------

def extract_by_rules(text: str) -> List[dict]:
    """
    Apply forward/backward causal patterns to each sentence in text.
    Returns list of dicts: {cause, relation, effect, direction, sentence}.
    """
    results = []
    sentences = _split_sentences(text)

    for sentence in sentences:
        for pattern in CAUSAL_FORWARD:
            regex = _PATTERN_RE[pattern]
            for m in regex.finditer(sentence):
                before = sentence[:m.start()]
                after = sentence[m.end():]
                cause = _trim_span_from_end(before, max_chars=250)
                effect = _trim_span(after, max_chars=250)
                # Final word-count limit
                cause = ' '.join(cause.split()[-20:]).strip().strip('.,;: ')
                effect = ' '.join(effect.split()[:20]).strip().strip('.,;: ')
                if not _is_valid_span(cause) or not _is_valid_span(effect):
                    continue
                results.append({
                    'cause': cause,
                    'relation': pattern,
                    'effect': effect,
                    'direction': 'forward',
                    'sentence': sentence,
                })

        for pattern in CAUSAL_BACKWARD:
            regex = _PATTERN_RE[pattern]
            for m in regex.finditer(sentence):
                before = sentence[:m.start()]
                after = sentence[m.end():]
                # Before the connective is the EFFECT; after is the CAUSE
                effect = _trim_span_from_end(before, max_chars=250)
                cause = _trim_span(after, max_chars=250)
                effect = ' '.join(effect.split()[-20:]).strip().strip('.,;: ')
                cause = ' '.join(cause.split()[:20]).strip().strip('.,;: ')
                if not _is_valid_span(cause) or not _is_valid_span(effect):
                    continue
                results.append({
                    'cause': cause,
                    'relation': pattern,
                    'effect': effect,
                    'direction': 'backward',
                    'sentence': sentence,
                })

    return results


# ---------------------------------------------------------------------------
# Method 2: spaCy dependency parsing
# ---------------------------------------------------------------------------

_CAUSAL_VERBS = {'cause', 'result', 'lead', 'trigger', 'contribute'}


def _is_valid_span(span: str) -> bool:
    """Return False if span is empty, a pronoun/relative word, or fewer than 2 words."""
    if not span or not span.strip():
        return False
    words = span.strip().split()
    if len(words) < 2:
        return False
    if words[0].lower() in _RELATIVE_WORDS and len(words) == 1:
        return False
    return True


def load_nlp():
    """
    Load en_core_web_sm spaCy model.  Returns None if spaCy or the model
    is not installed rather than raising an exception.
    """
    try:
        import spacy  # noqa: F401
    except ImportError:
        print("[NLP] spaCy not installed — dependency parsing unavailable.")
        return None

    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        return nlp
    except OSError:
        print("[NLP] en_core_web_sm not found — run: python -m spacy download en_core_web_sm")
        return None


def extract_by_deps(text: str, nlp) -> List[dict]:
    """
    Use spaCy dependency parsing to extract causal triples.

    For each sentence, find verbs whose lemma is in CAUSAL_VERBS.
    Extract nsubj as the cause and dobj or the first prep child as the effect.
    Returns list of {cause, relation, effect, method='deps'}.
    """
    if nlp is None:
        return []

    doc = nlp(text)
    results = []

    for sent in doc.sents:
        for token in sent:
            if token.pos_ == 'VERB' and token.lemma_.lower() in _CAUSAL_VERBS:
                cause = None
                effect = None

                for child in token.children:
                    if child.dep_ == 'nsubj':
                        # Expand to the full noun phrase
                        cause = ' '.join(t.text for t in child.subtree).strip()
                    elif child.dep_ == 'dobj':
                        effect = ' '.join(t.text for t in child.subtree).strip()
                    elif child.dep_ == 'prep' and effect is None:
                        effect = ' '.join(t.text for t in child.subtree).strip()

                if _is_valid_span(cause) and _is_valid_span(effect):
                    results.append({
                        'cause': cause,
                        'relation': token.lemma_,
                        'effect': effect,
                        'direction': 'forward',
                        'sentence': sent.text,
                        'method': 'deps',
                    })

    return results


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------

def extract_causal_triples(text: str, nlp=None) -> List[dict]:
    """
    Combine rule-based and dependency-parsing extraction, deduplicate,
    and return the merged list.
    """
    rule_results = extract_by_rules(text)
    for r in rule_results:
        r.setdefault('method', 'rules')

    dep_results = []
    if nlp is not None:
        dep_results = extract_by_deps(text, nlp)

    combined = rule_results + dep_results

    # Deduplicate on (cause, relation, effect) ignoring case/whitespace
    seen = set()
    deduped = []
    for item in combined:
        key = (
            item.get('cause', '').lower().strip(),
            item.get('relation', '').lower().strip(),
            item.get('effect', '').lower().strip(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def batch_extract(
    df: pd.DataFrame,
    nlp=None,
    sample_n: Optional[int] = None,
) -> List[dict]:
    """
    Apply extract_causal_triples to each row of df['narr_clean'].
    Attach ev_id to every triple.  Returns flat list of all triples.
    """
    if 'narr_clean' not in df.columns:
        raise ValueError("DataFrame must have a 'narr_clean' column — run preprocess_data first.")

    if sample_n is not None:
        df = df.head(sample_n)

    all_triples = []
    ev_ids = df['ev_id'].tolist() if 'ev_id' in df.columns else [None] * len(df)
    texts = df['narr_clean'].tolist()

    for ev_id, text in tqdm(zip(ev_ids, texts), total=len(texts), desc="Extracting causal triples"):
        if not isinstance(text, str) or not text.strip():
            continue
        triples = extract_causal_triples(text, nlp=nlp)
        for triple in triples:
            triple['ev_id'] = ev_id
        all_triples.extend(triples)

    return all_triples
