# backend/generate_quiz.py
"""
generate_quiz.py — MCQ v2 (strict, high-quality)

Purpose:
- Produce fewer but cleaner MCQs from lecture text.
- Strictly filter candidate sentences and key phrases.
- Only generate a question when a meaningful correct answer can be produced.
- Produce grammatical, context-aware distractors (no one-word junk).
- Prevent repeated keys in a single generated set.
- Keep the exact output schema required by the frontend:
  {
    "id": "...",
    "question": "...",
    "choices": { "A": "...", "B": "...", "C": "...", "D": "..." },
    "answer": "C"
  }

Notes:
- This is a single-file drop-in replacement for backend/generate_quiz.py.
- No external libraries required.
- Includes a small `__main__` demo to sanity-check generation locally.
"""

import json
import time
import random
import re
from typing import List, Dict, Callable, Optional

# ---------------------------
# Legacy-compatible function
# ---------------------------
MAX_RETRIES = 2
BACKOFF_BASE = 0.5


def generate_quiz_from_context(stem: str, context_text: str, n: int = 5,
                               llm_call: Optional[Callable[[str], str]] = None,
                               retries: int = MAX_RETRIES) -> Dict:
    """
    Backwards-compatible API expected by older code.
    If llm_call is None -> deterministic placeholders.
    If llm_call provided -> calls LLM and normalizes JSON.
    """
    if llm_call is None:
        quiz = {"stem": stem, "questions": []}
        for i in range(n):
            quiz["questions"].append({
                "id": f"{stem}_q{i+1}",
                "question": f"Placeholder question {i+1} about {stem}",
                "answer": f"Placeholder answer {i+1}",
                "difficulty": "medium"
            })
        return quiz

    prompt = (
        f"Create {n} short quiz items (question + answer) from the following context.\n\n"
        "Return valid JSON with a top-level key 'questions' which is a list of objects "
        "each having: id (string), question (string), answer (string), difficulty (easy|medium|hard).\n\n"
        f"Context:\n{context_text}\n\nReturn JSON ONLY."
    )

    attempt = 0
    last_exc = None
    while attempt <= retries:
        try:
            resp = llm_call(prompt)
            if isinstance(resp, dict):
                parsed = resp
            elif isinstance(resp, str):
                parsed = json.loads(resp)
            else:
                raise ValueError("llm_call returned unsupported type")
            questions = parsed.get("questions")
            if not isinstance(questions, list):
                raise ValueError("Response missing 'questions' list")
            if len(questions) > n:
                questions = questions[:n]
            out_qs = []
            for i, q in enumerate(questions):
                if not isinstance(q, dict):
                    continue
                out_qs.append({
                    "id": q.get("id") or f"{stem}_q{i+1}",
                    "question": q.get("question") or q.get("prompt") or "",
                    "answer": q.get("answer") or q.get("solution") or "",
                    "difficulty": q.get("difficulty") or "medium"
                })
            return {"stem": stem, "questions": out_qs}
        except Exception as e:
            last_exc = e
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"LLM quiz generation failed after {retries} retries: {e}")
            sleep = BACKOFF_BASE * (2 ** (attempt - 1))
            time.sleep(sleep)
    raise RuntimeError(f"LLM quiz generation failed: {last_exc}")


# ---------------------------
# MCQ v2 internals
# ---------------------------
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "these", "those", "from", "which", "what",
    "when", "where", "were", "have", "has", "had", "are", "is", "was", "been", "but", "not",
    "their", "they", "them", "can", "could", "would", "should", "will", "shall", "about", "into",
    "over", "under", "between", "within", "also", "such", "other", "than", "then", "there", "here",
    "you", "your", "our", "its", "it's", "i", "we", "me", "a", "an", "of", "by", "to", "as", "on"
}
_PRONOUNS = {"you", "your", "yours", "it", "its", "they", "them", "their", "he", "she", "we", "i", "me", "us"}
_BAD_STARTS = {"when", "for", "if", "because", "while", "could", "would", "should", "may", "might",
               "does", "do", "did", "is", "are", "was", "were", "how", "why", "what"}

# blacklist of tokens (strict)
_KEY_BLACKLIST_TOKENS = {
    "shorter", "short", "recent", "recently", "very", "need", "needs", "book",
    "chapter", "article", "articles", "section", "sections", "pages", "version",
    "versions", "introduction", "summary", "overview", "conclusion", "concept", "description", "full"
}

_STEMS_KEY = [
    "Which statement best defines {key}?",
    "What does {key} mean in this context?",
    "According to the passage, what is the role of {key}?",
    "Which option best describes the purpose of {key}?"
]
_STEMS_SNIPPET = [
    "What is the main idea of the following sentence: \"{snippet}\"?",
    "Which option best explains the sentence: \"{snippet}\"?"
]

_DISTRACTOR_TEMPLATES = [
    "{concept} is an unrelated concept mentioned elsewhere in the lecture.",
    "{concept} refers to a different topic discussed in the text.",
    "{concept} is an example rather than a definition.",
    "{concept} describes a separate process not asked here."
]

_GENERIC_FILLERS = [
    "A process for validating data quality.",
    "A technique to anonymize or protect user information.",
    "An organizational role responsible for data collection."
]


def _shorten_statement(s: str, max_len: int = 180) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    cut = s[:max_len]
    last_p = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_p > max_len - 60:
        return cut[:last_p+1].strip()
    return cut[:max_len-3].rstrip() + "..."


def _token_overlap_fraction(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    return len(inter) / max(len(ta), len(tb))


def _is_fragment_like(s: str) -> bool:
    if not s or len(s.strip()) < 3:
        return True
    first = re.match(r'^\s*([A-Za-z\'-]+)', s)
    if first:
        w = first.group(1).lower()
        if w in _BAD_STARTS or w in _PRONOUNS:
            if len(s.split()) < 6:
                return True
    return False


def _words_from_text(text: str) -> List[str]:
    found = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", (text or ""))
    out = []
    seen = set()
    for w in found:
        lw = w.strip()
        if lw.lower() in _STOPWORDS:
            continue
        if lw.lower() in seen:
            continue
        seen.add(lw.lower())
        out.append(lw)
    return out


# ---------------------------
# STEP 1: Strict key extraction
# ---------------------------
def _extract_key_phrase(sentence: str) -> Optional[str]:
    """
    STRICT rules:
      - Reject very short sentences (< 8 words).
      - Prefer determiner-led noun phrases or subject-before-copula.
      - Prefer capitalized multi-word terms.
      - FALLBACK: only accept capitalized 1-2 word phrases.
      - Before returning candidate, require at least 2 words (len >= 2).
      - Candidate tokens should be noun-like (not gerunds/verbs/adjectives only).
    """
    if not sentence or not sentence.strip():
        return None
    if len(sentence.split()) < 8:
        return None

    s = sentence.strip()

    def _candidate_is_noun_like(cand: str) -> bool:
        toks = [t for t in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", cand)]
        if len(toks) < 2:
            return False
        # require at least one token that is length >=4 and not in stopwords or blacklist
        good = [t for t in toks if len(t) >= 4 and t.lower() not in _STOPWORDS and t.lower() not in _KEY_BLACKLIST_TOKENS]
        if not good:
            return False
        # avoid tokens that look like verbs (ending in 'ing', 'ed') for all tokens
        if all(t.lower().endswith(("ing", "ed")) for t in toks):
            return False
        return True

    # 1) determiner-led noun phrase: "the X Y", "a middleware layer"
    det_re = re.compile(r'\b(?:the|a|an)\s+([A-Za-z][A-Za-z\'-]{2,}(?:\s+[A-Za-z][A-Za-z\'-]{2,}){0,2})', flags=re.IGNORECASE)
    for m in det_re.finditer(s):
        cand = m.group(1).strip()
        if cand and not _is_fragment_like(cand) and cand.lower() not in _KEY_BLACKLIST_TOKENS and _candidate_is_noun_like(cand):
            return cand

    # 2) subject before copula "X is ..." / "X provides ..."
    cop_re = re.compile(
        r'([A-Za-z][A-Za-z\'-]{2,}(?:\s+[A-Za-z][A-Za-z\'-]{2,}){0,2})\s+'
        r'(?:is|are|was|were|provides|offers|refers to|refers|involves|helps|aims to|serves as|acts as)\b',
        flags=re.IGNORECASE
    )
    m = cop_re.search(s)
    if m:
        cand = m.group(1).strip()
        if cand and not _is_fragment_like(cand) and cand.lower() not in _KEY_BLACKLIST_TOKENS and _candidate_is_noun_like(cand):
            return cand

    # 3) capitalized multi-word phrases (proper nouns / named concepts)
    cap_re = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b')
    for m in cap_re.finditer(s):
        cand = m.group(1).strip()
        if cand and not _is_fragment_like(cand) and len(cand.split()) >= 2 and cand.lower() not in _KEY_BLACKLIST_TOKENS and _candidate_is_noun_like(cand):
            return cand

    # 4) fallback: accept only capitalized 1-2 word phrases
    fallback_tokens = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", s)
    for t in fallback_tokens:
        if t and not _is_fragment_like(t) and len(t.split()) >= 2 and t.lower() not in _KEY_BLACKLIST_TOKENS and _candidate_is_noun_like(t):
            return t

    return None


# ---------------------------
# STEP 2: Rephrase correct answer BUT return None if not meaningful
# ---------------------------
def _clause_containing_key(sentence: str, key: Optional[str]) -> str:
    if not key:
        return sentence or ""
    parts = re.split(r'[;:,\-\—]|\s+which\s+|\s+that\s+', sentence or "", flags=re.IGNORECASE)
    for p in parts:
        if re.search(re.escape(key), p, flags=re.IGNORECASE):
            return p.strip()
    return sentence or ""


def _rephrase_correct_answer(sentence: str, key: Optional[str]) -> Optional[str]:
    """
    Produce a concise correct answer when predicate/object are clear.
    Return None if we cannot produce a meaningful answer (this causes the generator to skip).
    Requirements:
      - candidate must contain predicate and object with at least ~6 tokens total.
      - object must include at least one noun-like token (length >=4, not stopword/blacklist).
    """
    if not key:
        return None
    clause = _clause_containing_key(sentence, key)
    clause = (clause or "").strip().rstrip('.')

    # try copula/predicate approach
    m = re.search(r'\b(is|are|was|were|refers to|refers|means|provides|offers|helps|enables|aims to|involves|serves as|acts as|is used to|is a|is an)\b', clause, flags=re.IGNORECASE)
    if m:
        pred = clause[m.end():].strip(',;: ')
        verb = m.group(0).lower()
        if pred:
            pred_clean = re.sub(r'^\b(to|for|the|a|an)\b', '', pred, flags=re.IGNORECASE).strip()
            if not pred_clean:
                return None
            candidate = f"{key} {verb} {pred_clean}"
            candidate = re.sub(r'\s+', ' ', candidate).strip()
            toks = candidate.split()
            # require >= 6 tokens to avoid one-word/object outputs
            if len(toks) >= 6:
                # ensure object contains at least one noun-like token
                obj_tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", pred_clean)]
                noun_like = any(len(t) >= 4 and t.lower() not in _STOPWORDS and t.lower() not in _KEY_BLACKLIST_TOKENS and not t.lower().endswith(("ing", "ed")) for t in obj_tokens)
                if noun_like:
                    return _shorten_statement(candidate + ".", max_len=180)
    # else: refuse to guess — return None per STEP 2
    return None


# ---------------------------
# STEP 3: Distractor generation (avoid broken templates and single-token inserts)
# ---------------------------
def _collect_related_concepts(context_text: str, sentence: str, key: Optional[str], max_candidates: int = 8) -> List[str]:
    """
    Collect nearby multi-word concepts or strong tokens. Only return candidates that
    are multi-word (>=2 tokens) OR single tokens that are long and clearly noun-like.
    """
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (context_text or "")) if s.strip()]
    if not sents:
        return []
    idx = None
    for i, s in enumerate(sents):
        if sentence and (sentence[:40] in s or s in sentence or sentence in s):
            idx = i
            break
    if idx is None:
        idx = 0
    candidates = []
    for j in range(max(0, idx-1), min(len(sents), idx+2)):
        s = sents[j]
        for m in re.finditer(r'\b(?:the|a|an)\s+([A-Za-z][A-Za-z\'-]{2,}(?:\s+[A-Za-z][A-Za-z\'-]{2,}){0,2})', s, flags=re.IGNORECASE):
            ph = m.group(1).strip()
            if ph and ph.lower() != (key or "").lower() and not _is_fragment_like(ph):
                candidates.append(ph)
        for t in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", s):
            if t.lower() in _STOPWORDS or (key and t.lower() == key.lower()):
                continue
            if not _is_fragment_like(t):
                candidates.append(t)
    # global words
    for w in _words_from_text(context_text):
        if key and w.lower() == key.lower():
            continue
        if not _is_fragment_like(w):
            candidates.append(w)

    out = []
    seen = set()
    for c in candidates:
        cl = c.strip()
        low = cl.lower()
        if low in seen:
            continue
        toks = cl.split()
        # accept if multiword >=2 OR single long noun-like token
        if len(toks) >= 2:
            pass
        else:
            t = toks[0]
            if len(t) < 4 or t.lower() in _KEY_BLACKLIST_TOKENS:
                continue
        if len(cl.split()) > 4:
            continue
        seen.add(low)
        out.append(cl)
        if len(out) >= max_candidates:
            break
    return out


def _generate_plausible_distractors(correct: str, related: List[str], key: Optional[str]) -> List[str]:
    """
    Create up to 3 distractors from multi-word related concepts first, then safe generic fillers.
    Avoid inserting single token junk and avoid templates that produce broken grammar.
    """
    distractors: List[str] = []
    used = set()

    # Use related phrases (prefer multi-word)
    related_sorted = sorted(related, key=lambda x: (-len(x.split()), x))  # prefer multiword
    for r in related_sorted:
        if len(distractors) >= 3:
            break
        if not r:
            continue
        # ensure we don't insert single meaningless tokens
        if len(r.split()) < 2 and len(r) < 6:
            continue
        if _token_overlap_fraction(r, correct) > 0.6:
            continue
        t = random.choice(_DISTRACTOR_TEMPLATES)
        cand = t.replace("{concept}", r).strip()
        if not cand.endswith("."):
            cand = cand + "."
        if _token_overlap_fraction(cand, correct) > 0.75:
            continue
        if cand.lower() in used:
            continue
        distractors.append(_shorten_statement(cand, max_len=160))
        used.add(cand.lower())

    # fallback generics
    for g in _GENERIC_FILLERS:
        if len(distractors) >= 3:
            break
        if _token_overlap_fraction(g, correct) > 0.75:
            continue
        if g.lower() in used:
            continue
        distractors.append(g)
        used.add(g.lower())

    # last-resort filler
    while len(distractors) < 3:
        distractors.append("This statement is not supported by the lecture text.")
    return distractors[:3]


# ---------------------------
# Snippet-based fallback (used rarely; still strict)
# ---------------------------
def _snippet_based_question_and_choices(sentence: str, context_text: str):
    """
    Build a snippet-centered Q + correct + distractors.
    This is used only when key-based extraction isn't appropriate; still must satisfy strict filters.
    """
    correct = _shorten_statement(sentence.strip().rstrip('.'), max_len=180)
    if not correct.endswith('.'):
        correct = correct + '.'

    # gather other candidate sentences as distractors (shorten them)
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (context_text or "")) if s.strip()]
    distractor_snips = []
    for s in sents:
        if s.strip() == sentence.strip():
            continue
        if len(distractor_snips) >= 3:
            break
        if len(s.split()) < 8:
            continue
        sn = _shorten_statement(s.strip().rstrip('.'), max_len=160)
        if sn and sn.lower() != correct.lower() and _token_overlap_fraction(sn, correct) < 0.7:
            if not sn.endswith('.'):
                sn = sn + '.'
            distractor_snips.append(sn)
    # if not enough sentence-distractors, use safe generics
    gi = 0
    while len(distractor_snips) < 3 and gi < len(_GENERIC_FILLERS):
        cand = _GENERIC_FILLERS[gi]
        gi += 1
        if _token_overlap_fraction(cand, correct) < 0.8:
            distractor_snips.append(cand)
    while len(distractor_snips) < 3:
        distractor_snips.append("This statement is not supported by the lecture text.")
    snippet_short = _shorten_statement(sentence.strip().rstrip('.'), max_len=120)
    stem = random.choice(_STEMS_SNIPPET).format(snippet=snippet_short)
    return stem, correct, distractor_snips[:3]


# ---------------------------
# Main MCQ generator (strict)
# ---------------------------
def generate_mcq_from_context(context_text: str, n: int = 5) -> List[Dict]:
    """
    Generate up to n high-quality MCQs. It's acceptable to return fewer than n if strict rules block poor candidates.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (context_text or "")) if s.strip()]
    candidates = []
    for s in sentences:
        if len(s.split()) < 8:
            continue
        if _is_fragment_like(s.split('.')[0]):
            continue
        candidates.append(s)
    if not candidates:
        return []

    global_words = _words_from_text(context_text)
    if len(global_words) < 8:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", (context_text or ""))
        for t in tokens:
            if t.lower() not in _STOPWORDS and t not in global_words:
                global_words.append(t)
            if len(global_words) >= 12:
                break

    out_questions: List[Dict] = []
    used_keys = set()
    attempts = 0
    max_attempts = max(80, n * 12)
    created = 0

    while created < n and attempts < max_attempts:
        attempts += 1
        sent = random.choice(candidates)

        # avoid repeats of the same source sentence
        if any(q.get("question_source_sentence") == sent for q in out_questions) and random.random() < 0.7:
            continue

        key = _extract_key_phrase(sent)
        if not key:
            # strict: skip sentences without a strong key
            continue

        # avoid duplicate keys
        if key.lower() in used_keys:
            continue

        # build correct answer; must be meaningful and >= 6 tokens
        correct = _rephrase_correct_answer(sent, key)
        if not correct or len(correct.split()) < 6:
            # skip weak correct answers (STEP 3)
            continue

        related = _collect_related_concepts(context_text, sent, key, max_candidates=12)
        if not related:
            related = [w for w in global_words if not key or w.lower() != key.lower()][:8]

        distractors = _generate_plausible_distractors(correct, related, key)

        # assemble choices
        choices_list = [correct] + distractors[:3]

        # uniqueness and small qualifiers
        normalized = []
        seen = set()
        for ch in choices_list:
            chs = (ch or "").strip()
            kn = re.sub(r'\s+', ' ', chs.lower())
            if kn in seen:
                alt = chs.rstrip('.') + " (alternative)."
                if re.sub(r'\s+', ' ', alt.lower()) in seen:
                    alt = chs + " (variant)."
                normalized.append(alt)
                seen.add(re.sub(r'\s+', ' ', alt.lower()))
            else:
                normalized.append(chs)
                seen.add(kn)
        choices_list = normalized[:4]

        # fill to 4 if needed with safe generics
        if len(choices_list) < 4:
            for f in _GENERIC_FILLERS:
                if len(choices_list) >= 4:
                    break
                fk = re.sub(r'\s+', ' ', f.strip().lower())
                if fk not in seen:
                    choices_list.append(f)
                    seen.add(fk)

        # sanity checks
        bad = False
        for ch in choices_list:
            if not ch or len(ch.strip()) < 6:
                bad = True
                break
            if re.search(r'[^A-Za-z0-9\s,.\-()\'":;]', ch):
                bad = True
                break
        if bad or len(choices_list) < 4:
            continue

        random.shuffle(choices_list)
        labels = ["A", "B", "C", "D"]
        choices_dict = {lab: txt for lab, txt in zip(labels, choices_list)}

        # locate correct label
        correct_label = None
        for lab, txt in choices_dict.items():
            if txt.strip() == correct.strip():
                correct_label = lab
                break
        if correct_label is None:
            for lab, txt in choices_dict.items():
                if correct.strip().lower() in txt.strip().lower():
                    correct_label = lab
                    break
        if correct_label is None:
            slot = random.choice(labels)
            choices_dict[slot] = correct
            correct_label = slot

        # stem must be readable
        stem = random.choice(_STEMS_KEY).format(key=key)
        if _is_fragment_like(stem):
            continue

        used_keys.add(key.lower())

        out_questions.append({
            "id": f"q{created+1}",
            "question": stem,
            "choices": choices_dict,
            "answer": correct_label,
            "question_source_sentence": sent  # internal trace (will be stripped below)
        })
        created += 1

    # strip internals and return
    cleaned = []
    for q in out_questions:
        cleaned.append({
            "id": q["id"],
            "question": q["question"],
            "choices": q["choices"],
            "answer": q["answer"]
        })

    return cleaned


# ---------------------------
# Local sanity/demo runner
# ---------------------------
if __name__ == "__main__":
    # small demo: paste a short example text here to test
    demo_text = (
        "Information infrastructure is the set of systems and standards that enable data collection "
        "and sharing across urban departments. A middle-out approach seeks to connect institutional "
        "processes with local needs. Spatial data sets are more readily available and some cities have institutionalized data flows. "
        "The introduction chapter summarizes the concepts in the book. Built Heritage refers to the collection of historic structures."
    )
    print("Running generate_mcq_from_context demo (request 5 items)...\n")
    qs = generate_mcq_from_context(demo_text, n=5)
    print(f"Generated {len(qs)} questions:\n")
    for q in qs:
        print(f"ID: {q['id']}")
        print(q['question'])
        for L, txt in q['choices'].items():
            print(f"  {L}. {txt}")
        print("Answer:", q['answer'])
        print("-" * 60)
