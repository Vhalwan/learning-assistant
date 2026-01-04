# backend/generate_quiz.py
"""
Compatibility + improved MCQ generator.

Exports:
- generate_quiz_from_context(stem, context_text, n=5, llm_call=None, retries=2) -> dict
  (legacy/compat function, does not write files; if llm_call is None it returns deterministic placeholders)

- generate_mcq_from_context(context_text, n=5) -> List[Dict]
  (improved offline MCQ generator used by frontend; returns list of MCQs,
   each MCQ exactly follows the required schema:
   {
     "id": "q1",
     "question": "...",
     "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
     "answer": "B"
   }
)
"""

import json
import time
import random
import re
from typing import List, Dict, Callable, Optional

# ---------------------------
# Legacy compatibility: generate_quiz_from_context
# ---------------------------
MAX_RETRIES = 2
BACKOFF_BASE = 0.5


def generate_quiz_from_context(stem: str, context_text: str, n: int = 5,
                               llm_call: Optional[Callable[[str], str]] = None,
                               retries: int = MAX_RETRIES) -> Dict:
    """
    Backwards-compatible quiz generator kept for local / legacy usage.
    NOTE: This function does NOT write files to disk.

    Returns a dict: {"stem": <stem>, "questions": [ {id, question, answer, difficulty}, ... ] }

    - If llm_call is None: returns deterministic placeholders (safe for local runs).
    - If llm_call provided: calls it with a JSON-producing prompt and normalizes the result.
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
    # If exhausted
    raise RuntimeError(f"LLM quiz generation failed: {last_exc}")


# ---------------------------
# Improved offline MCQ generator
# ---------------------------

# small lexicons / heuristics
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

_KEY_BLACKLIST_TOKENS = {"shorter", "short", "recent", "recently", "very", "need", "needs", "book",
                         "chapter", "article", "articles", "section", "sections", "pages", "version",
                         "versions", "introduction", "summary", "overview", "conclusion"}

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

_ALTERNATIVE_ACTION_VERBS = [
    "protect", "anonymize", "evaluate", "collect", "govern", "normalize",
    "aggregate", "transform", "validate", "encrypt", "measure", "filter"
]


def _shorten(s: str, max_len: int = 200) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    cut = s[:max_len]
    last_p = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_p > max_len - 60:
        return cut[:last_p+1].strip()
    return cut[:max_len-3].rstrip() + "..."


def _token_overlap(a: str, b: str) -> float:
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


def _extract_key_phrase(sentence: str) -> Optional[str]:
    s = (sentence or "").strip()
    if not s:
        return None
    # determiner-based phrase
    det_re = re.compile(r'\b(?:the|a|an)\s+([A-Za-z][A-Za-z\'-]{2,}(?:\s+[A-Za-z][A-Za-z\'-]{2,}){0,2})', flags=re.IGNORECASE)
    for m in det_re.finditer(s):
        cand = m.group(1).strip()
        if not _is_fragment_like(cand) and cand.lower() not in _KEY_BLACKLIST_TOKENS:
            return cand
    # subject before copula
    cop_re = re.compile(r'([A-Za-z][A-Za-z\'-]{2,}(?:\s+[A-Za-z][A-Za-z\'-]{2,}){0,2})\s+(?:is|are|was|were|provides|offers|refers to|refers|involves|helps|aims to|serves as|acts as)\b', flags=re.IGNORECASE)
    m = cop_re.search(s)
    if m:
        cand = m.group(1).strip()
        if not _is_fragment_like(cand) and cand.lower() not in _KEY_BLACKLIST_TOKENS:
            return cand
    # capitalized multiword
    cap_re = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b')
    for m in cap_re.finditer(s):
        cand = m.group(1).strip()
        if not _is_fragment_like(cand) and len(cand.split()) <= 3 and cand.lower() not in _KEY_BLACKLIST_TOKENS:
            return cand
    # first non-stopword token or two-word sequence
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", s)
    if not tokens:
        return None
    for i in range(len(tokens)):
        t = tokens[i]
        if t.lower() in _STOPWORDS or t.lower() in _PRONOUNS or t.lower() in _KEY_BLACKLIST_TOKENS:
            continue
        if i + 1 < len(tokens) and tokens[i+1].lower() not in _STOPWORDS:
            cand = f"{t} {tokens[i+1]}"
            if not _is_fragment_like(cand) and cand.lower() not in _KEY_BLACKLIST_TOKENS:
                return cand
        if not _is_fragment_like(t) and t.lower() not in _KEY_BLACKLIST_TOKENS:
            return t
    return None


def _clause_for_key(sentence: str, key: Optional[str]) -> str:
    if not key:
        return sentence or ""
    parts = re.split(r'[;:,\-\—]|\s+which\s+|\s+that\s+', sentence or "", flags=re.IGNORECASE)
    for p in parts:
        if re.search(re.escape(key), p, flags=re.IGNORECASE):
            return p.strip()
    return sentence or ""


def _rephrase_correct_answer(sentence: str, key: Optional[str]) -> str:
    clause = _clause_for_key(sentence, key)
    clause = (clause or "").strip().rstrip('.')
    # handle "summary of X"
    m_summary = re.search(r'\bsummary\s+(?:of|about)\s+(.+)', clause, flags=re.IGNORECASE)
    if m_summary:
        tail = m_summary.group(1).strip().rstrip('.')
        return _shorten(f"A concise overview of {tail}.", max_len=200)
    # copula/predicate
    m = re.search(r'\b(is|are|was|were|refers to|refers|means|provides|offers|helps|enables|aims to|involves|serves as|acts as|is used to|is a|is an)\b', clause, flags=re.IGNORECASE)
    if m and key:
        pred = clause[m.end():].strip(',;: ')
        verb = m.group(0).lower()
        if pred:
            pred_clean = re.sub(r'^\b(to|for|the|a|an)\b', '', pred, flags=re.IGNORECASE).strip()
            candidate = f"{key} {verb} {pred_clean}"
            candidate = re.sub(r'\s+', ' ', candidate).strip()
            return _shorten(candidate + ".", max_len=200)
        else:
            return _shorten(f"{key} {verb}.", max_len=140)
    # for 'for/to' constructs
    m2 = re.search(r'\b(for|to)\b\s+(.+)$', clause, flags=re.IGNORECASE)
    if m2 and key:
        tail = m2.group(2).strip(' .;:')
        return _shorten(f"{key} is used to {tail}.", max_len=200)
    if key:
        return _shorten(f"{key} (a concept or practice described in the text).", max_len=200)
    return _shorten(clause.split('.')[0] + ".", max_len=200)


def _collect_related_concepts(context_text: str, sentence: str, key: Optional[str], limit: int = 8) -> List[str]:
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
            if ph and (not key or ph.lower() != key.lower()) and not _is_fragment_like(ph):
                candidates.append(ph)
        for t in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", s):
            if t.lower() in _STOPWORDS or (key and t.lower() == key.lower()):
                continue
            if not _is_fragment_like(t):
                candidates.append(t)
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
        if len(cl.split()) > 4:
            continue
        seen.add(low)
        out.append(cl)
        if len(out) >= limit:
            break
    return out


def _generate_distractors_from_predicate(key: str, obj_short: Optional[str]) -> List[str]:
    distracts = []
    obj_text = obj_short or "data"
    verbs = random.sample(_ALTERNATIVE_ACTION_VERBS, k=min(len(_ALTERNATIVE_ACTION_VERBS), 6))
    for alt in verbs:
        cand = f"{key} is used to {alt} {obj_text}."
        distracts.append(_shorten(cand, max_len=180))
        if len(distracts) >= 3:
            break
    return distracts[:3]


def _generate_plausible_distractors(correct: str, key: Optional[str], related: List[str], sentence: str) -> List[str]:
    distractors: List[str] = []
    used = set()
    pred_match = re.search(r'\b(is used to|is used for|helps|is used|is a|is an|refers to|involves|provides|measures|used to)\b\s*(.*)', correct, flags=re.IGNORECASE)
    if pred_match and key:
        obj = pred_match.group(2).strip().rstrip('.')
        obj_short = " ".join(obj.split()[:6]) if obj else None
        dlist = _generate_distractors_from_predicate(key, obj_short)
        for d in dlist:
            if _token_overlap(d, correct) < 0.8 and d.lower() not in used:
                distractors.append(d)
                used.add(d.lower())
            if len(distractors) >= 3:
                break
    for r in related:
        if len(distractors) >= 3:
            break
        if not r or (key and r.lower() == key.lower()):
            continue
        if " " in r:
            cand = f"{r} is a concept mentioned in the text."
        else:
            cand = f"{r} is used to manage related aspects."
        cand = _shorten(cand if cand.endswith('.') else cand + '.', max_len=180)
        if _token_overlap(cand, correct) > 0.8:
            continue
        if cand.lower() in used:
            continue
        distractors.append(cand)
        used.add(cand.lower())
    generics = [
        "A process for validating data quality.",
        "A technique to anonymize or protect user information.",
        "An organizational role responsible for data collection.",
        "A protocol designed to enhance interoperability."
    ]
    gi = 0
    while len(distractors) < 3 and gi < len(generics):
        g = generics[gi]
        gi += 1
        if _token_overlap(g, correct) > 0.8:
            continue
        if g.lower() in used:
            continue
        distractors.append(g)
        used.add(g.lower())
    while len(distractors) < 3:
        distractors.append("This statement is not supported by the lecture text.")
    return distractors[:3]


def _snippet_based_question_and_choices(sentence: str, context_text: str):
    correct = _shorten(sentence.strip().rstrip('.'), max_len=200)
    if not correct.endswith('.'):
        correct = correct + '.'
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (context_text or "")) if s.strip()]
    distractor_snips = []
    for s in sents:
        if s.strip() == sentence.strip():
            continue
        if len(distractor_snips) >= 3:
            break
        if len(s.split()) < 6:
            continue
        sn = _shorten(s.strip().rstrip('.'), max_len=160)
        if sn and sn.lower() != correct.lower() and _token_overlap(sn, correct) < 0.7:
            if not sn.endswith('.'):
                sn = sn + '.'
            distractor_snips.append(sn)
    if len(distractor_snips) < 3:
        related_words = _words_from_text(context_text)
        for w in related_words:
            if len(distractor_snips) >= 3:
                break
            cand = f"{w} is used to manage related aspects."
            if _token_overlap(cand, correct) < 0.75:
                distractor_snips.append(_shorten(cand, max_len=160))
    generics = [
        "A process for validating data quality.",
        "A technique to anonymize or protect user information.",
        "An organizational role responsible for data collection."
    ]
    gi = 0
    while len(distractor_snips) < 3 and gi < len(generics):
        cand = generics[gi]
        gi += 1
        if _token_overlap(cand, correct) < 0.8:
            distractor_snips.append(cand)
    while len(distractor_snips) < 3:
        distractor_snips.append("This statement is not supported by the lecture text.")
    snippet_short = _shorten(sentence.strip().rstrip('.'), max_len=120)
    stem = random.choice(_STEMS_SNIPPET).format(snippet=snippet_short)
    return stem, correct, distractor_snips[:3]


def generate_mcq_from_context(context_text: str, n: int = 5) -> List[Dict]:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (context_text or "")) if s.strip()]
    good_sentences = []
    for s in sentences:
        if len(s) < 20:
            continue
        first_word = re.match(r'^\s*([A-Za-z\'-]+)', s)
        if first_word and first_word.group(1).lower() in _BAD_STARTS and len(s.split()) < 8:
            continue
        if _is_fragment_like(s.split('.')[0]):
            continue
        good_sentences.append(s)
    if not good_sentences:
        good_sentences = sentences if sentences else [context_text.strip()]

    global_words = _words_from_text(context_text)
    if len(global_words) < 10:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", (context_text or ""))
        for t in tokens:
            if t.lower() not in _STOPWORDS and t not in global_words:
                global_words.append(t)
            if len(global_words) >= 20:
                break

    out: List[Dict] = []
    used_sentences = set()
    attempts = 0
    max_attempts = max(40, n * 10)
    created = 0

    while created < n and attempts < max_attempts:
        attempts += 1
        sent = random.choice(good_sentences)
        if sent in used_sentences and random.random() < 0.7:
            continue
        used_sentences.add(sent)

        key = _extract_key_phrase(sent)
        good_key = False
        if key:
            k_low = key.strip().lower()
            if (not _is_fragment_like(key)
                    and len(key.strip()) >= 3
                    and all(bad not in k_low for bad in _KEY_BLACKLIST_TOKENS)):
                good_key = True

        if good_key:
            correct = _rephrase_correct_answer(sent, key)
            if correct.strip().lower().startswith("information about") and key:
                clause = _clause_for_key(sent, key)
                vo = re.search(r'\b(?:helps|provides|allows|enables|is used to|is used for|measures|collects|validates|anonymizes|protects)\b\s*(.*)', clause, flags=re.IGNORECASE)
                if vo:
                    tail = vo.group(1).strip().rstrip('.')
                    if tail:
                        correct = f"{key} is used to {tail}."
                else:
                    nounm = re.search(r'\b([A-Za-z][A-Za-z\'-]{2,})\b', clause)
                    if nounm:
                        correct = f"{key} is a concept related to {nounm.group(1)}."
                    else:
                        correct = f"{key} (a concept described in the text)."
            related = _collect_related_concepts(context_text, sent, key, limit=12)
            if not related:
                related = [w for w in global_words if not key or w.lower() != key.lower()][:8]
            distractors = _generate_plausible_distractors(correct, key, related, sent)
            choices_list = [correct] + distractors[:3]
            stem = random.choice(_STEMS_KEY).format(key=key)
        else:
            stem, correct, distractors = _snippet_based_question_and_choices(sent, context_text)
            choices_list = [correct] + distractors[:3]

        normalized = []
        seen = set()
        for ch in choices_list:
            ch_clean = (ch or "").strip()
            k = re.sub(r'\s+', ' ', ch_clean.lower())
            if k in seen:
                alt = ch_clean.rstrip('.')
                alt = alt + " (alternative)."
                if re.sub(r'\s+', ' ', alt.strip().lower()) in seen:
                    alt = ch_clean + f" (variant)."
                normalized.append(alt)
                seen.add(re.sub(r'\s+', ' ', alt.strip().lower()))
            else:
                normalized.append(ch_clean)
                seen.add(k)
        choices_list = normalized[:4]
        if len(choices_list) < 4:
            fillers = [
                "A process for validating data quality.",
                "A technique to anonymize or protect user information.",
                "A measure used to evaluate performance."
            ]
            for f in fillers:
                if len(choices_list) >= 4:
                    break
                fk = re.sub(r'\s+', ' ', f.strip().lower())
                if fk not in seen:
                    choices_list.append(f)
                    seen.add(fk)

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
        choices = {lab: txt for lab, txt in zip(labels, choices_list)}

        correct_label = None
        for k_label, txt in choices.items():
            if txt.strip() == (correct or "").strip():
                correct_label = k_label
                break
        if correct_label is None:
            for k_label, txt in choices.items():
                if (correct or "").strip().lower() in txt.strip().lower():
                    correct_label = k_label
                    break
        if correct_label is None:
            slot = random.choice(labels)
            choices[slot] = correct
            correct_label = slot

        if _is_fragment_like(stem):
            continue

        out.append({
            "id": f"q{created+1}",
            "question": stem,
            "choices": choices,
            "answer": correct_label
        })
        created += 1

    if not out:
        for i in range(min(n, 3)):
            out.append({
                "id": f"q{i+1}",
                "question": f"Placeholder question {i+1}",
                "choices": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
                "answer": "A"
            })
    return out
