"""Utilities para extracción y scoring de frases (matched_phrases) y generación de payload para nubes de palabras.

Este módulo intenta importar dependencias (spaCy, sentence-transformers, rapidfuzz, sklearn) pero
funciona con heurísticas si faltan. Diseñado para integrarse con `analizador.py`.
"""
from typing import List, Dict, Any, Iterable, Optional, Tuple
import re
import unicodedata
from collections import Counter, defaultdict

# Stopwords en español para filtrar palabras vacías
SPANISH_STOPWORDS = {
    'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
    'por', 'con', 'su', 'para', 'es', 'o', 'este', 'sí', 'porque', 'esta',
    'son', 'entre', 'está', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene',
    'también', 'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado',
    'desde', 'todo', 'nos', 'durante', 'estados', 'todos', 'uno', 'les', 'ni',
    'contra', 'otros', 'fueron', 'ese', 'eso', 'estas', 'estaba', 'estamos',
    'algunas', 'algo', 'nosotros', 'mi', 'mío', 'tú', 'te', 'ti', 'tu', 'tus',
    'ellos', 'esas', 'esos', 'nuestra', 'nuestro', 'vuestra', 'vuestro', 'os',
    'mía', 'mías', 'míos', 'tuya', 'tuyo', 'tuyos', 'tuyas', 'suya', 'suyo',
    'suyos', 'suyas', 'nuestra', 'nuestras', 'nuestro', 'nuestros', 'vosotras',
    'vosotros', 'vuestra', 'vuestras', 'vuestro', 'vuestros', 'esa', 'eso',
    'hemos', 'habían', 'habías', 'habíamos', 'habían', 'habrá', 'habrán',
    'habría', 'habrían', 'haya', 'hayan', 'hayamos', 'hayáis', 'había',
    'habré', 'habremos', 'habréis', 'habrá', 'habrán', 'habría', 'habríamos',
    'habríais', 'habrían', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayan',
    'habi', 'habías', 'habíamos', 'habíais', 'habían', 'he', 'has', 'hemos',
    'habéis', 'han', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayan', 'había',
    'habías', 'habíamos', 'habíais', 'habían', 'habrá', 'habrás', 'habremos',
    'habréis', 'habrán', 'habría', 'habrías', 'habríamos', 'habríais', 'habrían',
    'haya', 'hayas', 'hayamos', 'hayáis', 'hayan', 'dé', 'des', 'demos', 'deis',
    'den', 'daba', 'dabas', 'dábamos', 'dabais', 'daban', 'daré', 'darás',
    'daremos', 'daréis', 'darán', 'daría', 'darías', 'daríamos', 'daríais',
    'darían', 'dé', 'des', 'demos', 'deis', 'den', 'daba', 'dabas', 'dábamos',
    'dabais', 'daban', 'daré', 'darás', 'daremos', 'daréis', 'darán', 'daría',
    'darías', 'daríamos', 'daríais', 'darían', 'dé', 'des', 'demos', 'deis', 'den'
}


# ---- Lazy imports for optional libs ----
_has_spacy = False
_has_transformers = False
_has_rapidfuzz = False
_has_sklearn = False

try:
    import spacy
    _has_spacy = True
except Exception:
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    _has_transformers = True
except Exception:
    SentenceTransformer = None

try:
    from rapidfuzz import fuzz
    _has_rapidfuzz = True
except Exception:
    fuzz = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _has_sklearn = True
except Exception:
    TfidfVectorizer = None


def _is_stopword(word: str) -> bool:
    """Verifica si una palabra es un stopword en español."""
    return word.lower() in SPANISH_STOPWORDS


def _filter_stopwords(phrase: str) -> str:
    """Filtra stopwords de una frase. Retorna la frase sin stopwords o vacío si solo tenía stopwords.

    IMPORTANTE: Solo filtra stopwords cuando la frase es muy corta (1-2 palabras).
    Para frases con múltiples palabras, las mantiene intactas para preservar etiquetas
    como 'plazo de reclamo', 'resolución de disputas', etc.
    """
    words = phrase.split()

    # Si la frase tiene 3+ palabras, asumimos que es una etiqueta definida y no la filtramos
    # Esto preserva etiquetas como "plazo de reclamo", "resolución de disputas"
    if len(words) >= 3:
        return phrase

    # Para frases cortas (1-2 palabras), filtrar stopwords
    filtered = [w for w in words if not _is_stopword(w)]
    return ' '.join(filtered).strip()


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # remove punctuation except internal hyphens
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_candidate_phrases(text: str, max_ngram: int = 3) -> List[str]:
    """Extrae candidatos de frases desde un texto.

    Intenta usar spaCy para noun_chunks; si no está disponible usa n-grams y regex.
    Retorna frases normalizadas y únicas.
    """
    text = text or ""
    norm = _normalize_text(text)
    candidates = []

    if _has_spacy:
        try:
            # cargar spaCy en modo lazy: usa modelo pequeño si no se ha cargado
            if not hasattr(extract_candidate_phrases, "_nlp"):
                try:
                    extract_candidate_phrases._nlp = spacy.load("es_core_news_sm")
                except Exception:
                    print("No se pudo cargar el modelo de spaCy.")
                    # fallback to blank spanish pipeline
                    extract_candidate_phrases._nlp = spacy.blank("es")

            doc = extract_candidate_phrases._nlp(text)
            for nc in doc.noun_chunks:
                s = _normalize_text(nc.text)
                if s:
                    candidates.append(s)
        except Exception:
            pass

    # n-grams fallback/augmentation
    tokens = [t for t in re.findall(r"\w+", norm) if len(t) > 0]
    L = len(tokens)
    for n in range(1, min(max_ngram, 5) + 1):
        for i in range(0, max(0, L - n + 1)):
            gram = " ".join(tokens[i:i + n])
            # filter stop-ish short tokens
            if len(gram) > 0:
                candidates.append(gram)

    # deduplicate preserving order
    seen = set()
    out = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)

    return out


def precompute_embeddings_for_phrases(category_phrases: Dict[str, List[str]], model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Precomputar embeddings para la lista de frases por categoría.

    Retorna dict: {category: {'phrases': [...], 'embeddings': np.array}}
    Si sentence-transformers no está disponible devuelve None.
    """
    if not _has_transformers:
        return {}

    model = SentenceTransformer(model_name)
    result = {}
    for cat, phrases in category_phrases.items():
        normalized = [_normalize_text(p) for p in phrases if p]
        if not normalized:
            result[cat] = {"phrases": [], "embeddings": None}
            continue
        embeddings = model.encode(normalized, convert_to_numpy=True)
        result[cat] = {"phrases": normalized, "embeddings": embeddings}
    return result


def _fuzzy_score(a: str, b: str) -> float:
    if not _has_rapidfuzz:
        # simple fallback: exact match -> 100, prefix -> 80, substring -> 60
        if a == b:
            return 100.0
        if a in b or b in a:
            return 80.0
        return 0.0
    try:
        return float(fuzz.token_sort_ratio(a, b))
    except Exception:
        return 0.0


def _embedding_similarity_score(vecs_cat, vec_candidate) -> List[float]:
    try:
        sims = _cosine_similarity(vec_candidate.reshape(1, -1), vecs_cat)
        # return flattened list
        return [float(x) for x in sims[0]]
    except Exception:
        return []


def _phrase_exists_in_categories(phrase: str, category_phrases: Dict[str, List[str]], similarity_threshold: float = 0.8) -> bool:
    """
    Verificar si una frase existe en las categorías especificadas.

    Args:
        phrase: Frase a verificar
        category_phrases: Diccionario de categorías y sus frases
        similarity_threshold: Umbral de similitud para considerar un match válido

    Returns:
        bool: True si la frase existe en alguna categoría con suficiente similitud
    """
    phrase_norm = _normalize_text(phrase)

    for category, phrases in category_phrases.items():
        for category_phrase in phrases:
            category_phrase_norm = _normalize_text(category_phrase)

            # Verificar coincidencia exacta
            if phrase_norm == category_phrase_norm:
                return True

            # Verificar si una es substring de la otra (para frases compuestas)
            if phrase_norm in category_phrase_norm or category_phrase_norm in phrase_norm:
                return True

            # Verificar similitud fuzzy si está disponible
            if _has_rapidfuzz:
                try:
                    similarity = fuzz.token_sort_ratio(phrase_norm, category_phrase_norm) / 100.0
                    if similarity >= similarity_threshold:
                        return True
                except Exception:
                    pass

    return False


def score_candidates(
    candidates: List[str],
    category_phrases: Dict[str, List[str]],
    precomputed_embeddings: Optional[Dict[str, Any]] = None,
    use_embedding: bool = True,
    use_fuzzy: bool = True,
    tfidf_vectorizer: Optional[Any] = None
) -> Dict[str, List[Tuple[str, float, str]]]:
    """Score candidates against each category.

    Returns a dict: {category: [(phrase, score, method), ...]}

    Methods: 'embed', 'fuzzy', 'tfidf' (tfidf only if tfidf_vectorizer provided)
    """
    results: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)

    # Precompute embeddings for candidates if needed
    candidate_embeddings = None
    if use_embedding and _has_transformers:
        model = getattr(score_candidates, "_embed_model", None)
        if model is None:
            score_candidates._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            model = score_candidates._embed_model
        candidate_embeddings = model.encode(candidates, convert_to_numpy=True)

    # If TF-IDF vectorizer provided, transform candidates
    candidate_tfidf = None
    if tfidf_vectorizer is not None:
        try:
            candidate_tfidf = tfidf_vectorizer.transform(candidates)
        except Exception:
            candidate_tfidf = None

    for cat, phrases in category_phrases.items():
        # prepare cat embeddings if available
        cat_emb = None
        if precomputed_embeddings and cat in precomputed_embeddings:
            cat_emb = precomputed_embeddings[cat].get("embeddings")

        for idx, cand in enumerate(candidates):
            cand_norm = _normalize_text(cand)
            best_score = 0.0
            best_method = ''

            # VALIDACIÓN PREVIA: Solo procesar si el candidato tiene alguna relación con las frases de la categoría
            if not _has_any_similarity_to_category(cand_norm, phrases):
                continue

            # embedding similarity (max over phrases in category)
            if use_embedding and candidate_embeddings is not None and cat_emb is not None:
                sims = _embedding_similarity_score(cat_emb, candidate_embeddings[idx])
                if sims:
                    maxsim = max(sims)
                    # convert cosine [-1,1] to 0-1, but be more conservative
                    score = max(0.0, float(maxsim))  # Ensure non-negative
                    if score > best_score:
                        best_score = score
                        best_method = 'embed'

            # fuzzy matching
            if use_fuzzy and phrases:
                local_best = 0.0
                for p in phrases:
                    s = _fuzzy_score(cand_norm, _normalize_text(p))
                    if s > local_best:
                        local_best = s
                # normalize fuzzy 0-100 to 0-1
                s_norm = local_best / 100.0
                if s_norm > best_score:
                    best_score = s_norm
                    best_method = 'fuzzy'

            # tfidf similarity (cosine with TF-IDF vectors) - if vectorizer provided and we can transform category phrases
            if tfidf_vectorizer is not None:
                try:
                    # transform category phrases to tfidf and compute cosine with candidate
                    # This is heavier; only do if candidate_tfidf available
                    pass
                except Exception:
                    pass

            # Solo almacenar si tiene un score significativo (> 0)
            if best_score > 0.0:
                results[cat].append((cand, float(best_score), best_method))

        # sort by score desc
        results[cat].sort(key=lambda x: x[1], reverse=True)

    return results


def _has_any_similarity_to_category(candidate_norm: str, phrases: List[str], min_similarity: float = 0.5) -> bool:
    """
    Verificar si un candidato tiene alguna similitud mínima con las frases de una categoría.

    Args:
        candidate_norm: Candidato normalizado
        phrases: Lista de frases de la categoría
        min_similarity: Similitud mínima requerida (0.0 a 1.0)

    Returns:
        bool: True si tiene similitud mínima con al menos una frase
    """
    if not phrases:
        return False

    for phrase in phrases:
        phrase_norm = _normalize_text(phrase)

        # Verificar coincidencia exacta o substring
        if candidate_norm == phrase_norm:
            return True
        if candidate_norm in phrase_norm or phrase_norm in candidate_norm:
            return True

        # Verificar similitud fuzzy si está disponible
        if _has_rapidfuzz:
            try:
                similarity = fuzz.token_sort_ratio(candidate_norm, phrase_norm) / 100.0
                if similarity >= min_similarity:
                    return True
            except Exception:
                pass

    return False


def match_phrases_for_clause(
    text: str,
    category_phrases: Dict[str, List[str]],
    top_n: int = 3,
    precomputed_embeddings: Optional[Dict[str, Any]] = None,
    min_score: float = 0.1,
    max_ngram: int = 3,
    use_embeddings: bool = True
) -> List[Dict[str, Any]]:
    """Extrae candidatos y devuelve las matched_phrases para la cláusula.

    Args:
        text: Texto de la cláusula a analizar
        category_phrases: Diccionario de frases por categoría
        top_n: Número máximo de frases a retornar
        precomputed_embeddings: Embeddings precomputados (opcional)
        min_score: Score mínimo para considerar una frase
        max_ngram: Máximo n-gram para extracción de candidatos
        use_embeddings: Si usar embeddings para similarity

    Devuelve lista de dicts: {'phrase':..., 'score':..., 'method':...}
    """
    # Validar que tenemos categorías válidas
    if not category_phrases or not any(phrases for phrases in category_phrases.values()):
        print("⚠️ No hay categorías válidas o están vacías")
        return []

    candidates = extract_candidate_phrases(text, max_ngram=max_ngram)
    if not candidates:
        return []

    scored = score_candidates(
        candidates,
        category_phrases,
        precomputed_embeddings=precomputed_embeddings,
        use_embedding=use_embeddings
    )

    # For each category, pick top candidate; then across categories pick global top_n
    global_hits: List[Tuple[str, float, str]] = []
    for cat, lst in scored.items():
        if not lst:
            continue
        # take best
        phrase, score, method = lst[0]
        if score >= min_score:
            # VALIDACIÓN ADICIONAL: Solo incluir si la frase realmente está en alguna categoría
            # Usar un umbral más estricto para la validación final
            if _phrase_exists_in_categories(phrase, category_phrases, similarity_threshold=0.85):
                global_hits.append((phrase, score, method))
            else:
                print(f"🚫 Etiqueta rechazada: '{phrase}' (score: {score:.3f}) - No existe en categorías")

    # deduplicate by normalized phrase
    seen = set()
    out = []
    # sort global hits by score desc
    global_hits.sort(key=lambda x: x[1], reverse=True)
    for phrase, score, method in global_hits:
        pnorm = _normalize_text(phrase)
        if pnorm in seen:
            continue
        seen.add(pnorm)
        out.append({"phrase": phrase, "score": float(score), "method": method})
        if len(out) >= top_n:
            break

    return out


def build_wordcloud_payload_from_clauses(clauses: Iterable[Dict[str, Any]], phrase_key: str = 'matched_phrases', min_score: float = 0.0) -> List[Dict[str, Any]]:
    """Agrega matched_phrases desde una colección de cláusulas y retorna lista [{text, value}].

    clauses: iterable of dicts como las que devuelve `analizar_contrato_completo` (cada cláusula puede tener key 'matched_phrases').
    Filtra stopwords en español para evitar palabras vacías como 'de', 'el', etc.
    """
    ctr = Counter()
    for c in clauses:
        if not c:
            continue
        mp = c.get(phrase_key) if isinstance(c, dict) else None
        if not mp:
            continue
        for entry in mp:
            if isinstance(entry, str):
                phrase = _normalize_text(entry)
                score = 1.0
            elif isinstance(entry, dict):
                phrase = _normalize_text(entry.get('phrase', '') or '')
                try:
                    score = float(entry.get('score', 0.0) or 0.0)
                except Exception:
                    score = 0.0
            else:
                continue
            if not phrase:
                continue
            if score < min_score:
                continue

            # ✅ NUEVO: Filtrar stopwords
            phrase_filtered = _filter_stopwords(phrase)
            if not phrase_filtered:  # Si después del filtro queda vacío, saltar
                continue

            # increment by 1 occurrence (could weight by score)
            ctr[phrase_filtered] += 1

    return [{"text": k, "value": v} for k, v in ctr.most_common()]


if __name__ == '__main__':
    # small demo if run directly
    sample = "El contratista recibirá un anticipo del 20% y un certificado provisional de pago. Se retendrá un porcentaje como garantía."
    categories = {
        'Pago': ['anticipo', 'certificado provisional de pago', 'retención', 'certificado final de pago'],
        'Penalidades': ['multa', 'penalidad', 'resolución']
    }

    hits = match_phrases_for_clause(sample, categories, top_n=3)
    print('Hits:', hits)
    wc = build_wordcloud_payload_from_clauses([{'matched_phrases': hits}])
    print('Wordcloud payload:', wc)
