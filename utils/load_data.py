from datasets import load_dataset
import re
import random
import json
import os


class ListAdapter:
    """Wraps a plain list of strings with a standard index/slice interface."""
    def __init__(self, sentences):
        self.sentences = sentences
        self.metadata = None  # set by dataset loaders

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.sentences[idx]
        return self.sentences[idx]

    def save(self, path):
        """Save sentences + metadata for reproducibility and tracing pipeline.

        Creates:
            {path}.txt  — plain text, one sentence per line (for tracing)
            {path}.json — full metadata (IO, S, answer, etc.)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        txt_path = path if path.endswith(".txt") else path + ".txt"
        json_path = path.replace(".txt", ".json") if path.endswith(".txt") else path + ".json"

        with open(txt_path, "w") as f:
            for s in self.sentences:
                f.write(s + "\n")
        print(f"Saved {len(self.sentences)} sentences to {txt_path}")

        if self.metadata:
            with open(json_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            print(f"Saved metadata to {json_path}")


# ==========================================
#  Sentence Splitting
# ==========================================

_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def _split_sentences(text):
    try:
        import nltk
        nltk.data.find('tokenizers/punkt_tab')
        return nltk.sent_tokenize(text)
    except (ImportError, LookupError):
        return _SENT_RE.split(text)


def _rough_token_count(text):
    return len(text.split())


# ==========================================
#  Pile Loaders
# ==========================================

def load_pile_sentences(
    target=10000,
    min_tokens=10,
    max_tokens=60,
    seed=42,
    cache_dir=None,
    stratified=True,
):
    rng = random.Random(seed)

    print("Loading NeelNanda/pile-10k...")
    ds = load_dataset("NeelNanda/pile-10k", split="train", cache_dir=cache_dir)

    domain_sentences = {}
    for item in ds:
        domain = item.get("meta", {}).get("pile_set_name", "unknown")
        text = item.get("text", "")
        if not text.strip():
            continue

        for sent in _split_sentences(text):
            sent = sent.strip()
            n = _rough_token_count(sent)
            if min_tokens <= n <= max_tokens:
                domain_sentences.setdefault(domain, []).append(sent)

    total_available = sum(len(v) for v in domain_sentences.values())
    print(f"Extracted {total_available} sentences across {len(domain_sentences)} domains:")
    for domain, sents in sorted(domain_sentences.items(), key=lambda x: -len(x[1])):
        print(f"  {domain}: {len(sents)}")

    if stratified and len(domain_sentences) > 1:
        selected = _stratified_sample(domain_sentences, target, rng)
    else:
        all_sents = [s for sents in domain_sentences.values() for s in sents]
        rng.shuffle(all_sents)
        selected = all_sents[:target]

    print(f"Selected {len(selected)} sentences (target: {target})")
    return ListAdapter(selected)


def _stratified_sample(domain_sentences, target, rng):
    domains = list(domain_sentences.keys())
    rng.shuffle(domains)

    for d in domains:
        rng.shuffle(domain_sentences[d])

    per_domain = target // len(domains)
    selected = []
    overflow = []

    for d in domains:
        sents = domain_sentences[d]
        if len(sents) <= per_domain:
            selected.extend(sents)
        else:
            selected.extend(sents[:per_domain])
            overflow.extend(sents[per_domain:])

    remaining = target - len(selected)
    if remaining > 0 and overflow:
        rng.shuffle(overflow)
        selected.extend(overflow[:remaining])

    rng.shuffle(selected)
    return selected


def load_pile_sentences_large(
    target=10000,
    min_tokens=10,
    max_tokens=60,
    seed=42,
    cache_dir=None,
    stratified=True,
    subsets=None,
):
    rng = random.Random(seed)

    if subsets is None:
        subsets = [
            "Pile-CC", "Wikipedia (en)", "Github", "ArXiv",
            "StackExchange", "PubMed Abstracts", "FreeLaw",
            "USPTO Backgrounds", "DM Mathematics", "Ubuntu IRC",
            "PhilPapers", "NIH ExPorter", "HackerNews", "Enron Emails",
        ]

    per_domain_target = (target // len(subsets)) * 3

    domain_sentences = {}
    for subset_name in subsets:
        print(f"Loading subset: {subset_name}...")
        try:
            ds = load_dataset(
                "ArmelR/the-pile-splitted", subset_name,
                split="train", streaming=True, cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"  Skipping {subset_name}: {e}")
            continue

        sents = []
        for item in ds:
            text = item.get("text", "")
            for sent in _split_sentences(text):
                sent = sent.strip()
                n = _rough_token_count(sent)
                if min_tokens <= n <= max_tokens:
                    sents.append(sent)
                    if len(sents) >= per_domain_target:
                        break
            if len(sents) >= per_domain_target:
                break

        domain_sentences[subset_name] = sents
        print(f"  Collected {len(sents)} sentences")

    total_available = sum(len(v) for v in domain_sentences.values())
    print(f"Total: {total_available} sentences across {len(domain_sentences)} domains")

    if stratified and len(domain_sentences) > 1:
        selected = _stratified_sample(domain_sentences, target, rng)
    else:
        all_sents = [s for sents in domain_sentences.values() for s in sents]
        rng.shuffle(all_sents)
        selected = all_sents[:target]

    print(f"Selected {len(selected)} sentences (target: {target})")
    return ListAdapter(selected)


# ==========================================
#  IOI Dataset
# ==========================================

_IOI_NAMES = [
    "Alice", "Bob", "Charlie", "Claire", "David", "Diana",
    "Edward", "Emma", "Frank", "Grace", "Harry", "Helen",
    "Jack", "James", "Jane", "Jim", "John", "Julia",
    "Karen", "Kevin", "Laura", "Linda", "Lisa", "Mark",
    "Mary", "Michael", "Nancy", "Oscar", "Paul", "Peter",
    "Rachel", "Richard", "Robert", "Rose", "Sam", "Sarah",
    "Steve", "Susan", "Thomas", "Tom", "Victor", "William",
]

_IOI_TEMPLATES_ABBA = [
    "{IO} and {S} went to the {PLACE}. {S} gave a {OBJECT} to",
    "{IO} and {S} went to the {PLACE}. {S} gave the {OBJECT} to",
    "{IO} and {S} were at the {PLACE}. {S} handed a {OBJECT} to",
    "{IO} and {S} were at the {PLACE}. {S} passed the {OBJECT} to",
    "{IO} and {S} arrived at the {PLACE}. {S} offered a {OBJECT} to",
    "{IO} and {S} met at the {PLACE}. {S} lent a {OBJECT} to",
    "{IO} and {S} stopped by the {PLACE}. {S} sold a {OBJECT} to",
    "{IO} and {S} visited the {PLACE}. {S} showed the {OBJECT} to",
    "{IO} and {S} walked to the {PLACE}. {S} threw the {OBJECT} to",
    "After {IO} and {S} went to the {PLACE}, {S} gave a {OBJECT} to",
    "When {IO} and {S} were at the {PLACE}, {S} handed a {OBJECT} to",
    "Because {IO} and {S} met at the {PLACE}, {S} offered a {OBJECT} to",
]

_IOI_TEMPLATES_BABA = [
    "{S} and {IO} went to the {PLACE}. {S} gave a {OBJECT} to",
    "{S} and {IO} went to the {PLACE}. {S} gave the {OBJECT} to",
    "{S} and {IO} were at the {PLACE}. {S} handed a {OBJECT} to",
    "{S} and {IO} were at the {PLACE}. {S} passed the {OBJECT} to",
    "{S} and {IO} arrived at the {PLACE}. {S} offered a {OBJECT} to",
    "{S} and {IO} met at the {PLACE}. {S} lent a {OBJECT} to",
    "{S} and {IO} stopped by the {PLACE}. {S} sold a {OBJECT} to",
    "{S} and {IO} visited the {PLACE}. {S} showed the {OBJECT} to",
    "{S} and {IO} walked to the {PLACE}. {S} threw the {OBJECT} to",
    "After {S} and {IO} went to the {PLACE}, {S} gave a {OBJECT} to",
    "When {S} and {IO} were at the {PLACE}, {S} handed a {OBJECT} to",
    "Because {S} and {IO} met at the {PLACE}, {S} offered a {OBJECT} to",
]

_IOI_PLACES = [
    "store", "market", "park", "school", "hospital", "office",
    "restaurant", "library", "church", "gym", "airport", "hotel",
    "beach", "museum", "station", "cafe", "bakery", "bank",
]

_IOI_OBJECTS = [
    "drink", "bottle", "ring", "book", "letter", "key", "ticket",
    "phone", "bag", "hat", "ball", "coin", "toy", "pen", "map",
    "cake", "flower", "gift", "card", "tool",
]


def load_ioi_dataset(target=1000, seed=42, names=None, symmetric=True):
    rng = random.Random(seed)
    names = names or list(_IOI_NAMES)

    sentences = []
    metadata = []

    for i in range(target):
        io_name, s_name = rng.sample(names, 2)
        place = rng.choice(_IOI_PLACES)
        obj = rng.choice(_IOI_OBJECTS)

        if symmetric:
            if i % 2 == 0:
                tmpl = rng.choice(_IOI_TEMPLATES_ABBA)
                tmpl_type = "ABBA"
            else:
                tmpl = rng.choice(_IOI_TEMPLATES_BABA)
                tmpl_type = "BABA"
        else:
            pool = _IOI_TEMPLATES_ABBA + _IOI_TEMPLATES_BABA
            tmpl = rng.choice(pool)
            tmpl_type = "ABBA" if "{IO}" == tmpl[:4] else "BABA"

        prompt = tmpl.format(IO=io_name, S=s_name, PLACE=place, OBJECT=obj)
        answer = " " + io_name

        sentences.append(prompt)
        metadata.append({
            "prompt": prompt,
            "answer": answer,
            "IO": " " + io_name,
            "S": " " + s_name,
            "template_type": tmpl_type,
        })

    adapter = ListAdapter(sentences)
    adapter.metadata = metadata

    abba_count = sum(1 for m in metadata if m["template_type"] == "ABBA")
    n_unique = len(set(sentences))
    print(f"Generated {len(sentences)} IOI sentences ({n_unique} unique)")
    print(f"  ABBA: {abba_count}, BABA: {len(sentences) - abba_count}")
    return adapter


# ==========================================
#  Factual Recall Dataset
# ==========================================

def load_factual_dataset(
    target=5000, seed=42, relation_ids=None,
    cache_dir=None, single_token_only=True,
):
    rng = random.Random(seed)

    print("Loading factual dataset: NeelNanda/counterfact-tracing ...")
    ds = load_dataset(
        "NeelNanda/counterfact-tracing", split="train", cache_dir=cache_dir,
    )

    sentences = []
    metadata = []

    for item in ds:
        prompt = item.get("prompt", "")
        answer = item.get("target_true", "")
        if not prompt or not answer:
            continue

        if single_token_only and len(answer.split()) > 1:
            continue

        relation = item.get("relation_id", "")
        if relation_ids is not None and relation not in relation_ids:
            continue

        sentences.append(prompt)
        metadata.append({
            "prompt": prompt,
            "answer": answer,
            "target_false": item.get("target_false", ""),
            "subject": item.get("subject", ""),
            "relation": relation,
        })

    combined = list(zip(sentences, metadata))
    rng.shuffle(combined)
    combined = combined[:target]
    sentences = [s for s, _ in combined]
    metadata = [m for _, m in combined]

    adapter = ListAdapter(sentences)
    adapter.metadata = metadata

    relation_counts = {}
    for m in metadata:
        r = m["relation"] or "unknown"
        relation_counts[r] = relation_counts.get(r, 0) + 1

    print(f"Loaded {len(sentences)} factual prompts (target: {target})")
    print(f"  Unique relations: {len(relation_counts)}")
    top = sorted(relation_counts.items(), key=lambda x: -x[1])[:10]
    for rel, cnt in top:
        print(f"    {rel}: {cnt}")
    if len(relation_counts) > 10:
        print(f"    ... and {len(relation_counts) - 10} more")

    return adapter


FACTUAL_RELATIONS = {
    "capital_of": "P36", "language": "P37", "located_in": "P131",
    "country": "P17", "birth_place": "P19", "death_place": "P20",
    "founded_in": "P740", "occupation": "P106", "genre": "P136",
    "instrument": "P1303", "currency": "P38", "continent": "P30",
    "shares_border": "P47", "native_language": "P103", "employer": "P108",
}


# ==========================================
#  Gender Pronoun Dataset
# ==========================================

# Professions with stereotypical gender associations.
# Based on Bureau of Labor Statistics data used in Winogender
# (Rudinger et al., 2018) and Professions benchmark
# (Bolukbasi et al., 2016).  Restricted to names that are
# typically a single token in GPT-NeoX / Pythia tokenizers.

_GENDER_PROFESSIONS = {
    # (profession, stereotypical_gender)
    # female-stereotyped
    "nurse":        "female",
    "secretary":    "female",
    "teacher":      "female",
    "librarian":    "female",
    "nanny":        "female",
    "receptionist": "female",
    "dietitian":    "female",
    "therapist":    "female",
    "counselor":    "female",
    "hygienist":    "female",
    "hairdresser":  "female",
    "maid":         "female",
    # male-stereotyped
    "doctor":       "male",
    "engineer":     "male",
    "surgeon":      "male",
    "mechanic":     "male",
    "plumber":      "male",
    "carpenter":    "male",
    "electrician":  "male",
    "programmer":   "male",
    "pilot":        "male",
    "architect":    "male",
    "janitor":      "male",
    "driver":       "male",
}

_GENDER_TEMPLATES = [
    "The {profession} finished the work and said that",
    "The {profession} looked up from the desk and",
    "The {profession} told the patient that",
    "The {profession} walked into the room and",
    "The {profession} said that",
    "The {profession} explained that",
    "The {profession} mentioned that",
    "After the meeting, the {profession} said that",
    "The {profession} was tired because",
    "The {profession} helped the client and then",
    "The {profession} completed the task and",
    "Everyone agreed that the {profession} did a great job and",
]


def load_gender_pronoun_dataset(target=200, seed=42):
    """Generate prompts for gender pronoun prediction.

    Each prompt ends just before a gendered pronoun.  The model's
    predicted pronoun (" he" or " she") is the target token, and
    the profession word is the expected attribution source.

    Metadata fields:
        prompt:         the input string
        answer:         " he" or " she" (stereotypical)
        profession:     the profession word
        stereotype:     "male" or "female"
        counter_answer: the non-stereotypical pronoun
    """
    rng = random.Random(seed)
    professions = list(_GENDER_PROFESSIONS.items())

    sentences = []
    metadata = []

    for i in range(target):
        prof, stereo = rng.choice(professions)
        tmpl = rng.choice(_GENDER_TEMPLATES)
        prompt = tmpl.format(profession=prof)

        if stereo == "female":
            answer = " she"
            counter = " he"
        else:
            answer = " he"
            counter = " she"

        sentences.append(prompt)
        metadata.append({
            "prompt": prompt,
            "answer": answer,
            "profession": prof,
            "stereotype": stereo,
            "counter_answer": counter,
        })

    adapter = ListAdapter(sentences)
    adapter.metadata = metadata

    n_female = sum(1 for m in metadata if m["stereotype"] == "female")
    print(f"Generated {len(sentences)} gender-pronoun sentences")
    print(f"  Female-stereotyped: {n_female}, Male-stereotyped: {len(sentences) - n_female}")
    return adapter


    # ============================================================================
# Append everything below to the bottom of utils/load_data.py
# ============================================================================

# ==========================================
#  IOI tokenization verification + ABC generation
# ==========================================

def verify_single_token_pool(
    tokenizer,
    names=None,
    places=None,
    objects=None,
):
    """Filter the IOI name / place / object pools to single-token entries.

    Names are checked in two contexts (with leading space for mid-sentence
    use, without leading space for sentence-initial templates). A name
    must tokenize to one BPE token in BOTH contexts to guarantee that
    target prompts and ABC references match in length regardless of
    which template they use.

    Places and objects only ever appear after "the " / "a ", so only the
    leading-space form is checked.

    Returns:
        dict {'names': [...], 'places': [...], 'objects': [...]}
        containing only the entries that pass. Multi-token entries are
        dropped (with a printed report).
    """
    names = names or list(_IOI_NAMES)
    places = places or list(_IOI_PLACES)
    objects = objects or list(_IOI_OBJECTS)

    def _filter(items, kind, both_contexts):
        single, multi = [], []
        for x in items:
            ids_lead = tokenizer.encode(" " + x, add_special_tokens=False)
            if both_contexts:
                ids_bare = tokenizer.encode(x, add_special_tokens=False)
                ok = (len(ids_lead) == 1 and len(ids_bare) == 1)
                detail = (len(ids_lead), len(ids_bare))
            else:
                ok = (len(ids_lead) == 1)
                detail = len(ids_lead)
            (single if ok else multi).append((x, detail))
        if multi:
            multi_str = ", ".join(f"{x}{n}" for x, n in multi)
            print(f"  {kind}: dropped {len(multi)} multi-token entries: "
                  f"{multi_str}")
        return [x for x, _ in single]

    print("Verifying single-token pool:")
    out = {
        "names":   _filter(names,   "names",   both_contexts=True),
        "places":  _filter(places,  "places",  both_contexts=False),
        "objects": _filter(objects, "objects", both_contexts=False),
    }
    print(f"  kept: {len(out['names'])} names, "
          f"{len(out['places'])} places, "
          f"{len(out['objects'])} objects")

    if len(out["names"]) < 10:
        raise ValueError(
            f"Name pool too small after filtering: {len(out['names'])}. "
            f"Need ≥10 to sample ABC triples (3 distinct names)."
        )
    return out


def _ioi_template_to_abc(template):
    """Turn an IOI template string ({IO}/{S}/{S}) into an ABC template
    ({A}/{B}/{C}). The first {S} becomes {B}, second {S} becomes {C},
    {IO} becomes {A}. Token positions are preserved exactly."""
    out = template.replace("{IO}", "__A__")
    parts = out.split("{S}")
    if len(parts) != 3:
        raise ValueError(
            f"IOI template must contain exactly two {{S}} slots; got "
            f"{len(parts) - 1} in: {template!r}"
        )
    out = parts[0] + "__B__" + parts[1] + "__C__" + parts[2]
    return (out.replace("__A__", "{A}")
               .replace("__B__", "{B}")
               .replace("__C__", "{C}"))


def _invert_ioi_target(meta):
    """Recover (template, place, object) from an IOI metadata entry by
    enumerating templates × places × objects and string-matching against
    the prompt. ~4,300 comparisons per call."""
    prompt = meta["prompt"]
    io = meta["IO"].strip()
    s = meta["S"].strip()
    templates = (_IOI_TEMPLATES_ABBA if meta["template_type"] == "ABBA"
                 else _IOI_TEMPLATES_BABA)
    for tmpl in templates:
        for place in _IOI_PLACES:
            for obj in _IOI_OBJECTS:
                if tmpl.format(IO=io, S=s, PLACE=place, OBJECT=obj) == prompt:
                    return tmpl, place, obj
    raise ValueError(
        f"Could not recover template / place / object for prompt: {prompt!r}"
    )


def build_abc_for_target(meta, n_refs, names=None, seed=0):
    """Generate `n_refs` ABC reference prompts matched to one IOI target.

    Each reference shares the target's template, place, and object —
    only the names change. The three names sampled are distinct and
    exclude the target's own IO and S, so the duplication signal that
    drives the IOI circuit is fully broken.

    Length matching to the target is guaranteed if `names` is single-
    token-verified (see `verify_single_token_pool`).
    """
    rng = random.Random(seed)
    name_pool = list(names) if names is not None else list(_IOI_NAMES)
    template, place, obj = _invert_ioi_target(meta)
    abc_template = _ioi_template_to_abc(template)

    target_io = meta["IO"].strip()
    target_s = meta["S"].strip()
    eligible = [n for n in name_pool if n != target_io and n != target_s]
    if len(eligible) < 3:
        raise ValueError(
            f"Need ≥3 eligible names after excluding IO={target_io!r} and "
            f"S={target_s!r}; got {len(eligible)} from pool size "
            f"{len(name_pool)}"
        )

    prompts = []
    seen = set()
    attempts = 0
    max_attempts = n_refs * 20
    while len(prompts) < n_refs and attempts < max_attempts:
        attempts += 1
        a, b, c = rng.sample(eligible, 3)
        p = abc_template.format(A=a, B=b, C=c, PLACE=place, OBJECT=obj)
        if p in seen:
            continue
        seen.add(p)
        prompts.append(p)

    if len(prompts) < n_refs:
        raise RuntimeError(
            f"Could not generate {n_refs} unique ABC prompts after "
            f"{max_attempts} attempts; got {len(prompts)}. Pool too small?"
        )
    return prompts


def load_ioi_with_abc(
    n_prompts,
    n_abc_refs,
    tokenizer,
    seed=42,
    abc_seed_offset=10000,
    symmetric=True,
):
    """Convenience: generate IOI prompts and per-target ABC references in
    one call, with single-token-verified name pool.

    Returns:
        list of dicts, each containing:
            'prompt':         str        target sentence
            'target_token':   str        answer with leading space
            'distractor_token': str      the S name, leading space
            'template_type':  str        'ABBA' or 'BABA'
            'IO':             str        the IO name (leading space)
            'S':              str        the S name (leading space)
            'abc_refs':       list[str]  n_abc_refs ABC sentences
            'abc_seed':       int        seed used for ABC generation

    Use abc_seed_offset to vary ABC samples while keeping the prompt
    set fixed.
    """
    pool = verify_single_token_pool(tokenizer)
    adapter = load_ioi_dataset(
        target=n_prompts, seed=seed, names=pool["names"],
        symmetric=symmetric,
    )

    out = []
    for i, meta in enumerate(adapter.metadata):
        abc_refs = build_abc_for_target(
            meta=meta,
            n_refs=n_abc_refs,
            names=pool["names"],
            seed=abc_seed_offset + i,
        )
        out.append({
            "prompt":           meta["prompt"],
            "target_token":     meta["answer"],
            "distractor_token": meta["S"],
            "template_type":    meta["template_type"],
            "IO":               meta["IO"],
            "S":                meta["S"],
            "abc_refs":         abc_refs,
            "abc_seed":         abc_seed_offset + i,
        })
    return out