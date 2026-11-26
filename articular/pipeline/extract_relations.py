"""Read each article's first sentence with a grammatical parser and label what its links say.

    uv run python -m articular.pipeline.extract_relations --wiki enwiki --workers 8

Reads first sentences from data/<wiki>_links/pages/ (articular/pipeline/extract_links.py) and writes
data/<wiki>_relations/*.parquet, one row per labeled link (and one for an unlinked kind):

    page_id, relation, prep, target, head, phrase, pos

relation is one of:
  kind        the head noun of "X is a …": what kind of thing the article is
  kind_also   a further head noun joined by "and"/"or" ("singer and songwriter")
  kind_part   a link inside the kind's noun phrase that is not on the head ("[[brand]] name")
  kind_text   the kind is not linked; `head` holds the noun and `target` is empty
  qualifier   a link describing the kind ("an [[American]] singer")
  prep        a link the kind is tied to by a preposition, given in `prep` ("capital of [[France]]")
  clause      a link in a clause hanging off the kind ("evidence obtained through [[sense]]")
  field       a link in an opening phrase ("In [[physics]], …"), with its preposition in `prep`
  aside       a link inside parentheses
  other       any other link in the sentence
`head` is the kind's noun (dictionary form) on every row of an article that has one, and `phrase` is that
noun with the words joined to it ("ice hockey player"), for matching to an article title.

The kind is looked for in the first "X is a …" of the sentence, wherever the parser puts the main verb,
and in "X is defined as a …" and "X refers to a …". Three kinds of noun are seen through: "a type of Y"
and "a species of Y" give Y, "one of the Ys" gives Y, and "someone who …" gives person.
Each worker parses and labels its own share of the articles; finished parts are skipped on a rerun.
"""

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import spacy
from tqdm import tqdm

PIECE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]|'{2,}|\(|\)|<[^>]+>|&nbsp;|&ndash;|&mdash;")
SKIP_PREFIX = ("file:", "image:", "category:", "media:")
SCHEMA = pa.schema([("page_id", pa.int64()), ("relation", pa.string()), ("prep", pa.string()),
                    ("target", pa.string()), ("head", pa.string()), ("phrase", pa.string()), ("pos", pa.int32())])
# nouns that only say the subject is some sort of the thing named after "of" ("specie" is the parser's "species")
SORT_NOUNS = {"type", "kind", "form", "sort", "variety", "species", "specie", "subspecies", "subspecie", "genus",
              "breed", "style", "example", "member"}
SOMEONE = {"someone", "somebody", "anyone", "anybody"}
NOUNS = ("NOUN", "PROPN")


def norm_title(target):
    t = target.split("#")[0].replace("_", " ").strip(" :")
    t = " ".join(t.split())
    return t[:1].upper() + t[1:]


def plain(sentence):
    """Markup-free text for the parser, the links' character spans in it, and links inside parentheses."""
    out, spans, asides, depth, pos, length = [], [], [], 0, 0, 0

    def emit(text):
        nonlocal length
        if depth == 0:
            out.append(text)
            length += len(text)

    for m in PIECE.finditer(sentence):
        emit(sentence[pos:m.start()])
        pos, tok = m.end(), m.group()
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth = max(0, depth - 1)
        elif tok.startswith("[["):
            if m.group(1).strip(" :").lower().startswith(SKIP_PREFIX):
                continue
            target, label = norm_title(m.group(1)), m.group(2) or m.group(1)
            if not target:
                continue
            if depth:
                asides.append(target)
            else:
                spans.append((length, length + len(label), target))
                emit(label)
        elif tok[0] in "<&":
            emit(" ")
    emit(sentence[pos:])
    return "".join(out), spans, asides


def after(token, preposition):
    """The noun that follows `preposition` under this token: "type of [moth]", "defined as [a movement]"."""
    for prep in token.children:
        if prep.dep_ == "prep" and prep.lower_ == preposition:
            return next((c for c in prep.children if c.dep_ == "pobj" and c.pos_ in NOUNS), None)
    return None


def find_kind(sent):
    """(verb, noun, word): the verb that states the kind, the noun token that names it, and the word to record."""
    verb = head = None
    for token in sent:                     # the first statement of what the subject is, in reading order
        if token.lemma_ == "be" and (token is sent.root or any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children)):
            head = next((c for c in token.children if c.dep_ == "attr" and c.pos_ in NOUNS + ("PRON", "NUM")), None)
        elif token.lemma_ in ("define", "describe", "classify", "regard"):
            head = after(token, "as")
        elif token.lemma_ == "refer":
            head = after(token, "to")
        if head is not None:
            verb = token
            break
    if head is None:
        root = sent.root                   # a verbless fragment: treat its noun as the kind
        return (None, root, root.lemma_.lower()) if root.pos_ in NOUNS else (None, None, None)
    if head.lower_ in SOMEONE:
        return verb, head, "person"
    if head.pos_ in ("PRON", "NUM") or head.lemma_.lower() in SORT_NOUNS:
        inner = after(head, "of")          # "a species of [moth]", "one of the [divisions]"
        if inner is not None:
            head = inner
        elif head.pos_ in ("PRON", "NUM"):
            return verb, None, None
    return verb, head, head.lemma_.lower()


def phrase_of(head, word):
    """The kind noun with the words joined to it, lower case: "ice hockey player"."""
    words, i = [word], head.i - 1
    while i >= head.left_edge.i and len(words) < 4 and head.doc[i].dep_ in ("compound", "amod") and head.doc[i].is_alpha:
        words.insert(0, head.doc[i].lower_)
        i -= 1
    return " ".join(words)


def label_links(doc, spans, asides):
    """Rows (relation, prep, target, head, phrase, pos) for one parsed sentence."""
    rows, used = [], set()

    def link_of(token):
        for n, (start, end, _) in enumerate(spans):
            if token.idx < end and token.idx + len(token) > start:
                return n
        return None

    def add(token, relation, prep=None):
        n = link_of(token)
        if n is None or n in used:
            return False
        used.add(n)
        rows.append((relation, prep, spans[n][2], n))
        return True

    sent = next(doc.sents, None)
    verb, head, word = find_kind(sent) if sent is not None else (None, None, None)
    if verb is not None and verb.lemma_ == "be":
        subject = next((c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        for c in verb.children:          # "In physics, X is …"
            if c.dep_ == "prep" and subject is not None and c.i < subject.i:
                for obj in c.subtree:
                    if obj.dep_ in ("pobj", "conj", "compound", "amod"):
                        add(obj, "field", c.lemma_.lower())
    if head is not None:
        if word == "person" and head.lower_ in SOMEONE or not add(head, "kind"):
            rows.append(("kind_text", None, None, -1))
        for other in head.conjuncts:
            add(other, "kind_also")
        for c in head.children:
            if c.dep_ == "prep":
                for obj in c.subtree:
                    if obj is not c:
                        add(obj, "prep", c.lemma_.lower())
            elif c.dep_ in ("acl", "relcl", "advcl"):
                for inner in c.subtree:
                    add(inner, "clause")
        for left in head.lefts:
            for inner in left.subtree:
                # a link that reaches the head itself was taken above; one that stops short is part of the phrase
                add(inner, "qualifier" if left.dep_ in ("amod", "nmod", "poss", "advmod", "npadvmod") else "kind_part")
    for n in range(len(spans)):
        if n not in used:
            rows.append(("other", None, spans[n][2], n))
    rows += [("aside", None, target, -1) for target in asides]
    phrase = phrase_of(head, word) if head is not None and head.lower_ not in SOMEONE else word
    return [(relation, prep, target, word, phrase, pos) for relation, prep, target, pos in rows]


def sentences(name, part, parts):
    """Yield (text, (page_id, spans, asides)) for every `parts`-th article of one pages file."""
    table = pq.read_table(name, columns=["page_id", "redirect_target", "is_disambig", "first_sentence"]).to_pylist()
    for row in table[part::parts]:
        if row["redirect_target"] is None and not row["is_disambig"] and row["first_sentence"]:
            text, spans, asides = plain(row["first_sentence"])
            yield text, (row["page_id"], spans, asides)


def label_file(job):
    """Parse one share of one pages file; parsing and labeling both happen in this worker."""
    name, part, parts, out, model = job
    final = out / f"{name.stem}-{part}of{parts}.parquet"
    if final.exists():
        return None, None
    nlp = spacy.load(model, disable=["ner"])
    columns, n_sentences = {column: [] for column in SCHEMA.names}, 0
    for doc, (page_id, spans, asides) in nlp.pipe(sentences(name, part, parts), as_tuples=True, batch_size=1000):
        n_sentences += 1
        for row in label_links(doc, spans, asides):
            for column, value in zip(SCHEMA.names, (page_id, *row)):
                columns[column].append(value)
    temp = final.with_suffix(".tmp")
    pq.write_table(pa.table(columns, schema=SCHEMA), temp)
    temp.rename(final)          # only complete files get their final name
    return n_sentences, len(columns["page_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="en_core_web_sm")
    args = parser.parse_args()

    out = args.data / f"{args.wiki}_relations"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((args.data / f"{args.wiki}_links" / "pages").glob("*.parquet"), key=lambda p: -p.stat().st_size)
    parts = max(1, -(-args.workers // len(files)))      # split files when there are fewer files than workers
    jobs = [(name, part, parts, out, args.model) for name in files for part in range(parts)]
    n_sentences = n_rows = 0
    with Pool(args.workers) as pool:
        for sentences_done, rows_done in tqdm(pool.imap_unordered(label_file, jobs), total=len(jobs), unit="part"):
            n_sentences += sentences_done or 0
            n_rows += rows_done or 0
    print(f"{out}: {n_sentences:,} sentences, {n_rows:,} labeled links (parts already present were skipped)")


if __name__ == "__main__":
    main()
