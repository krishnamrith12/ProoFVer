# GENRE for fairseq

First make sure that you have [fairseq](https://github.com/pytorch/fairseq) installed. Please install it using: 
```bash 
git clone https://github.com/pytorch/fairseq 
cd fairseq 
pip install --editable ./ 
``` 
as described in the [fairseq repository](https://github.com/pytorch/fairseq#requirements-and-installation) since `pip install fairseq` has issues. 

## Entity Disambiguation
Download one of the pre-trained models:

| Training Dataset | Model |
| -------- | -------- |
| BLINK | [fairseq_entity_disambiguation_blink](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_blink.tar.gz)|
| BLINK + AidaYago2 | [fairseq_entity_disambiguation_aidayago](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_aidayago.tar.gz)|

as well as the prefix tree from KILT Wikipedia titles ([kilt_titles_trie.pkl](http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie.pkl)).

Then load the trie and define the function to apply the constraints with the entities trie


```python
import pickle

with open("data/kilt_titles_trie.pkl", "rb") as f:
    trie = pickle.load(f)

def prefix_allowed_tokens_fn(batch_id, sent):
    return trie.get(sent.tolist())
```

Then, load the model


```python
from genre import GENRE
model = GENRE.from_pretrained("models/fairseq_entity_disambiguation_aidayago").eval()
```

and simply use `.sample` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
sentences = ["[START_ENT] Armstrong [END_ENT] was the first man on the Moon."]

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'Neil Armstrong', 'logprob': tensor(-0.1443)},
      {'text': 'William Armstrong', 'logprob': tensor(-1.4650)},
      {'text': 'Scott Armstrong', 'logprob': tensor(-1.7311)},
      {'text': 'Arthur Armstrong', 'logprob': tensor(-1.7356)},
      {'text': 'Rob Armstrong', 'logprob': tensor(-1.7426)}]]



## Document Retieval
Download one of the pre-trained models:

| Training Dataset | Models |
| -------- | -------- |
| KILT | [fairseq_wikipage_retrieval](http://dl.fbaipublicfiles.com/GENRE/fairseq_wikipage_retrieval.tar.gz)|

Then, load the model


```python
model = GENRE.from_pretrained("models/fairseq_wikipage_retrieval").eval()
```

and simply use `.sample` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
sentences = ["Armstrong was the first man on the Moon."]

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'Neil Armstrong', 'logprob': tensor(-0.1593)},
      {'text': 'Apollo 11', 'logprob': tensor(-0.7673)},
      {'text': 'Astronaut', 'logprob': tensor(-1.1418)},
      {'text': 'Buzz Aldrin', 'logprob': tensor(-1.4446)},
      {'text': 'Apollo 17', 'logprob': tensor(-1.4594)}]]



## End-to-End Entity Linking

Download one of the pre-trained models:

| Training Dataset | Models |
| -------- | -------- |
| WIKIPEDIA | [fairseq_e2e_entity_linking_wiki_abs](http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_wiki_abs.tar.gz)|
| WIKIPEDIA + AidaYago2 | [fairseq_e2e_entity_linking_aidayago](http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_aidayago.tar.gz)|

Then, load the model


```python
model = GENRE.from_pretrained("models/fairseq_e2e_entity_linking_aidayago").eval()
```

and 
1. get the `prefix_allowed_tokens_fn` with the only constraints to annotate the original sentence (i.e., no other constrains on mention nor candidates)
2. use `.sample` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq

sentences = ["In 1921, Einstein received a Nobel Prize."]

prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(model, sentences)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ].',
       'logprob': tensor(-0.9068)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ] {. } [ Nobel Prize in Physiology or Medicine ]',
       'logprob': tensor(-0.9301)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ] {. } [ Albert Einstein ]',
       'logprob': tensor(-0.9943)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Physiology ].',
       'logprob': tensor(-1.0778)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ] {. } [ Ernest Einstein ]',
       'logprob': tensor(-1.1164)}]]



You can constrain the mentions with a prefix tree (no constrains on candidates)


```python
from genre.trie import Trie

prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(e)[1:].tolist()
        for e in [" Einstein"]
    ])
)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a Nobel Prize.',
       'logprob': tensor(-1.4009)},
      {'text': 'In 1921, { Einstein } [ Einstein (crater) ] received a Nobel Prize.',
       'logprob': tensor(-1.6665)},
      {'text': 'In 1921, { Einstein } [ Albert Albert Einstein ] received a Nobel Prize.',
       'logprob': tensor(-1.7498)},
      {'text': 'In 1921, { Einstein } [ Ernest Einstein ] received a Nobel Prize.',
       'logprob': tensor(-1.8327)},
      {'text': 'In 1921, { Einstein } [ Max Einstein ] received a Nobel Prize.',
       'logprob': tensor(-1.8757)}]]



You can constrain the candidates with a prefix tree (no constrains on mentions)


```python
prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(
    model,
    sentences,
    candidates_trie=Trie([
        model.encode(" }} [ {} ]".format(e))[1:].tolist()
        for e in ["Albert Einstein", "Nobel Prize in Physics", "NIL"]
    ])
)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ].',
       'logprob': tensor(-0.8925)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize. } [ Nobel Prize in Physics ]',
       'logprob': tensor(-0.8990)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize } [ Nobel Prize in Physics ].',
       'logprob': tensor(-0.9330)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ] {. } [ Nobel Prize in Physics ]',
       'logprob': tensor(-0.9781)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ] {. } [ Albert Einstein ]',
       'logprob': tensor(-0.9854)}]]



You can constrain the candidate sets given a mention (no constrains on mentions)


```python
prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(
    model,
    sentences,
    mention_to_candidates_dict={
        "Einstein": ["Einstein"],
        "Nobel": ["Nobel Prize in Physics"],
    }
)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize.',
       'logprob': tensor(-1.5417)},
      {'text': 'In 1921, { Einstein } [ Einstein ] received a Nobel Prize.',
       'logprob': tensor(-2.1319)},
      {'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize } [ NIL ].',
       'logprob': tensor(-2.2816)},
      {'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize. } [ NIL ]',
       'logprob': tensor(-2.3914)},
      {'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel Prize. } [ NIL ]',
       'logprob': tensor(-2.9078)}]]



A combiation of these constraints is also possible


```python
prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ].',
       'logprob': tensor(-0.8925)},
      {'text': 'In 1921, { Einstein } [ Einstein (surname) ] received a { Nobel Prize } [ Nobel Prize in Physics ].',
       'logprob': tensor(-1.3275)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a Nobel Prize.',
       'logprob': tensor(-1.4009)},
      {'text': 'In 1921, Einstein received a { Nobel Prize } [ Nobel Prize in Physics ].',
       'logprob': tensor(-1.8266)},
      {'text': 'In 1921, Einstein received a Nobel Prize.',
       'logprob': tensor(-3.4495)}]]



Finally, you can also use some functions from `genre.utils` that wraps pre- and post-processing of strings (e.g., normalization and outputs the character offsets and length of the mentions)


```python
from genre.utils import get_entity_spans_fairseq

get_entity_spans_fairseq(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)
```




    [[(9, 8, 'Albert_Einstein'), (29, 11, 'Nobel_Prize_in_Physics')]]



and with the `entity_spans` generate Markdown with clickable links


```python
from genre.utils import get_markdown
from IPython.display import Markdown

entity_spans = get_entity_spans_fairseq(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)

Markdown(get_markdown(sentences, entity_spans)[0])
```




In 1921, [Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) received a [Nobel Prize](https://en.wikipedia.org/wiki/Nobel_Prize_in_Physics).



## Custom End-to-End Entity Linking evaluation

We have some useful function to evaluate End-to-End Entity Linking predictions. Let's suppose we have a `Dict[str, str]` with document IDs and text as well as the gold entites spans as a `List[Tuple[str, int, int, str]]` containing documentID, start offset, length and entity title respectively.


```python
documents = {
    "id_0": "In 1921, Einstein received a Nobel Prize.",
    "id_1": "Armstrong was the first man on the Moon.",
}

gold_entities = [
    ("id_0", 3, 4, "1921"),
    ("id_0", 9, 8, 'Albert_Einstein'),
    ("id_0", 29, 11, 'Nobel_Prize_in_Physics'),
    ("id_1", 0, 9, 'Neil_Armstrong'),
    ("id_1", 35, 4, 'Moon'),
]
```

Then we can get preditions and using `get_entity_spans_fairseq` to have spans. `guess_entities` is then a `List[List[Tuple[int, int, str]]]` containing for each document, a list of entity spans (without the document ID). We further need to add documentIDs to `guess_entities` and remove the nested list to be compatible with `gold_entities`.


```python
guess_entities = get_entity_spans_fairseq(
    model,
    list(documents.values()),
)

guess_entities = [
    (k,) + x
    for k, e in zip(documents.keys(), guess_entities)
    for x in e
]
```

Finally, we can import all functions from `genre.utils` to compute scores.


```python
from genre.utils import (
    get_micro_precision,
    get_micro_recall,
    get_micro_f1,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
)

micro_p = get_micro_precision(guess_entities, gold_entities)
micro_r = get_micro_recall(guess_entities, gold_entities)
micro_f1 = get_micro_f1(guess_entities, gold_entities)
macro_p = get_macro_precision(guess_entities, gold_entities)
macro_r = get_macro_recall(guess_entities, gold_entities)
macro_f1 = get_macro_f1(guess_entities, gold_entities)
```


```python
print(
   "micro_p={:.4f} micro_r={:.4f}, micro_f1={:.4f}, macro_p={:.4f}, macro_r={:.4f}, macro_f1={:.4f}".format(
       micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1
   )
)
```

    micro_p=0.2500 micro_r=0.4000, micro_f1=0.3077, macro_p=0.2500, macro_r=0.4167, macro_f1=0.3095



```python
assert 2 / 8 == micro_p
assert 2 / 5 == micro_r
assert 2 * (2 / 8 * 2 / 5) / (2 / 8 + 2 / 5) == micro_f1
assert (1 / 4 + 1 / 4) / 2 == macro_p
assert (1 / 3 + 1 / 2) / 2 == macro_r
assert (2 * (1 / 4 * 1 / 3) / (1 / 4 + 1 / 3) + 2 * (1 / 4 * 1 / 2) / (1 / 4 + 1 / 2)) / 2 == macro_f1
```
