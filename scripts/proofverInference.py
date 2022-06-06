from genre import GENRE
import pickle
from genre.trie import Trie
import torch
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_fairseq as get_entity_spans
from spacy.lang.en import English
nlp = English()
from tqdm import tqdm
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = GENRE.from_pretrained("<----Model checkpoint ---> ").eval()
#model = model.to(device)

#testSet = open("../GenreDataFiles/NovFiles/encTest.txt").read().split("\n")[:-1]
devSet = open("<---- data -->").read().split("\n")[:-1]
aTemp = []
for item in tqdm(devSet):
    sentences = [item]
    claim = sentences[0].split("</s>")[0]
    claim = claim.strip()
    claim = claim.split()
    del claim[0:2]

    
    #create claim spans as candidates
    k = 7
    answers = set([" " + " ".join(claim[start: start + i]) for start in range(len(claim)) for i in range(k)
            if len(claim[start: start + k]) <= k])

    answers = [item.strip() if item[0:2] == " ^"  else item for item in answers  ]
    answers.remove(" ")
    answers

    #remove spans which only have stop words
    toRemove = set()
    for item in answers:
        toks = [tok.text for tok in tokenizer(item.strip()) if tok.is_punct == False and tok.is_stop == False]
        if len(toks) < 1:
            toRemove.add(item)
    answers = set(answers) - toRemove

    
    #create evidence spans
    evidence = sentences[0].split("</s>")[1:]
    evidence =[sent.strip().split() for sent in evidence]
    evidSet = set()
    for sent in evidence:
        _temp = set([ " ".join(sent[start: start + i]) for start in range(len(sent)) for i in range(k)
        if len(sent[start: start + k]) <= k])
        evidSet = evidSet.union(_temp)

    toRemove = set()
    for item in evidSet:
        toks = [tok.text for tok in tokenizer(item.strip()) if tok.is_punct == False and tok.is_stop == False]
        if len(toks) < 1:
            toRemove.add(item)

    evidSet = evidSet - toRemove



    prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
        model,
        sentences,
        mention_trie=Trie([
            model.encode(e)[1:].tolist()
            for e in answers
        ]),
        candidates_trie=Trie([
            model.encode(" }} [ {} ]".format(e))[1:].tolist()
            for e in evidSet
        ])
    )

    a = model.sample(
        sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )
    aTemp.append(a)