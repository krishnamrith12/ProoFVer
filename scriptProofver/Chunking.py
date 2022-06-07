#install flair by using `pip install flair'
#install huggingface "transformers" 

from flair.data import Sentence
from flair.models import SequenceTagger
import json
from tqdm import tqdm

tagger = SequenceTagger.load("flair/chunk-english")
f = open("< path to fever file>.jsonl").readlines()

mutDict = list()
for item in tqdm(f):
    jso = json.loads(item)
    mutDict.append(jso)
mutIds = dict()
for item in tqdm(mutDict):
    sentence = Sentence(item["claim"])
    tagger.predict(sentence)
    
    
    aList = list()
    for entity in sentence.get_spans(): # or try get_spans("np")
        _Dict = dict()
        _Dict["startPos"] = entity.start_pos
        _Dict["endPos"] = entity.end_pos
        _Dict["tokLists"] = [item.text for item in entity.tokens]
        _Dict["text"] = entity.text
        
        aList.append(_Dict)
    
    mutIds[item["id"]] = aList


ff = open("mutChunker.json","w")
json.dump(mutIds,ff)
ff.close()
