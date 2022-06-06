from simalign import SentenceAligner #available at https://github.com/cisnlp/simalign
import torch
import re
import json
from tqdm import tqdm
import pickle

import spacy #install space for rule based tokenisation
from spacy.lang.en import English
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer


#preprocess fever sentences and wikipedia evidence sentences
def process_sent(sentence):
    sentence = re.sub(" LSB.*?RSB", "", sentence)
    sentence = re.sub("LRB RRB ", "", sentence)
    sentence = re.sub("LRB", " ( ", sentence)
    sentence = re.sub("RRB", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    
    return sentence

def process_wiki_title(title):
    title = re.sub("_", " ", title)
    title = re.sub("-LRB-", " ( ", title)
    title = re.sub("LRB", " ( ", title)
    title = re.sub("-RRB-", " )", title)
    
    title = re.sub("RRB", " )", title)
    title = re.sub("COLON", ":", title)
    return title
    
print("loading aligner")
myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai",device="cuda:0")

"""
devDict = dict()
for item in tqdm(k):
    jso = json.loads(item)
    devDict[jso["id"]] = jso
"""

devDict = {"id": 75397, "evidence": [["Fox_Broadcasting_Company", 0, "The Fox Broadcasting Company LRB often shortened to Fox and stylized as FOX RRB is an American English language commercial broadcast television network that is owned by the Fox Entertainment Group subsidiary of 21st Century Fox ."], ["Nikolaj_Coster-Waldau", 7, "He then played Detective John Amsterdam in the short lived Fox television series New Amsterdam LRB 2008 RRB , as well as appearing as Frank Pike in the 2009 Fox television film Virtuality , originally intended as a pilot ."]], "label": "SUPPORTS", "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."}
#load your dictionary here
for item in devDict:
    evidList = list()        
    for thing in devDict[item]["evidence"]:
        assert(len(thing)) == 3
        astr = process_wiki_title(thing[0]) + " "
        astr += process_sent(thing[2])
        evidList.append(astr)        
    devDict[item]["linEvid"] = evidList

for item in tqdm(devDict):
  claimToks = [stuff.text for stuff in tokenizer(devDict[item]["claim"])]
  devDict[item]["claimToks"] = claimToks

  evidA = list()
  for thing in devDict[item]["linEvid"]:
      evidToks = [stuff.text for stuff in tokenizer(thing)]
      evidA.append(evidToks)

  devDict[item]["evidToksList"] = evidA
    
 
#Aligning claim with each evidence sentence
alignedDict = dict()
for item in tqdm(devDict):
    claimToks = devDict[item]["claimToks"]
    alignedList = list()
    for stuff in devDict[item]["evidToksList"]:
        #Aligning claim with each evidence sentence
        alginments = myaligner.get_word_aligns(claimToks, stuff)
        alignedList.append(alginments)
    alignedDict[item] = alignedList
    
    
    
    