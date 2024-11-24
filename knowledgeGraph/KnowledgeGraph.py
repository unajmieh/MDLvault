import pandas as pd
from tqdm import tqdm
import networkx as nx
import bs4
import requests
import spacy
import re
import numpy as np
import pandas as pd
import os
import csv
import wikipedia
from spacy import displacy
from IPython.display import Image
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from spacy.matcher import Matcher
from pyvis.network import Network
from spacy.symbols import nsubj, VERB
from urllib.request import urlopen
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

nlp = spacy.load('en_core_web_sm')
customized_topic = input('To see a knowledge graph Please enter a wikiperdia subject :')
url = 'https://en.wikipedia.org/w/api.php'
params = {
            'action': 'parse',
            'page': customized_topic,
            'format': 'json',
            'prop':'text',
            'redirects':''
        }
response = requests.get(url, params=params)
data = response.json()
raw_html = data['parse']['text']['*']
#----- The end of Wikipedia API to retrieve wikipedia articles ----
# Specify url of the web page
# source_1 = urlopen(response).read()
soup = BeautifulSoup(raw_html,'html.parser')
soup
# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))
# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))
# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)
# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
tokeneized = nltk.sent_tokenize(text)
df1 = pd.DataFrame(tokeneized)
# print(df1.shape)
headers =  ["Sentences"]
df1.columns = headers
data_top = df1.head()
# print(data_top)
#------------ The start of entity extraction ---
def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5  
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    return [ent1.strip(), ent2.strip()]
##------ end of entity extraction 
entity_pairs = []
for i in tqdm(df1["Sentences"]):
    entity_pairs.append(get_entities(i))

print(entity_pairs)
#--------- Relation extraction 
def get_subject_phrase(doc):
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

def get_object_phrase(doc):
    for token in doc:
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]       

def get_verb_phrase(doc):
    for token in doc:
         if (token.pos_ == 'Verb'):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]    

#for sentence in Sentences:
    #doc = nlp(sentences)
    subject_phrase = get_subject_phrase(doc)
    object_phrase = get_object_phrase(doc)
    verb_phrase = get_verb_phrase(doc)
    # print(subject_phrase)
    # print(object_phrase)                 


def get_relation(sent):
    doc = nlp(sent)
    matcher = Matcher(nlp.vocab)    
    
    pattern = [
                    {'DEP':'ROOT'},
                    {'DEP':'prep','OP':"?"},
                    {'DEP':'agent','OP':"?"},
                    {'POS':'ADJ','OP':"?"}
    ]

    matcher.add("matching_sentence_structure", [pattern])
    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return(span.text)

# print(get_relation("we started the work"))    
relations = [get_relation(i) for i in tqdm(df1['Sentences'])]
pd.Series(relations).value_counts()[:50]
source = [i[0] for i in entity_pairs] # extract subject
target = [i[1] for i in entity_pairs] # extract object
edge_data = zip(source, target, relations)
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
print(kg_df)

###### The end of Relation extraction #################################
###### Here we create a directed-graph here from a dataframe #########
G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(16,16))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='purple', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()
net = Network(directed=True, width="700px", height="700px", bgcolor="#eeeeee")
for e in edge_data:
                src = e[0]
                dst = e[1]
                w = e[2]

                net.add_node(src, src, title=src)
                net.add_node(dst, dst, title=dst)
                net.add_edge(src, dst, value=w)
net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
net.set_edge_smooth('dynamic')
# conet.show("KnowledgeGraph.html")
