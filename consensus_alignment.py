import random

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def label_flip(txt, labels, target):
    tokens = word_tokenize(txt)
    find_label = False
    for i in range(len(tokens)):
        token = tokens[i]
        tag = tagged[i][1]
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemma = WordNetLemmatizer().lemmatize(token, pos=wordnet_pos)
        if lemma.lower() in labels:
            txt = txt.replace(token, target)
            find_label = True
    if not find_label:
        txt = insert_word(txt, target)
    return txt


def text_augment(txt, label, target):
    txt_pool = []  
    tokens = word_tokenize(txt)
    tokens = [token for token in tokens if (token!=label and token!=target)]
    tagged = nltk.pos_tag(tokens)
    for i in range(len(tokens)):
        token = tokens[i]

        tag = tagged[i][1]
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemma = WordNetLemmatizer().lemmatize(token, pos=wordnet_pos)

        if wordnet_pos == wordnet.ADJ:
            synonyms = []
            for synset in wordnet.synsets(lemma, pos=wordnet.ADJ):
                for word in synset.lemmas():
                    if word.name() != token:
                        synonyms.append(word.name())
            if synonyms:
                txt_pool.append([txt.replace(token, synonyms[0]), 'ADJ_replace'])
        elif wordnet_pos == wordnet.VERB:
            synonyms = []
            for synset in wordnet.synsets(lemma, pos=wordnet.VERB):
                for word in synset.lemmas():
                    if word.name() != lemma:
                        synonyms.append(word.name())
            if synonyms:
                for i in range(len(synonyms)):
                    synonyms[i] = synonyms[i].replace("_", " ")
                txt_pool.extend([[txt.replace(token, syn), 'VERB_replace'] for syn in synonyms])

        elif wordnet_pos == wordnet.ADV:
            synonyms = []
            for synset in wordnet.synsets(lemma, pos=wordnet.ADV):
                for word in synset.lemmas():
                    if word.name() != lemma:
                        synonyms.append(word.name())
            if synonyms:
                txt_pool.append([txt.replace(token, synonyms[0]), 'ADV_replace'])

        if lemma == label:
            synsets = wordnet.synsets(lemma)
            hypernyms = synsets[0].hypernyms()
            txt = txt.replace(token, target)  
            if hypernyms:
                hypernyms = hypernyms[0].lemmas()[0].name()
                txt_pool.append([txt.replace(token, hypernyms), 'label_hypernyms_replace'])
            hyponyms = synsets[0].hyponyms()
            if hyponyms:
                hyponyms = hyponyms[0].lemmas()[0].name()
                txt_pool.append([txt.replace(token, hyponyms), 'label_hyponyms_replace'])

            meronyms = synsets[0].part_meronyms()
            if meronyms:
                meronyms = meronyms[0].lemmas()[0].name()
            holonyms = synsets[0].part_holonyms()
            if holonyms:
                holonyms = holonyms[0].lemmas()[0].name()
    return txt_pool



def has_duplicates(lst):
    return len(lst) != len(set(lst))

