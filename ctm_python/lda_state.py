# encoding: utf-8


import os
import re
import string
# import glob
import logging

from gensim import corpora, models

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# improved list from Stone, Denis, Kwantes (2010)
STOPWORDS = """
a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can
cannot cant co computer con could couldnt cry de describe
detail did do doesn done down due during
each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire first five for former formerly forty found four from front full further get give go
had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred i ie
if in inc indeed interest into is it its itself keep last latter latterly least less ltd
just
kg km
made many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely
neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off
often on once one only onto or other others otherwise our ours ourselves out over own part per
perhaps please put rather re
quite
rather really regarding
same see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system take ten
than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under
until up unless upon us used using
various very very via
was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves
"""
STOPWORDS = frozenset(w for w in STOPWORDS.split() if w)


def remove_stopwords(s):
    return " ".join(w for w in s.split() if w not in STOPWORDS)


def strip_punctuation(s):
    return re.sub("([%s]+)" % string.punctuation, " ", s)


def strip_punctuation2(s):
    return s.translate(string.maketrans("", ""), string.punctuation)


def strip_tags(s):
    # assumes s is already lowercase
    return re.sub(r"<([^>]+)>", "", s)


def strip_short(s, minsize=3):
    return " ".join(e for e in s.split() if len(e) >= minsize)


def strip_numeric(s):
    return re.sub(r"[0-9]+", "", s)


def strip_non_alphanum(s):
    # assumes s is already lowercase
    # FIXME replace with unicode compatible regexp, without the assumption
    return re.sub(r"[^a-z0-9\ ]", " ", s)


def strip_multiple_whitespaces(s):
    return re.sub(r"(\s|\\n|\\r|\\t)+", " ", s)
    #return s


def split_alphanum(s):
    s = re.sub(r"([a-z]+)([0-9]+)", r"\1 \2", s)
    return re.sub(r"([0-9]+)([a-z]+)", r"\1 \2", s)

DEFAULT_FILTERS = [str.lower, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short]


def preprocess_string(s, filters=DEFAULT_FILTERS):
    for f in filters:
        s = f(s)
    return s.split()


def preprocess_documents(docs):
    return map(preprocess_string, docs)


# get the folder name which containing all the files
filenames = os.listdir('/Users/sean/personal_research_doc/lda_explained/state')
# print 'filenames', filenames
# initialize the list to store all the filenames


#initial a string to save all the file contents
state_corpus = []
for thefile in filenames:
    # print 'the file is ', thefile
    with open("./state/" + thefile, "rb") as f:
        strings = f.read()
        state_corpus.append(strings)

print "there are ", len(state_corpus), "document"
for doc in state_corpus:
    print len(doc)
state = preprocess_documents(state_corpus)
# print hp_corpus_processed[2][40:100]
print 'corpus processed'

# make dictionary
dictionary = corpora.Dictionary(state)
# save it to disk for further use
dictionary.save('state_dict.dict')

# make corpus
corpus = [dictionary.doc2bow(hpp) for hpp in state]
# save it to disk for further use, adopting mm format
corpora.MmCorpus.serialize('state_corpus.mm', corpus)

# build lda model
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100, update_every=1, chunksize=1000, passes=10)
lda.print_topics(20)
