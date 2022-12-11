
#import medex.be.sources as src
from .sources.bsbi import BSBIIndex
from .sources.compression import VBEPostings

# Create your views here.

def search(query):
    return bm25(query)

def bm25(query):
    results = []
    iterat = 1
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
        aresult = [iterat, doc, score]
        return aresult
        #if (score > .3):
            #docs_to_letor.append(doc)