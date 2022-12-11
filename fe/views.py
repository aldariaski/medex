from django.shortcuts import render, redirect
from django.urls import reverse
from django.core.paginator import Paginator
# import medex
# import be.views as backend
# import medex.search as search
import time
import bsbi as bsbi
import compression as compression


# from be.views import *
# Create your views here.

def home(request):
    BSBI_instance = bsbi.BSBIIndex(data_dir='collection',
                                   postings_encoding=compression.VBEPostings,
                                   output_dir='index')
    ab = BSBI_instance.get_doctext_only()
    doc = ab[0]
    isi = ab[1]
    context = {"doc": doc,
               "isi": isi}
    return render(request, 'fe/home.html', context)


def faq(request):
    return render(request, 'fe/faq.html')


def indeksi(request):
    BSBI_instance = bsbi.BSBIIndex(data_dir='collection',
                                   postings_encoding=compression.VBEPostings,
                                   output_dir='index')
    BSBI_instance.index()
    len = BSBI_instance.get_len_only()
    context = {"len": len,
               "query": "A"}
    return render(request, 'fe/indeksi.html', context)


def results(request):
    result_pages = []

    query = ""
    # result_pages = [[1,"A"], [2,"B"]]
    if request.method == "POST":
        query = request.POST.get('search')
        iterat = 1
        BSBI_instance = bsbi.BSBIIndex(data_dir='collection',
                                       postings_encoding=compression.VBEPostings,
                                       output_dir='index')
        # BSBI_instance.index()
        for (score_text, doc) in BSBI_instance.retrieve_bm25(query, k=1000):
            score = score_text[0]
            teks = score_text[1]
            aresult = [iterat, doc, teks, score]
            result_pages.append(aresult)
            iterat += 1
            # return aresult
            # if (score > .3):
            # docs_to_letor.append(doc)
        for i in range(0):
            aresult = [iterat, "doc" + str(iterat), iterat, 0]
            result_pages.append(aresult)
            iterat += 1

    # paginator = Paginator(result_pages, per_page=-(result_pages//-8))
    context = {"query": query,
               "result_pages": result_pages, }
    # "pages":pages}
    return render(request, 'fe/results.html', context)
