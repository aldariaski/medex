# Sumber dilihat dari beberapa laman di internet
# https://github.com/har07/PySastrawi
# https://blog.csdn.net/jianglingbixin/article/details/109552318
# https://github.com/Lichuanro/Information-Retrieval-and-Web-Search

from bsbi import BSBIIndex
from compression import VBEPostings
#from letor import import_data, phi, linear, loss, reg, rmse, \
    #mu_simple, preproses, sigma_simple, closed_form_sol, sgd
from letor import letor
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir ='collection', \
                          postings_encoding = VBEPostings, \
                          output_dir ='index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]
for query in queries:
    print("Query  : ", query)
    print("Results:")

    docs_to_letor = []
    print("Hasil dengan BM25")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
        if (score > .3):
            docs_to_letor.append(doc)

        print(f"{doc:30} {score:>.3f}")

        #print("Hasil dengan Letor")
        #print("doc"+doc)

    print("\n\n")
    print("Hasil BM25 diolah Letor")
    letor(docs_to_letor, query)
    #print(docs_to_letor)

    print("\n\n")
    print("Hasil dengan TF IDF")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):

        print(f"{doc:30} {score:>.3f}")# x, y = import_data(doc)
        # # x = np.hstack((np.ones((x.shape[0], 1)), x))
        # n, m = x.shape
        #
        # # Data partition
        # n_train = int(0.8 * n)
        # n_valid = int((n - n_train) / 2)
        #
        # x_train = x[:n_train, :]
        # x_valid = x[n_train:n_train + n_valid, :]
        # x_test = x[n_train + n_valid:, :]
        #
        # x_train, idx = preproses(x_train)
        # x_valid = x_valid[:, idx]
        # x_test = x_test[:, idx]
        #
        # y_train = y[:n_train, :]
        # y_valid = y[n_train:n_train + n_valid, :]
        # y_test = y[n_train + n_valid:, :]
        #
        # l = 10
        # lamb = 1
        # lr_low = 10e-17
        # lr_high = 10e-10
        # epochs = 50
        # mbatch_size = 50
        #
        # mu = mu_simple(x_train, l)
        # sigma = sigma_simple(x_train, l)
        #
        # # Random initialization
        # w0 = np.random.uniform(0, 1, (len(mu) + 1, 1))
        # p0 = linear(x_test, w0, mu, sigma)
        # rmse0 = rmse(y_test, p0, w0, lamb, y_test.shape[0])
        # print("Error with random weights: " + str(rmse0))
        #
        # # Closed form solution
        # w = closed_form_sol(x_train, y_train, mu, sigma, lamb)
        #
        # p = linear(x_test, w, mu, sigma)
        # print(w)
        # rmse0 = rmse(y_test, p, w, lamb, y_test.shape[0])
        # print("Error using closed form solution: " + str(rmse0))
        #
        # # SGD
        # w = sgd(x_train, y_train, lamb, lr_low, lr_high, mu, sigma, epochs, mbatch_size)
        # p = linear(x_test, w, mu, sigma)
        # print(w)
        # rmse0 = rmse(y_test, p, w, lamb, y_test.shape[0])
        # print("Error using SGD:" + str(rmse0))

    #Letor

    print()



