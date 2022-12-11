import lightgbm
import numpy as np
import random

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary

#SUMBER https://colab.research.google.com/drive/1zsmcwN5fNBrVQzvE1YPEn8gJQIHXL8wa#scrollTo=wdZ5LzgiFJd8
#Diolah edit sendiri agar berjalan disini

def letor(filelocs, query_in):

    documents = {}
    #nfcorpus
    with open("nfcorpus/train.docs") as file:
    #with open("./collection/"+fileloc) as file:
    #with open("./collection/"+fileloc, 'r') as file:
      for line in file:
        #print("line")
        #print(line)
        doc_id, content = line.split("\t")
        #print("konten atas")
        #print(content)
        documents[doc_id] = content.split()

    # test untuk melihat isi dari 2 dokumen
    #print(documents["MED-329"])
    #print(documents["MED-330"])

    queries = {}
    with open("nfcorpus/train.vid-desc.queries", encoding='latin1') as file:
    #with open("./collection/"+fileloc, encoding='latin1') as file:
      #print("file")
      #print(file)
      for line in file:
        q_id, content = line.split("\t")
        queries[q_id] = content.split()

    # test untuk melihat isi dari 2 query
    #print(queries["PLAIN-2428"])
    #print(queries["PLAIN-2435"])

    # melalui qrels, kita akan buat sebuah dataset untuk training
    # LambdaMART model dengan format
    #
    # [(query_text, document_text, relevance), ...]
    #
    # relevance awalnya bernilai 1, 2, 3 --> tidak perlu dinormalisasi
    # biarkan saja integer (syarat dari library LightGBM untuk
    # LambdaRank)
    #
    # relevance level: 3 (fully relevant), 2 (partially relevant), 1 (marginally relevant)


    NUM_NEGATIVES = 1

    q_docs_rel = {} # grouping by q_id terlebih dahulu
    with open("nfcorpus/train.3-2-1.qrel") as file:
      for line in file:
        q_id, _, doc_id, rel = line.split("\t")
        if (q_id in queries) and (doc_id in documents):
          if q_id not in q_docs_rel:
            q_docs_rel[q_id] = []
          q_docs_rel[q_id].append((doc_id, int(rel)))

    # group_qid_count untuk model LGBMRanker
    group_qid_count = []
    dataset = []
    for q_id in q_docs_rel:
      docs_rels = q_docs_rel[q_id]
      group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
      for doc_id, rel in docs_rels:
        dataset.append((queries[q_id], documents[doc_id], rel))
      # tambahkan satu negative (random sampling saja dari documents)
      dataset.append((queries[q_id], random.choice(list(documents.values())), 0))

    # test
    #print("number of Q-D pairs:", len(dataset))
    #print("group_qid_count:", group_qid_count)
    assert sum(group_qid_count) == len(dataset), "ada yang salah"
    #print(dataset[:2])



    # bentuk dictionary, bag-of-words corpus, dan kemudian Latent Semantic Indexing
    # dari kumpulan 3612 dokumen.
    NUM_LATENT_TOPICS = 200

    dictionary = Dictionary()
    bow_corpus = [dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
    model = LsiModel(bow_corpus, num_topics = NUM_LATENT_TOPICS) # 200 latent topics

    # test melihat representasi vector dari sebuah dokumen & query
    def vector_rep(text):
      rep = [topic_value for (_, topic_value) in model[dictionary.doc2bow(text)]]
      return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

    #print(vector_rep(documents["MED-329"]))
    #print(vector_rep(queries["PLAIN-2435"]))

    # kita ubah dataset menjadi terpisah X dan Y
    # dimana X adalah representasi gabungan query+document,
    # dan Y adalah label relevance untuk query dan document tersebut.
    #
    # Bagaimana cara membuat representasi vector dari gabungan query+document?
    # cara simple = concat(vector(query), vector(document)) + informasi lain
    # informasi lain -> cosine distance & jaccard similarity antara query & doc

    from scipy.spatial.distance import cosine

    def features(query, doc):
      v_q = vector_rep(query)
      v_d = vector_rep(doc)
      q = set(query)
      d = set(doc)
      cosine_dist = cosine(v_q, v_d)
      jaccard = len(q & d) / len(q | d)
      return v_q + v_d + [jaccard] + [cosine_dist]

    X = []
    Y = []
    for (query, doc, rel) in dataset:
      X.append(features(query, doc))
      Y.append(rel)

    # ubah X dan Y ke format numpy array
    X = np.array(X)
    Y = np.array(Y)

    #print(X.shape)
    #print(Y.shape)



    ranker = lightgbm.LGBMRanker(
                        objective="lambdarank",
                        boosting_type = "gbdt",
                        n_estimators = 100,
                        importance_type = "gain",
                        metric = "ndcg",
                        num_leaves = 40,
                        learning_rate = 0.02,
                        max_depth = -1)

    # di contoh kali ini, kita tidak menggunakan validation set
    # jika ada yang ingin menggunakan validation set, silakan saja
    ranker.fit(X, Y,
               group = group_qid_count,
               verbose = 10)

    # test, prediksi terhadap training data itu sendiri
    ranker.predict(X)


    #docs=dict()
    #docs = []
    #with open("./collection/"+fileloc) as file:
    # # with open("./collection/"+fileloc, 'r') as file:
    #     iterat = 1
    #     for line in file:
    #         docs.append([])
    #         # print("line")
    #         # print(line)
    #         #doc_id, content = line.split("\t")
    #         #documents[doc_id] = content.split()
    #
    #         content = line.strip()
    #         #print("contentbawah")
    #         #print(content)
    #         #doc_id = "MED" + str(random.randint(0, 9999))
    #
    #
    #         try:
    #             #if docs[doc_id][0]:
    #             if docs[iterat-1][0]:
    #                 pass
    #
    #         except:
    #             #print("docs now")
    #             #print(docs[iterat-1])
    #             nilai_d = ("D" + str(iterat))
    #             #docs[nilai_d] = content #.split()
    #             docs[iterat-1].append(nilai_d)
    #             docs[iterat-1].append(content)   #.split()
    #         #docs[doc_id][0] = ("D" + str(iterat))
    #         #docs[doc_id][1] = content.split()
    #         iterat = iterat + 1

    docs = []
    iterat = 1
    content=""
    for fileloc in filelocs:
        with open("./collection/"+fileloc, 'r') as file:
            content = ""
            docs.append([])
            for line in file:
                line = line.strip()
                content = content + " " + line
            #content = file #.strip()
            try:
                if docs[iterat-1][0]:
                    pass

            except:
                #nilai_d = ("D" + str(iterat))
                #docs[nilai_d] = content #.split()
                docs[iterat-1].append(fileloc) # + " " + nilai_d)
                docs[iterat-1].append(content)   #.split()
                #print("current docs")
                #print(docs[iterat-1])
            #docs[doc_id][0] = ("D" + str(iterat))
            #docs[doc_id][1] = content.split()
            content=""
            iterat = iterat + 1


    #docs = documents
    query = query_in #"how much cancer risk can be avoided through lifestyle change ?"
    #print("documints - AA")
    #print(docs)

    # docs =[("D1", "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"),
    #        ("D2", "study hard as your blood boils"),
    #        ("D3", "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"),
    #        ("D4", "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"),
    #        ("D5", "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")]

    # sekedar pembanding, ada bocoran: D3 & D5 relevant, D1 & D4 partially relevant, D2 tidak relevan

    # bentuk ke format numpy array
    X_unseen = []
    #("documents")
    #print(docs)

    for doc_id, doc in docs: #.items():
      X_unseen.append(features(query.split(), doc.split()))

    X_unseen = np.array(X_unseen)

    # hitung scores
    scores = ranker.predict(X_unseen)
    #print(scores)

    # Ranking pada SERP

    # sekedar pembanding, ada bocoran: D3 & D5 relevant, D1 & D4 partially relevant, D2 tidak relevan
    # apakah LambdaMART berhasil merefleksikan hal ini?

    did_scores = [x for x in zip([did for (did, _) in docs], scores)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

    print("query        :", query)
    print("SERP/Ranking :")
    for (did, score) in sorted_did_scores:
      #print(did, score)
      print(f"{did:30} {score:>f}")
