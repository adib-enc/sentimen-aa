import re
import pandas as pd # normal pandas

#modins
# import modin.pandas as pd 
# import ray
# ray.init()

import numpy as np
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
from pprint import pprint
import time
"""
# create stemmer
from pprint import pprint
factory = StemmerFactory()
stemmer = factory.create_stemmer()
"""

import matplotlib.pyplot as plt

# sklearns
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import LinearSVC, SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

# validation modules
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

"""
speed up training
https://medium.com distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1
https://towardsdatascience.com pca-using-python-scikit-learn-e653f8989e60
https://medium.com distributed-computing-with-ray/how-to-speed-up-pandas-with-modin-84aa6a87bcdb
"""

# save model
"""
# https://machinelearningmastery com/save-load-machine-learning-models-python-scikit-learn/
"""
import joblib

from baseio import BaseIO
from processor import Processor
from preprocessor import PreProcessor
from svmnbcprocessor import SVMNBCProcessor

class KMeansProcessor(Processor):
    name = "KMeansProcessor"
    results = {
        "cluster.x10": None,
    }

    def __init__(self, *args, **kwargs):
        super(KMeansProcessor, self).__init__(*args, **kwargs)

    """
    # optimal number of clusters
    X = np compatible dataset

    """

    def optimizeClusterN(self, X):
        print("kmeans::optimizing..")
        wcss = []
        distortions = []
        from scipy.spatial.distance import cdist

        rn = range(1, 26)

        for i in rn:
            print("wcss",i)
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter= 300, n_init= 10, random_state= 0)
            kmeans.fit(X)
            # distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
            wcss.append(kmeans.inertia_)
            # try:
            # except Exception as e:
            #     print("error @ i=",i)

        print(self.now())
        open("wcss"+self.now(),"w").write(str(wcss))
        # open("distortions"+self.now(),"w").write(str(distortions))
        plt.plot(range(1, len(wcss)+1),wcss)
        # plt.plot(range(1, len(distortions)+1),distortions)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters K')
        plt.ylabel('Average Within-Cluster distance to Centroid (WCSS) distort')
        plt.show()

    """
    https://stackabuse.com/k-means-clustering-with-scikit-learn/
    """
    def process(self):
        print("processing kmeans..")
        print(self.now())
        df = self.getDataframe("sentimenY1-200MB-8k")
        dfprep = df['preprocessed']
        df = df.drop([
            'no',
            'Unnamed: 0', 
            'Unnamed: 0.1', 
            'Unnamed: 0.1.1', 
            'classified',
            'status_id', 'created_at', 'screen_name', 'text', 'preprocessed', 'classify_data'], axis=1)
        # df = df.replace(
        #     ['positive', 'negative', 'netral'],
        #     [1, -1, 0])
        
        # npDf = df.to_numpy()
        # npDf = df.values
        df = df.replace(
            ['positive', 'negative', 'netral'],
            [1, -1, 0])

        # X = df.iloc[: , [1, df.shape[1]-1]].values
        X = df.values
        print(X)
        print(X.size)

        # self.optimizeClusterN(X)
        
        # return
        i = 5
        print("i",i)
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter= 300, n_init= 10, random_state= 0)
        kmeans.fit(X)
        # todo : joblib
        df['kmean_label'] = kmeans.labels_
        df['preprocessed'] = dfprep
        # df.to_csv("dummy/sentimenY1-200MB-8k-kmeansd.csv")
        # df.to_csv("dummy/sentimenY1-200MB-8k-kmeansd-2.csv")
        df.to_csv("dummy/sentimenY1-200MB-8k-kmeansd-5c.csv")

        fname = "kmeans.model."+self.now()
        joblib.dump(kmeans, fname)
        print(fname)
        print(self.now())

        pass

class RegresiLMProcessor(Processor):
    name = "Regresi"
    results = {
        "sentimen.y2": None
    }

    def __init__(self, *args, **kwargs):
        super(RegresiLMProcessor, self).__init__(*args, **kwargs)

    """
    jml retweet
    hashtag
    keyword
    jml follower
    jml following
    jml likes
    jml kata @tweet
    status verifikasi
    cluster
    source / perangkat

    need proc:
    hashtag .
    keyword .
    jml kata @tweet 
    status verifikasi
    cluster
    source / perangkat

    https://www.delftstack com/howto/python-pandas/pandas-replace-values-in-column/
    """
    def buildVariabelPrediktor(self):
        print("buildVariabelPrediktor")
        dfs = self.getDataframe("df-partial")
        df = self.concatDataframeRows(dfs)
        # ffname = "dummy/_dummylinear.csv"
        fname = "dummy/_dummylinear2.csv"
        print(df['favorite_count'])
        print(df['retweet_count'])
        df.to_csv(fname)
        print("built")
        return df

    def mergeVPwithKmean(self, dfL = None):
        print("merging...")
        df = self.getDataframe("sentimenY1-200MB-8k-kmean")
        
        if dfL is None:
            dfL = pd.read_csv("dummy/_dummylinear.csv")
            
        dfL['kmean_label'] = df['kmean_label']
        dfL.to_csv("dummy/_dummylinear.csv")
        print(dfL['word_count'])

        return dfL
    
    def mergeDummyWithSvmnbc(self):
        print("merging...")
        df = pd.read_csv("dummy/svmnbc.classifications.csv")
        
        dfL = pd.read_csv("dummy/_dummylinear.csv")
        dfL['svm_class'] = df['svm_class']
        dfL['nbc_class'] = df['nbc_class']

        dfL.to_csv("dummy/_dummylinear.csv")
        return dfL

    def merge10colsWithPreprocessed(self):
        print("merge10colsWithPreprocessed")
        # dfL = self.getDataframe(typef = "rl10+cols")
        # dfL2 = self.getDataframe(typef = "classified-clean")
        
        # dfL['preprocessed'] = dfL2['preprocessed']
        fname = "dummy/regresi.linear.10+cols.preped.csv"
        # dfL.to_csv(fname)
        # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # df = pd.read_csv(fname)
        # df.to_csv(fname)

        pass
        #keyword
    
    def inserColRetweetNFavorit(self):
        print("inserColRetweetNFavorit")
        cols = [
            'retweet_count', 
            'favorite_count'
        ]
        dfL = pd.read_csv("dummy/_dummylinear.csv")
        dfL2 = pd.read_csv("dummy/_dummylinear2.csv")

        for c in cols:
            dfL[c] = dfL2[c]
        
        dfL.to_csv("dummy/_dummylinear3.csv")
        #keyword

    def process(self):
        pass

class Wordcloud(Processor):
    preprocessor = None
    def __init__(self, *args, **kwargs):
        super(Wordcloud, self).__init__(*args, **kwargs)
        self.preprocessor = PreProcessor()
    
    def formCloudFromKmeans(self):
        print("formCloudFromKmeans")
        df = self.getDataframe(typef = "sentimenY1-200MB-8k-kmean-2")
        uniqs = df['kmean_label'].unique()
        print(uniqs)
        for u in uniqs:
            dfdata = df.loc[df.kmean_label == u]
            fname = "cluster." + str(u) + ".wcloud"
            with open(fname, "w") as wcloudfile:
                print("writing",fname)
                for preprocessed in dfdata['preprocessed']:
                    preprocessed = str(preprocessed)
                    write = preprocessed.replace(" ", "\n")
                    wcloudfile.write(write)
                wcloudfile.close()
                print("to file",fname)

        pass

    def form3Class(self):
        print("form3Class")
        df = self.getDataframe(typef = "classified-clean")
        preprocessor = self.preprocessor
        classes = ['negative', 'positive', 'netral']

        for c in classes:
            dfdata = df.loc[df.classified == c]
            print(c)
            print(dfdata)
            fname = "wcloud."+c
            with open(fname, "w") as wcloudfile:
                print("writing",fname)
                for preprocessed in dfdata['preprocessed']:
                    write = preprocessed.replace(" ", "\n")
                    wcloudfile.write(write)
                wcloudfile.close()
                print("to file",fname)

        return
    
    def formWordcloudFromSVMPrediction(self):
        print("formWordcloudFromSVMPrediction")
        df = self.getDataframe(typef = "rl10+cols")
        df = pd.read_csv('dummy/regresi.linear.10+cols.preped.csv')
        preprocessor = self.preprocessor
        classes = [1, 0, -1]

        for c in classes:
            dfdata = df.loc[df.svm_class == c]
            print(c)
            print(dfdata)
            fname = "wcloud.svm." + str(c)
            with open(fname, "w") as wcloudfile:
                print("writing",fname)
                for preprocessed in dfdata['preprocessed']:
                    # write = 
                    if isinstance(preprocessed,str):
                        write = preprocessed.replace(" ", "\n")
                        wcloudfile.write(write)
                wcloudfile.close()
                print("to file",fname)

        return

baseio = BaseIO()
p = Processor()
preprocessor = PreProcessor()
# preprocessor = preprocessor.initStemmer().initData()
pKmp = KMeansProcessor()
pSvmnbc = SVMNBCProcessor()
pRegresi = RegresiLMProcessor()
pWcloud = Wordcloud()

def initialize():
    crawled = [
        # baseio.inputFmt("ruu.minol", 'ruu.minol.csv'),
        # baseio.inputFmt("ruu.minol2", 'ruu.minol2.csv'),
        # baseio.inputFmt("ruu.minuman.beralkohol", 'ruu.minuman.beralkohol.csv'),
        # baseio.inputFmt("ruu.minuman.beralkohol2", 'ruu.minuman.beralkohol2.csv'),
        # baseio.inputFmt("ruu.miras", 'ruu.miras.csv'),
        baseio.inputFmt("ruu.miras2", 'ruu.miras2.csv'),
    ]

    preprocessor.resultToDict()

    # print(crawled)
    for c in crawled:
        # get classified
        # filen = './dummy/classified/' + c['name'] + '.classified.csv'
        # print(filen)
        # c['dataframe'] = pd.read_csv(filen, header=0, lineterminator='\n')
        # preprocessor.results['sentimen.y1'][c['name']] = c

        # get preprocessed
        # filen = './dummy/preprocess/' + c['name'] + '.preprocessed.csv'
        # c['dataframe'] = pd.read_csv(filen, header=0, lineterminator='\n')
        # preprocessor.results['preprocess'][c['name']] = c

        # print(pPre.results['crawling'])
        # get crawled
        c['dataframe'] = pd.read_csv('./dummy/'+c['filename'], header=0)
        preprocessor.results['crawling'][c['name']] = c
        # break


# initialize()
# preprocessor.preproccess()
# df = pd.read_excel('dummypreproc.3.text.xls', header=0)
# fn = "dummy/ruu.miras2.csv"
# fn = "dummy/classified/ruu.all.classified.csv"
# fn = "dummy/preprocess/reclean.preprocessed.csv"
fn = "dummy/classified/reclean.classified.csv"
# df = pd.read_csv(fn, header=0)
# dummy/classified/ruu.all.classified.clean.csv
# dfpreproc = preprocessor.preproccess(df, "text", "reclean")

# fn = "dummy/preprocess/dummyPreproc.preprocessed.csv"
# df = pd.read_csv(fn, header=0)
# preprocessor.classify(df, "preprocessed", "reclean")
# preprocessor.formGlobalWords(df, "reclean", toFile = True)
# fn = "dummy/classified/reclean.classified.csv"
# dfTerms = preprocessor.getGlobalWords(typef = "df8k")
# preprocessor.documentFrequency(df, dfTerms, "w8k")

# dfTerms = preprocessor.getGlobalWords(typef = "df8k")
# preprocessor.documentFrequency(df, dfTerms, "w8k")

# dfTerms = preprocessor.getGlobalWords(typef = "df-docfreq8k")
# preprocessor.tfidfWeighting(df, dfTerms, "w8k")
# preprocessor.classify()
# preprocessor.formGlobalWords()
# preprocessor.documentFrequency()
# preprocessor.tfidfWeighting()
# pprint(preprocessor.termFrequency(["a","b","c","c","c"]))
# pprint(preprocessor.termFrequency("aabbbcdefggg"))
# pprint(preprocessor.termFrequency(preprocessor.getGlobalWords()))

### pKmp.process()
pKmp.process()


###### svm

# loading
# df = pSvmnbc.getDF()
# dffeature, label = pSvmnbc.getFeatureAndLabel()
# pSvmnbc.doFullSVM(df)
# pSvmnbc.trainTestPairs = self.getDataWithTest(df, 'classified')
# doKFold(df.values, df['classified'])
# pSvmnbc.doSVMwithGridSearch(df, 3)
# pSvmnbc.C = 1000
# pSvmnbc.gamma = 1e-3
# kfoldResults = pSvmnbc.doKFold(dffeature.values, label, 10)

# for d in kfoldResults:
#     print("nsplit",d['nsplit'])
#     pSvmnbc.trainTestPairs = d['data']
# model, X_train, X_test, y_train, y_test = pSvmnbc.doSVM("rbf")

# df = pd.DataFrame()
# df["svc_classes"] = model.classes_
# df.to_csv("svc_classes.csv")
# print("to svc_classes.csv")
###### svm

# y_pred = pSvmnbc.doTestModel(model, X_test, y_test, "SVM::" + model.kernel)
"""
RPA micro: {'recall': 0.8314238952536824, 'precision': 0.8314238952536824, 'accuracy': 0.8314238952536824, 'confMatrix': array([[330,   5,  33],
       [ 10,  47,  12],
       [ 30,  13, 131]])}
RPA macro: {'recall': 0.7769240379810095, 'precision': 0.7864289989289989, 'accuracy': 0.8314238952536824, 'confMatrix': array([[330,   5,  33],
       [ 10,  47,  12],
       [ 30,  13, 131]])}
"""
# pSvmnbc.interactivePredict()

# pSvmnbc = SVMNBCProcessor()

# use previous pSvmnbc.trainTestPairs
# modelnbc, nbcX_train, nbcX_test, nbcy_train, nbcy_test = pSvmnbc.doNBC()
# nbcy_pred = pSvmnbc.doTestModel(modelnbc, nbcX_test, nbcy_test)
"""
RPA micro: {'recall': 0.5270049099836334, 'precision': 0.5270049099836334, 'accuracy': 0.5270049099836334, 'confMatrix': array([[170,  52, 146],
       [ 14,  42,  13],
       [ 47,  17, 110]])}
RPA macro: {'recall': 0.5676120273196735, 'precision': 0.507743682464872, 'accuracy': 0.5270049099836334, 'confMatrix': array([[170,  52, 146],
       [ 14,  42,  13],
       [ 47,  17, 110]])}
"""

# pSvmnbc.doKFold()
# pSvmnbc.kfoldAndTrain()
# pSvmnbc.kfoldAndTrainLinear()

# pRegresi.mergeDummyWithSvmnbc()

"""
wordcloud
1. clust 1
2. clust 2
3. clust 3
4. pos
5. neg
6. net
"""

# pWcloud.form3Class()
# pWcloud.formCloudFromKmeans()

# df = pSvmnbc.getDF()
# X_train, X_test, y_train, y_test = pSvmnbc.procKFold(df.values, df['classified'])
# print((X_train, X_test, y_train, y_test))

# print(type(pRegresi.getWordCnt))
# df = pRegresi.buildVariabelPrediktor()
# pRegresi.mergeVPwithKmean(df)


# df = pd.read_csv("dummy/_dummylinear.csv")
# df.source.replace()
# print(df['kmean_label'].unique())
# print(df['hashtags_vp'].unique())
# print(df['source'].unique())
# search

# ================================ 2021-05-22 ================================
# pRegresi.inserColRetweetNFavorit()
# pRegresi.merge10colsWithPreprocessed()
# pWcloud.formWordcloudFromSVMPrediction()
# 
# preprocessor.tfPerTweet(df, dfTerms, "w8k")
# df = p.getDataframe("tf-pertweet")

# pWcloud.formWordcloudFromSVMPrediction()
# 
"""
df.loc[df['status_id'] == 'x1328208693461196803']
df.loc[df['user_id'] == 'x765149955396734976']['text']
"""
# t1 = 'Pengusul RUU itu kata warga seakan berniat menghapus budaya NTT. Sementara anggota DPRD dari Bali menilai RUU minol bertentangan dengan kebhinekaan Indonesia. https://t.co/97tsrJyj5k https://t.co/natBlBTqAh'
# tz = preprocessor.tokenize(t1, toWords = True)
# tz = preprocessor.tokenize(t1)
# print(tz)

"""
monkey patcher

from processor import *; 
from processor import *; preprocessor.preproccess();
from processor import *; preprocessor.classify();
from processor import *; preprocessor.formGlobalWords();
df = preprocessor.results['crawling']['ruu.minol']['dataframe']; dfp = preprocessor.results['preprocess']['ruu.minol']['dataframe']
"""