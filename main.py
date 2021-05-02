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

from sklearn.model_selection import train_test_split

# validation modules
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix

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
        for i in range(1, 11):
            print("wcss",i)
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter= 300, n_init= 10, random_state= 0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            # try:
            # except Exception as e:
            #     print("error @ i=",i)

        plt.plot(range(1, len(wcss)+1),wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters K')
        plt.ylabel('Average Within-Cluster distance to Centroid (WCSS)')
        plt.show()

    """
    https://stackabuse.com/k-means-clustering-with-scikit-learn/
    """
    def process(self):
        print("processing kmeans..")
        df = self.getDataframe("sentimenY1-200MB-8k")
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
        npDf = df.values

        # X = df.iloc[: , [1, df.shape[1]-1]].values
        X = df.values

        self.optimizeClusterN(X)

        pass

class SVMNBCProcessor(Processor):
    name = "SVMNBCProcessor"
    results = {
        "sentimen.y2": None
    }

    models = {
        "nbc": None,
        "svm": None,
    }

    def __init__(self, *args, **kwargs):
        super(SVMNBCProcessor, self).__init__(*args, **kwargs)
    
    """
    https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
    """

    def getDF(self):
        df = self.getDataframe("sentimenY1-200MB-8k")
        df = df.drop([
            'no',
            'Unnamed: 0', 
            'Unnamed: 0.1', 
            'Unnamed: 0.1.1', 
            'status_id', 'created_at', 'screen_name', 'text', 'preprocessed', 'classify_data'], axis=1)
        df = df.replace(
            ['positive', 'negative', 'netral'],
            [1, -1, 0])
        return df

    def getDataWithTest(self, df, label):
        # Split dataset into training set and test set
        # 70% training and 30% test
        # dfin = df.reindex(df.columns)
        X_train, X_test, y_train, y_test = train_test_split(df, df[label], test_size=0.3,random_state=109)

        return X_train, X_test, y_train, y_test

    """
    kernel = ['linear', 'poly', 'rbf']
    """
    def doSVM(self, kernel="linear"):
        print("SVM::{}::training ...".format(kernel))
        df = self.getDF()

        print(df.values)

        # features, label = (df.values, df['classified'])
        X_train, X_test, y_train, y_test = self.getDataWithTest(df, 'classified')
        features, label = (X_train, y_train)

        model = svm.SVC(kernel=kernel)
        model = model.fit(features, label)
        self.models['svm'] = model

        return model, X_train, X_test, y_train, y_test
    
    def doNBC(self):
        print("NBC training ...")
        df = self.getDF()
        # iloc[: , [2, df.shape[1]-1]]
        print(df.values)

        # features, label = (df.values, df['classified'])
        X_train, X_test, y_train, y_test = self.getDataWithTest(df, 'classified')
        features, label = (X_train, y_train)

        model = GaussianNB()
        model = model.fit(features, label)
        self.models['nbc'] = model

        return model, X_train, X_test, y_train, y_test
    
    def getConfussionMatrix(self, y_true, y_pred, labels = "a"):
        # , labels=labels
        return confusion_matrix(y_true, y_pred)

    def doTestModel(self, model, X_test, y_test, alias = "nbc"):
        print("Testing::{}".format(alias),"...")
        #Predict the response for test dataset
        y_pred = model.predict(X_test)
        y_true = y_test

        mirecall = metrics.recall_score(y_true, y_pred, average='micro')
        miprecision = metrics.precision_score(y_true, y_pred, average='micro')
        accuracy = metrics.accuracy_score(y_test, y_pred)
        # f1_micro = metrics.f1_score

        marecall = metrics.recall_score(y_true, y_pred, average='macro')
        maprecision = metrics.precision_score(y_true, y_pred, average='macro')
        # maaccuracy = metrics.accuracy_score(y_test, y_pred)

        rpaMic = {
            "recall": mirecall,
            "precision": miprecision,
            "accuracy": accuracy,
            "confMatrix": self.getConfussionMatrix(y_true, y_pred, labels = "a"),
        }
        # Model Accuracy, how often is the classifier correct?
        print("RPA micro:",rpaMic)

        rpaMac = {
            "recall": marecall,
            "precision": maprecision,
            "accuracy": accuracy,
            "confMatrix": self.getConfussionMatrix(y_true, y_pred, labels = "a"),
        }
        print("RPA macro:",rpaMac)

        return y_pred, rpaMic, rpaMac

    def interactivePredict(self):
        print("init interactiv predict..")
        while True:
            banner = """
Use model
>1. NBC
2. SVM
>"""
            inp = input(banner)
            model = self.models['nbc']

            inp = [int(i) for i in inp.split(" ")]

            print("inputed> ",inp)
                # predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild
            predicted = model.predict([inp]) # 0:Overcast, 2:Mild
            print("Predicted Value:", predicted)
            try:
                pass
            except:
                print("err")


    def process(self):
        pass

class RegresiLMProcessor(Processor):
    name = "Regresi"
    results = {
        "sentimen.y2": None
    }

    def __init__(self, *args, **kwargs):
        super(RegresiLMProcessor, self).__init__(*args, **kwargs)

    def process(self):
        pass

baseio = BaseIO()
p = Processor()
preprocessor = PreProcessor()
# preprocessor = preprocessor.initStemmer().initData()
pKmp = KMeansProcessor()
pSvmnbc = SVMNBCProcessor()
pRegresi = RegresiLMProcessor()

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

# pKmp.process()

# model, X_train, X_test, y_train, y_test = pSvmnbc.doNBC()
# y_pred = pSvmnbc.doTestModel(model, X_test, y_test)
# model, X_train, X_test, y_train, y_test = pSvmnbc.doSVM('poly')
# y_pred = pSvmnbc.doTestModel(model, X_test, y_test, "SVM::" + model.kernel)
# pSvmnbc.interactivePredict()

# search
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