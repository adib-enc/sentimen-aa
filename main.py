import re
# import pandas as pd # normal pandas

#modins
import modin.pandas as pd 
import ray
ray.init()

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

        # self.optimizeClusterN(X)
        # return
        i = 3
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter= 300, n_init= 10, random_state= 0)
        kmeans.fit(X)
        # todo : joblib
        df['kmean_label'] = kmeans.labels_
        df['preprocessed'] = dfprep
        # df.to_csv("dummy/sentimenY1-200MB-8k-kmeansd.csv")
        df.to_csv("dummy/sentimenY1-200MB-8k-kmeansd-2.csv")

        print(self.now())

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

    trainTestPairs = ()
    dfProccess = None

    C = 1
    CRange = [1e-2, 1, 1e2]
    gamma = 1
    gammaRange = [1e-1, 1, 1e1]

    tuned_parameters = [
        {
            'kernel':['linear'], 
            'C':[0.01, 0.1, 1, 10, 100, 1000, 10000]
        }, 
        {
            'kernel':['rbf'],
            'C':[0.01, 0.1, 1, 10, 100, 1000, 10000],
            'gamma':[1000, 100, 10, 1, 0.1, 0.01, 0.001]
        }
    ]


    def __init__(self, *args, **kwargs):
        super(SVMNBCProcessor, self).__init__(*args, **kwargs)
    
    """
    https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
    """

    def getDF(self):
        print("get 200 MB csv")
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
    
    """
    return dffeature
    """
    def getFeatureAndLabel(self):
        df = self.getDF()
        print(df)
        dfr = df
        feature = dfr.drop("classified", axis=1)
        label = df["classified"]

        return feature, label

    def getDataWithTest(self, df, label, test_size=0.1):
        # Split dataset into training set and test set
        # 70% training and 30% test
        # dfin = df.reindex(df.columns)
        X_train, X_test, y_train, y_test = train_test_split(df, df[label], test_size=test_size,random_state=109)

        return X_train, X_test, y_train, y_test

    """
    train:test
    90:10

    https://scikit-learn org/stable/modules/generated/sklearn.model_selection.KFold.html
    https://scikit-learn org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

    X_train, X_test, y_train, y_test = self.doKFold(df.values, df['classified'], nsplits=5)
    """
    def doKFold(self, feature, label, nsplits=5):
        print("10fold")
        print("StratifiedShuffleSplit")
        
        X, y = (feature, label)
        
        X_train, X_test, y_train, y_test = (None,None,None,None,)

        # kf = KFold(n_splits=nsplits)
        kf = StratifiedShuffleSplit(n_splits=nsplits, test_size=0.1)
        kf.get_n_splits(X)

        i = 1
        print("init::",self.now())
        # results = []
        for train_index, test_index in kf.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("i",i)
            i+=1
            # too big for memory
            # results.append({
            #     "nsplit": i,
            #     "data": (X_train, X_test, y_train, y_test)
            # })
            print("="*32)
            # print("nsplit",d['nsplit'])

            # === training
            pSvmnbc.trainTestPairs = (X_train, X_test, y_train, y_test)
            model, X_train, X_test, y_train, y_test = pSvmnbc.doSVM("rbf")
            y_pred, rpaMic, rpaMac = pSvmnbc.doTestModel(model, X_test, y_test, "SVM::" + model.kernel)
            print(self.now())
            self.toFileWithTimestamp("nsplit" + str(i), str((y_pred, rpaMic, rpaMac)) )
        
        # return results

    def kfoldAndTrain(self):
        print("kfoldAndTrain")
        pSvmnbc = self
        dffeature, label = pSvmnbc.getFeatureAndLabel()
        pSvmnbc.C = 1000
        pSvmnbc.gamma = 1e-3
        pSvmnbc.doKFold(dffeature.values, label, 10)

        print(self.now())

    def procKFold(self, df, label):
        print("proc do kfold")
        features, label = (df, label)
        X_train, X_test, y_train, y_test = self.doKFold(features, label)

        return X_train, X_test, y_train, y_test
    
    """
    kernel = ['linear', 'poly', 'rbf']

    # print(df.values)

    # features, label = (df.values, df['classified'])
    # ,,,, = self.getDataWithTest(df, 'classified')

    # use kfold crossval
    # X_train, X_test, y_train, y_test = self.doKFold(df.values, df['classified'], nsplits=5)

    current best:
    {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
    """
    def doSVM(self, kernel="linear"):
        print("SVM::{}::training ...".format(kernel))
        print("C:",self.C,"gamma:",self.gamma)
        # df = self.getDF()
        # df = self.dfProccess
        # print(df.values)

        # features, label = (df.values, df['classified'])
        # X_train, X_test, y_train, y_test = self.getDataWithTest(df, 'classified')
        X_train, X_test, y_train, y_test = self.trainTestPairs
        features, label = (X_train, y_train)

        model = None
        if kernel == "linear":
            model = svm.SVC(kernel=kernel, C=self.C)
        elif kernel == "rbf":
            model = svm.SVC(kernel=kernel, C=self.C, gamma=self.gamma)

        model = model.fit(features, label)
        self.models['svm'] = model

        return model, X_train, X_test, y_train, y_test
    
    def doSVMRBF(self):
        return self.doSVM("rbf")

    def doFullSVM(self, df):
        print("do full svm")
        # df = self.getDF()
        self.dfProccess = df
        self.trainTestPairs = self.doKFold(df.values, df['classified'], nsplits=5)

        # linear kernel
        svmResults = []
        svmCs = [10e-2, 10e-1, 1, 10e1, 10e2, 10e3, 10e4]
        print("svmCs",svmCs)
        self.CRange = svmCs
        self.gammaRange = svmCs

        for C in svmCs:
            for gamma in svmCs:
                self.C = C
                model, X_train, X_test, y_train, y_test = self.doSVM()
                # rbf, convent : C use same C, gamma use same C
                # pSvmnbc.gammaRange = pSvmnbc.CRange
                pSvmnbc.gamma = gamma
                model, X_train, X_test, y_train, y_test = pSvmnbc.doSVMRBF()

                svmResults.append("todo: rpas")
    
    """
    thx to PUTRI AYU @its

    w/ n_jobs=4
    """
    def doSVMwithGridSearch(self, df = None, n_jobs=4):
        print("pa :: doSVMwithGridSearch")
        # self.trainTestPairs = self.getDataWithTest(df, 'classified')
        # X_train, X_test, y_train, y_test = self.trainTestPairs
        
        dflabel = df['classified']
        df = df.drop('classified', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df, dflabel, test_size=0.1,random_state=109)

        # df = df.head(10)
        print(self.now())
        print("df")
        print(df)

        scores = ['precision', 'recall']

        for score in scores:
            print ("# Turning hyper-parameters for %s" % score)
            print ("w/ n_jobs=4")
            print ()
            # clf = GridSearchCV(SVC(), self.tuned_parameters, cv=KFold(n_splits=10, random_state=0))
            clf = GridSearchCV(SVC(), self.tuned_parameters, cv=KFold(n_splits=10), n_jobs=n_jobs)
            clf.fit(X_train, y_train)
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on training set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            paramsOut = []
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                p = "%0.3f (+/-%0.3f) for %r" % (mean, std * 2, params)
                print(p)
                paramsOut.append(p)

            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            
            cm = confusion_matrix(y_true, y_pred)
            cr = classification_report(y_true, y_pred)

            print(cm)
            print(cr)

            fname = score + ".txt"
            cont = {
                "bestparam" : clf.best_params_,
                "means" : means,
                "stds" : stds,
                "params" : paramsOut,
                "confmatrix" : cm,
                "classfreport" : cr,
            }
            
            # todo
            print("Serializing gridsearch clf..")
            
            self.toFileWithTimestamp(fname, str(cont))
            print()
            print(self.now())

        print(self.now())
    
    def doNBC(self):
        print("NBC training ...")

        # features, label = (df.values, df['classified'])
        # X_train, X_test, y_train, y_test = self.getDataWithTest(df, 'classified')
        X_train, X_test, y_train, y_test = self.trainTestPairs
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

    def mergeWithKeyword(self):
        pass
        #keyword
    
    def inserColRetweetNFavorit(self):
        print("inserColRetweetNFavorit")
        cols = [
            ['retweet_count'], 
            ['favorite_count']
        ]
        dfL = pd.read_csv("dummy/_dummylinear.csv")
        dfL2 = pd.read_csv("dummy/_dummylinear2.csv")

        for c in cols:
            dfL[c] = dfL2[c]
        
        dfL.to_csv("dummy/_dummylinear.csv")
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

# pKmp.process()


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
# 
# preprocessor.tfPerTweet(df, dfTerms, "w8k")
# df = p.getDataframe("tf-pertweet")
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