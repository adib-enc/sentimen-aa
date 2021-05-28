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

import joblib

from baseio import BaseIO
from processor import Processor
from preprocessor import PreProcessor

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

    def doKFoldNBC(self, feature, label, nsplits=5):
        print("nbc 10fold ")
        print("StratifiedShuffleSplit")
        
        X, y = (feature, label)
        
        X_train, X_test, y_train, y_test = (None,None,None,None,)

        # kf = KFold(n_splits=nsplits)
        pSvmnbc = self
        kf = StratifiedShuffleSplit(n_splits=nsplits, test_size=0.1)
        kf.get_n_splits(X)

        i = 1
        print("init::",self.now())
        # results = []
        eq = "="*16
        for train_index, test_index in kf.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            print(eq,"split",i,eq)
            print("TRAIN, TEST")
            print(len(train_index), len(test_index))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("="*32)
            print("i",i)

            # === training
            pSvmnbc.trainTestPairs = (X_train, X_test, y_train, y_test)
            model, X_train, X_test, y_train, y_test = pSvmnbc.doNBC()
            #  + model.kernel
            y_pred, rpaMic, rpaMac = pSvmnbc.doTestModel(model, X_test, y_test, "NBC::")
            print(self.now())
            
            dic = {
                "train": len(train_index),
                "test": len(test_index),
            }
            out = str((y_pred, rpaMic, rpaMac, dic))
            self.toFileWithTimestamp("nbc.nsplit" + str(i), out)
            i+=1
        
        # return results

    """
    train:test
    90:10

    https://scikit-learn org/stable/modules/generated/sklearn.model_selection.KFold.html
    https://scikit-learn org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

    X_train, X_test, y_train, y_test = self.doKFold(df.values, df['classified'], nsplits=5)
    """
    def doKFold(self, feature, label, nsplits=5, kernel="rbf"):
        print("10fold",kernel)
        print("StratifiedShuffleSplit")
        
        X, y = (feature, label)
        
        X_train, X_test, y_train, y_test = (None,None,None,None,)

        # kf = KFold(n_splits=nsplits)
        kf = StratifiedShuffleSplit(n_splits=nsplits, test_size=0.1)
        kf.get_n_splits(X)

        i = 1
        print("init::",self.now())
        # results = []
        eq = "="*16
        for train_index, test_index in kf.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            print(eq,"split",i,eq)
            print("TRAIN, TEST")
            print(len(train_index), len(test_index))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("="*32)
            print("i",i)
            
            # too big for memory
            # results.append({
            #     "nsplit": i,
            #     "data": (X_train, X_test, y_train, y_test)
            # })
            # print("nsplit",d['nsplit'])

            # === training
            pSvmnbc.trainTestPairs = (X_train, X_test, y_train, y_test)
            model, X_train, X_test, y_train, y_test = pSvmnbc.doSVM(kernel)
            y_pred, rpaMic, rpaMac = pSvmnbc.doTestModel(model, X_test, y_test, "SVM::" + model.kernel)
            print(self.now())
            
            dic = {
                "train": len(train_index),
                "test": len(test_index),
            }
            out = str((y_pred, rpaMic, rpaMac, dic))
            self.toFileWithTimestamp(kernel + "nsplit" + str(i), out)
            i+=1
        
        # return results


    def kfoldAndTrain(self):
        print("kfoldAndTrain")
        pSvmnbc = self
        dffeature, label = pSvmnbc.getFeatureAndLabel()
        pSvmnbc.C = 1000
        pSvmnbc.gamma = 1e-3
        pSvmnbc.doKFold(dffeature.values, label, 10)

        print(self.now())

    def kfoldAndTrainLinear(self):
        print("kfoldAndTrain linear")
        pSvmnbc = self
        dffeature, label = pSvmnbc.getFeatureAndLabel()
        pSvmnbc.C = 0.01
        pSvmnbc.doKFold(dffeature.values, label, 10, kernel="linear")

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

        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        rpaMac = {
            "recall": marecall,
            "precision": maprecision,
            "accuracy": accuracy,
            "confMatrix": self.getConfussionMatrix(y_true, y_pred, labels = "a"),
            "ravel": confusion_matrix(y_true, y_pred).ravel(),
            "fmt": "(tn, fp, fn, tp)"
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
