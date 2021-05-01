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

"""
speed up training
https://medium.com distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1
https://towardsdatascience.com pca-using-python-scikit-learn-e653f8989e60
https://medium.com distributed-computing-with-ray/how-to-speed-up-pandas-with-modin-84aa6a87bcdb
"""

from baseio import BaseIO

class Processor:
    name = "Base processor"
    params = {}
    results = {}
    baseio = None

    progressModer = 1000

    def __init__(self, name = "", params = {}):
        # self.name = name
        self.params = params
        self.baseio = BaseIO()
    
    def resultToDict(self):
        for r in self.results.keys():
            self.results[r] = {}

    def addParam(self, key, data):
        self.params[key] = data

        return self

    def setParam(self):
        return self

    def process(self):
        pass
    
    def logs(self, text = ""):
        print(self.name + "::" + text)
        pass
    
    def percentProgress(self, now, total):
        return (now / total) * 100
    
    """
    sample

    self.progressor({
        'type': "start",
        'total': total,
        'now': now,
    })

    self.progressor({
        'type': "progress",
        'total': total,
        'now': now,
    })
    """
    def progressor(self, arg):
        typ = arg['type']
        if typ == "start":
            starttime = time.time()
            print("total: ",arg['total'])
            print("start: ",starttime)

            return starttime
        elif typ == "progress":
            now = arg['now']
            total = arg['total']
            if now % self.progressModer == 0:
                print("progress", self.percentProgress(now, total),"% @ {}".format(str(now)+"/"+str(total)))

    def getGlobalWords(self, typef = "df"):
        if typef == "file":
            r = open("dummy/globalWords",'r').read().split("\n")
        elif typef == "df":
            r = pd.read_csv("dummy/term.freq.9k.csv", header=0, lineterminator='\n')
        elif typef == "df-docfreq":
            r = pd.read_csv("dummy/term.docfreq.9k.csv", header=0, lineterminator='\n')
        elif typef == "df8k":
            r = pd.read_csv("dummy/term.freq.8k.csv", header=0, lineterminator='\n')
        elif typef == "df-docfreq8k":
            r = pd.read_csv("dummy/w8k.term.docfreq.csv", header=0, lineterminator='\n')

        return r
    
    def getDataframe(self, typef = "df"):
        r = None
        if typef == "file":
            r = None
        elif typef == "sentimenY1-200MB":
            r = pd.read_csv("dummy/sentimenY1result.csv", header=0, lineterminator='\n')
        elif typef == "sentimenY1-200MB-8k":
            r = pd.read_csv("dummy/w8ksentimenY1result.csv", header=0, lineterminator='\n')
        elif typef == "sentimenY1-200MB-dfval":
            pass
            # r = pd.read_csv('dummy/sentimenY1result.csv', sep=',',header=0)
            # r = df.values
            # r = pd.read_csv("dummy/sentimenY1result.csv", header=0, lineterminator='\n')
        elif typef == "sentimenY1-sample":
            r = pd.read_csv("dummy/sentimenY1result.10.csv", header=0, lineterminator='\n')
        elif typef == "df-docfreq":
            r = pd.read_csv("dummy/term.docfreq.9k.csv", header=0, lineterminator='\n')

        return r
