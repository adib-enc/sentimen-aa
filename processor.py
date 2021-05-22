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

from datetime import datetime

import matplotlib.pyplot as plt

# sklearns
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

import joblib

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

    def now(self, fmt = None):
        if fmt is None:
            fmt = '%Y-%m-%d-%H:%M:%S'

        return datetime.today().strftime(fmt)

    def toFile(self, fname, cont):
        print("out to::",fname)
        return open(fname,'w').write(cont)

    def toFileWithTimestamp(self, fname, cont):
        fname = self.now() + "." + fname
        return self.toFile(fname, cont)
    
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
        elif typ == "progress" or typ == "report":
            now = arg['now']
            total = arg['total']
            if now % self.progressModer == 0:
                print("progress", self.percentProgress(now, total),"% @ {}".format(str(now)+"/"+str(total)))

    def concatDataframeRows(self, dataframes):
        return pd.concat(dataframes, axis=0, join='outer')

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
    
    def filterDataframe(self, df, typef = "df"):
        retdf = None
        if typef == "liner-regress":
            df.loc[df.hashtag == '','hashtag_vp'] = 0
            df.loc[df.hashtag != '','hashtag_vp'] = 1
            retdf = df

        return retdf

    def getWordCnt(self, sentc):
        cnt = 0
        
        if isinstance(sentc,str):
            sentc = sentc.split(" ")
        
        cnt = len(sentc)

        return cnt

    def getDataframe(self, typef = "df"):
        r = None
        if typef == "file":
            r = None
        elif typef == "tf-pertweet":
            r = pd.read_csv("dummy/w8ktf.pertweet.csv", header=0, lineterminator='\n')
        elif typef == "classified-clean":
            r = pd.read_csv("dummy/classified/reclean.classified.csv", header=0, lineterminator='\n')
        elif typef == "classified-filtered":
            r = pd.read_csv("dummy/sentimenY1result.csv", header=0, lineterminator='\n')
        elif typef == "sentimenY1-200MB":
            r = pd.read_csv("dummy/sentimenY1result.csv", header=0, lineterminator='\n')
        elif typef == "sentimenY1-200MB-8k-kmean":
            useCols = ['kmean_label']
            r = pd.read_csv("dummy/sentimenY1-200MB-8k-kmeansd.csv", header=0, lineterminator='\n',usecols=useCols)
        elif typef == "sentimenY1-200MB-8k-kmean-2":
            useCols = ['kmean_label', "preprocessed"]
            r = pd.read_csv("dummy/sentimenY1-200MB-8k-kmeansd-2.csv", header=0, lineterminator='\n',usecols=useCols)
        elif typef == "sentimenY1-200MB-8k":
            r = pd.read_csv("dummy/w8ksentimenY1result.csv", header=0, lineterminator='\n')
        elif typef == "sentimenY1-200MB-dfval":
            pass
            # r = pd.read_csv('dummy/sentimenY1result.csv', sep=',',header=0)
            # r = df.values
            # r = pd.read_csv("dummy/sentimenY1result.csv", header=0, lineterminator='\n')
        elif typef == "sentimenY1-sample":
            r = pd.read_csv("dummy/sentimenY1result.10.csv", header=0, lineterminator='\n')
        elif typef == "sentimenY1-sample-30":
            r = pd.read_csv("dummy/w8kY1.30.csv", header=0, lineterminator='\n')
        elif typef == "df-docfreq":
            r = pd.read_csv("dummy/term.docfreq.9k.csv", header=0, lineterminator='\n')
        elif typef == "df-partial":
            baseio = self.baseio
            crawled = [
                baseio.inputFmt("ruu.minol", 'ruu.minol.csv'),
                baseio.inputFmt("ruu.minol2", 'ruu.minol2.csv'),
                baseio.inputFmt("ruu.minuman.beralkohol", 'ruu.minuman.beralkohol.csv'),
                baseio.inputFmt("ruu.minuman.beralkohol2", 'ruu.minuman.beralkohol2.csv'),
                baseio.inputFmt("ruu.miras", 'ruu.miras.csv'),
                baseio.inputFmt("ruu.miras2", 'ruu.miras2.csv'),
            ]
            catReplace = {
                "ruu.minol": 0,
                "ruu.minol2": 0,
                "ruu.minuman.beralkohol": 2,
                "ruu.minuman.beralkohol2": 2,
                "ruu.miras": 1,
                "ruu.miras2": 1,
            }
            useCols = [
                'text',
                'hashtags',
                'followers_count',
                'friends_count',
                'favourites_count',
                'favorite_count',
                'retweet_count', 
                'verified',
                'source', #src device
            ]
            
            dfs = []
            for c in crawled:
                print("ok",c)
                df = pd.read_csv('./dummy/'+c['filename'], header=0, usecols=useCols)
                # df['hashtags_vp'] = 0
                print(df.head())
                
                print("====")
                df.hashtags = df.hashtags.replace(regex={r'.*': 1, '': 0})
                df.hashtags = df.hashtags.replace({'nan':0})
                
                # replace source
                devReplace = {
                    'Twitter for iPhone': 0, 
                    'Twitter for Android': 1, 
                    'Twitter Web App': 2, 
                }
                verifiedReplace = {
                    True: 1,
                    False: 0,
                }
                srcUniq = list(df.source.unique())
                
                try:
                    srcUniq.remove('Twitter for iPhone')
                    srcUniq.remove('Twitter for Android')
                    srcUniq.remove('Twitter Web App')
                except:
                    pass
                
                for sr in srcUniq:
                    df.source = df.source.replace(sr, 3)
                
                df.source = df.source.replace(devReplace)
                df.verified = df.verified.replace(verifiedReplace)

                df.fillna(value=0, inplace=True)
                # print(df.source.unique())
                # print("hashtg")
                
                df['word_count'] = df['text'].apply(self.getWordCnt)
                df['classified'] = ''
                df['keyword'] = catReplace[c['name']]
                
                df.replace(to_replace=[r'\s+'], value=[""], regex=True)
                df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)
                # df.replace("gt", " / ")
                df.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
                df.text.str.translate(str.maketrans("","",string.punctuation))
                df.text.str.strip()
                print(df['hashtags'].unique())
                #word count
                dfs.append(df)
            r = dfs

        return r

    def saveModel(self):
        pass
        """
        # save the model to disk
        filename = 'finalized_model.sav'
        joblib.dump(model, filename)
        
        # some time later...
        
        # load the model from disk
        loaded_model = joblib.load(filename)
        result = loaded_model.score(X_test, Y_test)
        print(result)
        """
