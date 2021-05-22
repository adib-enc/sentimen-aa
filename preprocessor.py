import re
# import pandas as pd # normal pandas

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

"""
speed up training
https://medium.com distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1
https://towardsdatascience.com pca-using-python-scikit-learn-e653f8989e60
https://medium.com distributed-computing-with-ray/how-to-speed-up-pandas-with-modin-84aa6a87bcdb
"""

from baseio import BaseIO
from processor import Processor

class PreProcessor(Processor):
    name = "Preprocessor"
    results = {
        "crawling": None,
        "preprocess": None,
        "sentimen.y1": None,
        "tfidf": None,
    }
    current = {
        "file": "",
        "index": "",
    }
    preprocesseds = []
    wordsCache = {}
    progress = 0
    globalWords = []

    stopwords = None
    insetLexicons = None

    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()
    stemmer = None

    def __init__(self, *args, **kwargs):
        super(PreProcessor, self).__init__(*args, **kwargs)
        # self.initStemmer().initData()

    def initStemmer(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        return self
    
    def initData(self):
        # https://github.com/masdevid/ID-Stopwords
        self.stopwords = open("dummy/id.stopwords.02.01.2016.txt").read()
        self.insetLexicons = pd.read_csv("dummy/inset.lex/all-inset-lex.csv", header=0).set_index('word').to_dict()['weight']

        return self

    def crawl(self):
        print("crawling")
    
    def caseFolds(self, sentence):
        pass

    def tokenize(self, text, toWords = False):
        # tokenized = ""
        tokenized = re.sub("[\r\n]"," ", text)
        if toWords:
            tokenized = re.split(r'(\W+)', tokenized)
        else:
            tokenized = re.split(r"[\.\?\!][ \n]", tokenized) # sentences
        tokenized = [t.strip().lower() for t in tokenized if t!="\r"]

        return tokenized

    def printProgress(self):
        if self.progress % 100 == 0:
            pref = ["file", str(self.current['file']),"idx", str(self.current['index'])]
            pref = "::".join(pref)
            print("progress::", pref, self.progress)

    def splitToWords(self, text):
        return [e for e in re.split(r'(\W+)', text) if ' ' not in e]

    def stripLowerAndStem(self, text):
        t = text.strip().lower()
        if self.stemmer:
            return self.stemmer.stem(t)
        
        return t

    """
    yunus tweet cleaner
    """
    def remove_tweet_special(self, text):
        # remove tab, new line, ans back slice
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        # remove non ASCII (emoticon, chinese word, .etc)
        text = text.encode('ascii', 'replace').decode('ascii')
        # remove mention, link, hashtag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        # remove incomplete URL
        return text.replace("http://", " ").replace("https://", " ")

    #remove number
    def remove_number(self, text):
        return  re.sub(r"\d+", "", text)

    #remove punctuation
    def remove_punctuation(self, text):
        return text.translate(str.maketrans("","",string.punctuation))

    #remove whitespace leading & trailing
    def remove_whitespace_LT(self, text):
        return text.strip()

    #remove multiple whitespace into single whitespace
    def remove_whitespace_multiple(self, text):
        return re.sub('\s+',' ',text)
    
    def cleanTweet(self, text):
        cleanTw = text

        cleanTw = self.remove_tweet_special(cleanTw)
        cleanTw = self.remove_number(cleanTw)
        cleanTw = self.remove_punctuation(cleanTw)
        cleanTw = self.remove_whitespace_LT(cleanTw)
        cleanTw = self.remove_whitespace_multiple(cleanTw)

        return cleanTw
    """
    ./yunus
    """

    """
        1 method 4 all
        # Tokenisasi
        # Data cleaning
        # Case folding
        # Filterisasi atau stopword removal
        # Stemming
    """
    def tokenizeAndClean(self, text, toWords = False):
        detailDict = {}
        detailDict['text'] = text
        text = self.cleanTweet(text)
        detailDict['cleaning'] = text

        tokenized = re.sub("[\r\n]"," ", text)
        stopwords = self.stopwords
        
        if toWords:
            tokenized = [e for e in re.split(r'(\W+)', tokenized) if ' ' not in e]
            detailDict['tokenized'] = tokenized
            
            res = []

            joined = " ".join(tokenized).lower()
            detailDict['folding'] = joined
            
            for t in tokenized:
                if t == '':
                    # print(t)
                    continue

                lowered = t.lower()
                

                if len(t) < 2:
                    continue

                # print(lowered, self.wordsCache.keys())
                # if lowered in self.wordsCache.keys():
                #     # res.append(self.wordsCache[lowered])
                #     continue

                if t!="\r" and t not in string.punctuation and t not in stopwords and t != '':
                    wordresult = self.stripLowerAndStem(t)
                    if wordresult != '':
                        res.append(wordresult)
                    # self.wordsCache[lowered] = wordresult
                self.progress += 1
                self.current['index'] = self.progress
                self.printProgress()
            detailDict['stopwordremov-stemmed'] = res
            ret = res
            
        else:
            tokenized = re.split(r"[\.\?\!][ \n]", tokenized) # sentences
            tokenized = [t.strip().lower() for t in tokenized if t!="\r"]
            ret = tokenized

        return ret, detailDict
    
    def classifyWord(self, word):
        try:
            insetLexicons = self.insetLexicons
            if word in insetLexicons:
                score = insetLexicons[word]
            else:
                score = 0

            return int(score)
        except Exception as e:
            print(e)
            return 0
    
    def classifySentence(self, sentence):
        score = 0
        
        if isinstance(sentence, str):
            sentence = self.splitToWords(sentence)

        dic = {}
        if isinstance(sentence, list):
            for word in sentence:
                wscore = self.classifyWord(word)
                score += wscore
                dic[word] = wscore
            
            return dic, score
        else:
            return {}, 0
        # try:
        # except Exception as e:
        #     print(e)
        #     return {}, 0
        

    def preproccess(self, df, col="preprocessed", alias="preprocessed"):
        print("preproccess")
        dftext = df[col]
        self.current['file'] = alias
        df["preprocessed"] = 1
        preprocesseds = []
        self.dftext = dftext
        
        total = len(self.dftext)
        now = 0
        print(alias)
        print("self.dftext length",total)
        
        preprocList = []

        started = self.progressor({
            'type': "start",
            'total': total,
            'now': now,
        })
        self.progressModer = 100

        for index, text in enumerate(self.dftext):
            print("text::",index)
            preprocessed, detailDict = self.tokenizeAndClean(text, True)
            if preprocessed != '':
                preprocesseds.append(" ".join(preprocessed))
            
            self.progressor({
                'type': "progress",
                'total': total,
                'now': now,
            })
            # preprocList.append(detailDict)

            self.preprocesseds.append(preprocesseds)
            now += 1
            # break
        
        # df = pd.DataFrame(preprocList)
        # df.to_excel(crawled+"preproc.3.text.xls")

        # break
        try:
            df['preprocessed'] = preprocesseds
            df.to_csv('./dummy/preprocess/' + alias + ".preprocessed.csv")
        except Exception as e:
            print(e)
            print(self.preprocesseds)
            pass
        
        return df

    def preproccessAll(self):
        print("preproccess")

        crawlResult = self.results['crawling']
        keys = self.results['crawling'].keys()
        for crawled in keys:
            if crawled != "ruu.miras2":
                continue
            df = crawlResult[crawled]['dataframe']
            df = self.preproccess(df, "preprocessed", alias=crawled)
                
            self.results['preprocess']['dataframe'] = df
    
    def classify(self, df, col="preprocessed", alias="classified"):
        print("classify::inset lexicon")
        dfpreprocessed = df[col]

        df['classify_data'] = 1 # default as positiv
        df['classified'] = 'p' # default as positiv
        self.dfpreprocessed = dfpreprocessed
        self.current['file'] = alias
        classify_datas = []
        classifieds = []

        total = len(dfpreprocessed)
        now = 0
        
        started = self.progressor({
            'type': "start",
            'total': total,
            'now': now,
        })
        
        for index, text in enumerate(self.dfpreprocessed):
            dic, score = self.classifySentence(text)
            classify_datas.append(dic)

            if score > 0:
                # classifieds.append('positive')
                classifieds.append(1)
            elif score == 0:
                # classifieds.append('netral')
                classifieds.append(0)
            elif score < 0:
                # classifieds.append('negative')
                classifieds.append(-1)
            
            self.progressor({
                'type': "progress",
                'total': total,
                'now': now,
            })
            now += 1
            # print(self.classifySentence(sentence))
            # break
            
        try:
            df['classify_data'] = classify_datas
            df['classified'] = classifieds
            df.to_csv('./dummy/classified/' + alias + ".classified.csv")
        except Exception as e:
            print(e)
            print(classifieds)
            pass

        # bug, wrong file, not global result
        # self.results["sentimen.y1"]['dataframe'] = df
        return df

    def classifyAll(self):
        print("classify::inset lexicon")

        preprocessResult = self.results['preprocess']
        keys = preprocessResult.keys()
        
        df = None

        for preprocessed in keys:
            df = preprocessResult[preprocessed]['dataframe']
            df = self.classify(df, col="preprocessed", alias=preprocessed)

        # bug, wrong file, not global result
        self.results["sentimen.y1"]['dataframe'] = df
    
    def formGlobalWords(self, df, alias = "reclean", toFile = True):
        print("forming global words..")
        dfpreprocessed = df['preprocessed']
        # self.dfpreprocessed = dfpreprocessed
        alias = str(alias)
        
        for index, text in enumerate(dfpreprocessed):
            print(alias,"text::",index)
            
            if isinstance(text, str):
                arr = text.split(' ')
                for w in arr:
                    self.globalWords.append(w)
        
        if toFile:
            fo = open("dummy/"+alias+"globalWords",'w')
            gw = map(lambda x:x+'\n', self.globalWords)
            fo.writelines(gw)
            fo.close()

        pass

    def formGlobalWordsAll(self, toFile = True):
        sentimenY1sResult = self.results['sentimen.y1']
        keys = sentimenY1sResult.keys()

        for sentimenY1 in keys:
            df = sentimenY1sResult[sentimenY1]['dataframe']
            dfpreprocessed = df['preprocessed']
            # self.dfpreprocessed = dfpreprocessed
            
            for index, text in enumerate(dfpreprocessed):
                print(sentimenY1,"text::",index)
                
                if isinstance(text, str):
                    arr = text.split(' ')
                    for w in arr:
                        self.globalWords.append(w)
        if toFile:
            fo = open("dummy/globalWords",'w')
            gw = map(lambda x:x+'\n', self.globalWords)
            fo.writelines(gw)
            fo.close()

        pass

    # global
    def termFrequency(self, terms):

        if isinstance(terms, str):
            dic = dict(zip(list(terms),[list(terms).count(i) for i in list(terms)]))
        else:
            dic = dict(zip(terms,[terms.count(i) for i in terms]))

        return dic

    # filen = './dummy/classified/ruu.all.classified.csv'
    # print(filen)
    # dfClassified = pd.read_csv(filen, header=0, lineterminator='\n')
    # dfTerms = self.getGlobalWords(typef = "df8k")

    def documentFrequency(self, dfClassified, dfTerms, alias = "classified"):
        print("forming doc freq")
        time.sleep(1)
        preprocessed = dfClassified['preprocessed']

        total = len(preprocessed) * len(dfTerms['words'])
        docFreqs = []
        now = 0
        
        started = self.progressor({
            'type': "start",
            'total': total,
            'now': now,
        })
        self.progressModer = 100000
        
        for i, w in enumerate(dfTerms['words']):
            DF = 0

            for p in preprocessed:
                if isinstance(p, str):
                    psplit = p.split(" ")
                    if w in psplit:
                        DF += 1

                    self.progressor({
                        'type': "progress",
                        'total': total,
                        'now': now,
                    })
                    now += 1
            
            docFreqs.append(DF)
        
        dfTerms['docfreq'] = docFreqs
        dfTerms.to_csv("dummy/"+alias+".term.docfreq.csv")

    # filen = './dummy/classified/ruu.all.classified.csv'
    # print(filen)
    # dfClassified = pd.read_csv(filen, header=0, lineterminator='\n')
    # dfTerms = self.getGlobalWords(typef = "df-docfreq")

    def tfidfWeighting(self, dfClassified, dfTerms, alias = "tfidf1"):
        print("tfidf weighting..")
        time.sleep(1)
        dfWordKey = dfTerms.set_index('words').to_dict()
        docFreq = dfWordKey['docfreq']
        
        preprocessed = dfClassified['preprocessed']
        
        dfRet = None
        N = 6103
        # wordWeights = {}

        total = len(preprocessed) * len(dfTerms['words'])
        now = 0
        
        started = self.progressor({
            'type': "start",
            'total': total,
            'now': now,
        })

        self.progressModer = 100000

        for word in dfTerms['words']:
            wList = []
            for p in preprocessed:
                if isinstance(p, str):
                    psplit = p.split(" ")

                    self.progressor({
                        'type': "progress",
                        'total': total,
                        'now': now,
                    })

                    TF = psplit.count(word)
                    W = 0
                    
                    if docFreq[word] > 0:
                        IDF = np.log( N / docFreq[word] )
                        W = TF * IDF

                    wList.append(W)
                else:
                    wList.append(0)
                now += 1
                
            dfClassified[word] = wList
            # wordWeights[word] = wList
        # self.results["tfidf"] = 1
        dfClassified.to_csv("dummy/"+alias+"sentimenY1result.csv")
    
    def tfPerTweet(self, dfClassified, dfTerms, alias = "tf"):
        print("tfidf pertweet weighting..")
        time.sleep(1)
        
        preprocessed = dfClassified['preprocessed']
        # wordWeights = {}

        total = len(preprocessed) * len(dfTerms['words'])
        now = 0
        
        started = self.progressor({
            'type': "start",
            'total': total,
            'now': now,
        })

        self.progressModer = 100000

        for word in dfTerms['words']:
            wList = []
            for p in preprocessed:
                if isinstance(p, str):
                    psplit = p.split(" ")

                    self.progressor({
                        'type': "report",
                        'total': total,
                        'now': now,
                    })

                    TF = psplit.count(word)

                    wList.append(TF)
                else:
                    wList.append(0)
                now += 1
                
            dfClassified[word] = wList
            # wordWeights[word] = wList
        # self.results["tfidf"] = 1
        dfClassified.to_csv("dummy/"+alias+"tf.pertweet.csv")

    def process(self):
        pass

    def resetProgress(self):
        self.progress = 0
