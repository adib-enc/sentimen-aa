import re
import pandas as pd
import numpy as np
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
"""
# create stemmer
from pprint import pprint
factory = StemmerFactory()
stemmer = factory.create_stemmer()
"""

# https://github.com/masdevid/ID-Stopwords
stopwords = open("dummy/id.stopwords.02.01.2016.txt").read()
insetLexicons = pd.read_csv("dummy/inset.lex/all-inset-lex.csv", header=0).set_index('word').to_dict()['weight']

class BaseIO:
    datas = {
        "csv": None,
        "xls": None,
        "dataframe": None,
    }

    def __init__(self):
        pass
    
    def fromCsv(self, filename):
        pass
    
    def toCsv(self, filename):
        pass
    
    def toXls(self, filename):
        pass
    
    def inputFmt(self, name, filename):
        return {
            "name": name,
            "filename": filename,
        }

class Processor:
    name = "Base processor"
    params = {}
    results = {}
    baseio = None

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

    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()
    stemmer = None

    def __init__(self, *args, **kwargs):
        super(PreProcessor, self).__init__(*args, **kwargs)
        self.initStemmer()

    def initStemmer(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

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
        1 method 4 all
        # Tokenisasi
        # Data cleaning
        # Case folding
        # Filterisasi atau stopword removal
        # Stemming
    """
    def tokenizeAndClean(self, text, toWords = False):
        tokenized = re.sub("[\r\n]"," ", text)
        
        if toWords:
            tokenized = [e for e in re.split(r'(\W+)', tokenized) if ' ' not in e]
            
            res = []
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

            ret = res
            
        else:
            tokenized = re.split(r"[\.\?\!][ \n]", tokenized) # sentences
            tokenized = [t.strip().lower() for t in tokenized if t!="\r"]
            ret = tokenized

        return ret
    
    def classifyWord(self, word):
        try:
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
        

    def preproccess(self):
        print("preproccess")

        crawlResult = self.results['crawling']
        keys = self.results['crawling'].keys()
        for crawled in keys:
            df = crawlResult[crawled]['dataframe']
            dftext = df['text']
            self.current['file'] = crawled
            df["preprocessed"] = 1
            preprocesseds = []
            self.dftext = dftext
            
            print(crawled)
            print("self.dftext length",len(self.dftext))
            for index, text in enumerate(self.dftext):
                print("text::",index)
                preprocessed = self.tokenizeAndClean(text, True)
                if preprocessed != '':
                    preprocesseds.append(" ".join(preprocessed))
                
                self.preprocesseds.append(preprocesseds)
                # break
            
            try:
                df['preprocessed'] = preprocesseds
                df.to_csv('./dummy/preprocess/' + crawled + ".preprocessed.csv")
            except Exception as e:
                print(e)
                print(self.preprocesseds)
                pass
                
            self.results['preprocess']['dataframe'] = df
    
    def classify(self):
        print("classify::inset lexicon")

        preprocessResult = self.results['preprocess']
        keys = preprocessResult.keys()
        
        df = None

        for preprocessed in keys:
            df = preprocessResult[preprocessed]['dataframe']
            dfpreprocessed = df['preprocessed']

            df['classify_data'] = 1 # default as positiv
            df['classified'] = 'p' # default as positiv
            self.dfpreprocessed = dfpreprocessed
            self.current['file'] = preprocessed
            classify_datas = []
            classifieds = []
            
            for index, text in enumerate(self.dfpreprocessed):
                print(preprocessed,"text::",index)
                dic, score = self.classifySentence(text)
                classify_datas.append(dic)

                if score > 0:
                    classifieds.append('positive')
                elif score == 0:
                    classifieds.append('netral')
                elif score < 0:
                    classifieds.append('negative')
                # print(self.classifySentence(sentence))
                # break
            
            try:
                df['classify_data'] = classify_datas
                df['classified'] = classifieds
                df.to_csv('./dummy/classified/' + preprocessed + ".classified.csv")
            except Exception as e:
                print(e)
                print(classifieds)
                pass

        self.results["sentimen.y1"]['dataframe'] = df
    
    def tfidfWeighting(self):
        print("classify::tfidfWeighting")
        self.results["tfidf"] = 1

    def process(self):
        pass

    def resetProgress(self):
        self.progress = 0

class KMeansProcessor(Processor):
    name = "KMeansProcessor"
    results = {
        "cluster.x10": None,
    }

    def __init__(self, *args, **kwargs):
        super(KMeansProcessor, self).__init__(*args, **kwargs)

    def process(self):
        pass

class SVMNBCProcessor(Processor):
    name = "SVMNBCProcessor"
    results = {
        "sentimen.y2": None
    }

    def __init__(self, *args, **kwargs):
        super(SVMNBCProcessor, self).__init__(*args, **kwargs)

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
pKmp = KMeansProcessor()
pSvmnbc = SVMNBCProcessor()
pRegresi = RegresiLMProcessor()

def initialize():
    crawled = [
        baseio.inputFmt("ruu.minol", 'ruu.minol.csv'),
        baseio.inputFmt("ruu.minol2", 'ruu.minol2.csv'),
        baseio.inputFmt("ruu.minuman.beralkohol", 'ruu.minuman.beralkohol.csv'),
        baseio.inputFmt("ruu.minuman.beralkohol2", 'ruu.minuman.beralkohol2.csv'),
        baseio.inputFmt("ruu.miras", 'ruu.miras.csv'),
        baseio.inputFmt("ruu.miras2", 'ruu.miras2.csv'),
    ]

    preprocessor.resultToDict()

    # print(crawled)
    for c in crawled:
        # print(pPre.results['crawling'])
        filen = './dummy/preprocess/' + c['name'] + '.preprocessed.csv'
        print(filen)
        c['dataframe'] = pd.read_csv(filen, header=0, lineterminator='\n')
        preprocessor.results['preprocess'][c['name']] = c
        # c['dataframe'] = pd.read_csv('./dummy/'+c['filename'], header=0)
        # preprocessor.results['crawling'][c['name']] = c
        # break


initialize()
# preprocessor.preproccess()
# preprocessor.classify()
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
df = preprocessor.results['crawling']['ruu.minol']['dataframe']; dfp = preprocessor.results['preprocess']['ruu.minol']['dataframe']
"""