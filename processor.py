
class Processor:
    name = "Base processor"
    params = {}
    results = {}

    def __init__(self, name = "", params = {}):
        # self.name = name
        self.params = params
    
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

    def __init__(self, *args, **kwargs):
        super(PreProcessor, self).__init__(*args, **kwargs)

    def crawl(self):
        print("crawling")
    
    def preproccess(self):
        print("preproccess")
    
    def classify(self):
        print("classify::inset lexicon")
        self.results["sentimen.y1"] = 1
    
    def tfidfWeighting(self):
        print("classify::tfidfWeighting")
        self.results["tfidf"] = 1

    def process(self):
        pass

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

p = Processor()
pPre = PreProcessor()
pKmp = KMeansProcessor()
pSvmnbc = SVMNBCProcessor()
pRegresi = RegresiLMProcessor()

print(pPre.name)
print(pKmp.name)
print(pSvmnbc.name)
print(pRegresi.name)