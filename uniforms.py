import pandas as pd
import numpy as np

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

baseio = BaseIO()

"""
https://stackabuse com/how-to-merge-dataframes-in-pandas/#mergedataframesusingappend


pd.concat(dataframes, axis=0, join='outer', ignore_index=False, keys=None,
        levels=None, names=None, verify_integrity=False, sort=False, copy=True)

Here are the most commonly used parameters for the concat() function:

objs is the list of DataFrame objects ([df1, df2, ...]) to be concatenated
axis defines the direction of the concatenation, 0 for row-wise and 1 for column-wise
join can either be inner (intersection) or outer (union)
ignore_index by default set to False which allows the index values to remain as they were in the original DataFrames, can cause duplicate index values. If set to True, it will ignore the original values and re-assign index values in sequential order
keys allows us to construct a hierarchical index. Think of it as another level of the index that appended on the outer left of the DataFrame that helps us to distinguish indices when values are not unique

"""
def concatDataframeRows(dataframes):
    return pd.concat(dataframes, axis=0, join='outer')

def initialize():
    crawled = [
        baseio.inputFmt("ruu.minol", 'ruu.minol.csv'),
        baseio.inputFmt("ruu.minol2", 'ruu.minol2.csv'),
        baseio.inputFmt("ruu.minuman.beralkohol", 'ruu.minuman.beralkohol.csv'),
        baseio.inputFmt("ruu.minuman.beralkohol2", 'ruu.minuman.beralkohol2.csv'),
        baseio.inputFmt("ruu.miras", 'ruu.miras.csv'),
        baseio.inputFmt("ruu.miras2", 'ruu.miras2.csv'),
    ]

    useCols = [
        'status_id',
        'created_at',
        'screen_name',
        'text',
        'preprocessed',
        'classify_data',
        'classified'
    ]

    dataframes = []

    # print(crawled)
    for c in crawled:
        # get classified
        filen = './' + c['name'] + '.classified.csv'
        print(filen)
        df = pd.read_csv(filen, header=0, lineterminator='\n', usecols=useCols)
        dataframes.append(df)
        # preprocessor.results['sentimen.y1'][c['name']] = c
    
    concated = concatDataframeRows(dataframes)
    concated.to_csv("ruu.all.classified.csv")

initialize()
"""
'status_id', 'created_at', 'screen_name', 'text', 'preprocessed', 'classify_data', 'classified'
"""