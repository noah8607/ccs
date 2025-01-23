import pickle
import pandas as pd
import os

class AllData():
    filepath = "data/alldata.pkl"
    xlsxpath = "通话数据/通话数据.xlsx"
    mp3path = "通话数据/mp3/"
    alldata = None

    def __init__(self) -> None:
        if AllData.alldata is None:
            if os.path.exists(AllData.filepath):
                with open(AllData.filepath, 'rb') as f:
                    AllData.alldata = pickle.load(f)
            else:
                AllData.alldata = {}

    @property
    def types(self):
        return AllData.alldata['types'].copy()
    
    def gettid(self, typestr):
        return AllData.alldata['types'].index(typestr)
    
    def gettype(self, tid):
        return AllData.alldata['types'][tid]

    def save(self):
        with open(AllData.filepath, 'wb') as f:
            pickle.dump(AllData.alldata, f)

    def loadexcel(self):
        df = pd.read_excel(AllData.xlsxpath)
        origdata = df.to_dict("records")
        AllData.alldata['origdata'] = origdata
        types = sorted(list(set(df.to_dict()['跟进等级'].values())))
        AllData.alldata['types'] = types
        self.save()

    def loadmp3(self):
        records = []
        for r in AllData.alldata['origdata']:
            rid = r['录音id']
            filepath = AllData.mp3path + rid + '.mp3'
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    mp3 = f.read()
                tid = AllData.alldata['types'].index(r['跟进等级'])

                records.append({'rid':rid, 'mp3':mp3, 'typeid':tid})
        AllData.alldata['train_records'] = records
        self.save()


if __name__ == '__main__':
    ad = AllData()
    if len(AllData.alldata) > 0:
        print("存在数据，退出数据初始化程序.")
    else:
        ad.loadexcel()
        print(f'原始分类表包含数据{len(AllData.alldata["origdata"])}条')
        ad.loadmp3()
        print(f'加载mp3记录{len(AllData.alldata["train_records"])}条')

