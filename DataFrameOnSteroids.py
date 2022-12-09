import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from collections import OrderedDict
from Utils import track, toLinesOfStr

# df.astype({"a": int, "b": complex})
# df.drop(columns, inplace=True, axis=1)
#df.drop_duplicates(subset=['brand', 'style'], keep='last')

def isCategorical(t):
    return t in [str, object]

def isNumeric(t):
    return not isCategorical(t)

class DataFrameOnSteroids(pd.DataFrame):
    def __init__(self, yNames=None, labelEncoders=None, normalizers=None, testDataIdx=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDerivedState(yNames=yNames, labelEncoders=labelEncoders, normalizers=normalizers, testDataIdx=testDataIdx)

    def getNonBaseState(self):
        return OrderedDict([("yNames",self.yNames), ("labelEncoders",self.labelEncoders),
                            ("normalizers",self.normalizers), ("testDataIdx", self.testDataIdx)])

    def copy(self):
        return DataFrameOnSteroids(data=super().copy(), **self.getNonBaseState())

    #def __getitem__(self, *args, **kwargs): #TODO overload = [] loc iloc
    #    DataFrameOnSteroids(data=super().__getitem__(*args, **kwargs), **self.getNonBaseState())

    def __reduce__(self):
        return type(self), (*self.getNonBaseState().values(), dict(self))

    def setDerivedState(self, yNames=None, labelEncoders=None, normalizers=None, testDataIdx=None):
        self.setTestDataIdx(testDataIdx)
        self.set_yNames(yNames)
        self.labelEncoders = labelEncoders if labelEncoders else {}
        self.normalizers = normalizers if normalizers else {}

    @property
    def normalizers(self) -> dict[str:object]:
        return self._normalizers
    @normalizers.setter
    def normalizers(self, n):
        self._normalizers = n

    @property
    def labelEncoders(self) -> dict[str:object]:
        return self._labelEncoders
    @labelEncoders.setter
    def labelEncoders(self, l):
        self._labelEncoders = l


    def set_yNames(self, yNames):
        if yNames is None:
            self.yNames = None
            return
        if type(yNames) == str:
            yNames = [yNames,]
        self.yNames = list(yNames)

    def setTestDataIdx(self, testDataIdx):
        if type(testDataIdx) in [int, float]:
            test_len = testDataIdx
            test_len = test_len if test_len >= 1 else int(test_len*self.shape[0])
            self.testDataIdx = np.random.choice(self.shape[0], test_len, replace=False)
            pass
        else:
            self.testDataIdx = testDataIdx

    @property
    def test_len(self) -> int:
        if self.testDataIdx is None:
            return 0
        return self.testDataIdx.size

    @property
    def dfTest(self): #TODO view
        if self.testDataIdx is None:
            self.setTestDataIdx(testDataIdx=0.33)
        d = self.getNonBaseState()
        return DataFrameOnSteroids(data=self.iloc[self.testDataIdx],
                                   **{k:d[k] for k in set(d) - {"testDataIdx"}})

    @property
    def dfTrain(self):
        if self.testDataIdx is None:
            self.setTestDataIdx(testDataIdx=0.33)
        d = self.getNonBaseState()
        return DataFrameOnSteroids(data=self[~self.index.isin(self.testDataIdx)],
                                   **{k:d[k] for k in set(d) - {"testDataIdx"}})

    @property
    def xNames(self):
        return list(self.columns.difference(self.yNames))

    @property
    def x(self): #TODO: currently retains all encoders and normalizers
        if self.testDataIdx is None:
            self.setTestDataIdx(testDataIdx=0.33)
        d = self.getNonBaseState()
        return DataFrameOnSteroids(data=self[self.xNames],
                                   **{k:d[k] for k in set(d) - {"yNames"}})

    @property
    def y(self): #TODO: currently retains all encoders and normalizers
        if self.testDataIdx is None:
            self.setTestDataIdx(testDataIdx=0.33)
        d = self.getNonBaseState()
        return DataFrameOnSteroids(data=self[self.yNames],
                                   **{k: d[k] for k in set(d) - {"yNames"}})

    @property
    def dfCategorical(self):
        d = self.getNonBaseState()
        return DataFrameOnSteroids(data=self[self.categoricalColumns],
                                   **{k:d[k] for k in set(d) - {"normalizers"}})

    @property
    def dfNumeric(self):
        d = self.getNonBaseState()
        return DataFrameOnSteroids(data=self[self.numericColumns],
                                   **{k:d[k] for k in set(d) - {"labelEncoders"}})

    @property
    def categoricalColumns(self) -> list[str]:
        if self.labelEncoders:
            return list(self.labelEncoders.keys())
        if self.normalizers:
            return list(set(self.columns) - set(self.normalizers.keys()))
        return [col for t, col in zip(self.dtypes, self.columns) if isCategorical(t)]

    @property
    def numericColumns(self) -> list[str]:
        if self.normalizers:
            return list(self.normalizers.keys())
        if self.labelEncoders:
             return list(set(self.columns) - set(self.labelEncoders.keys()))
        return [col for t, col in zip(self.dtypes, self.columns) if isNumeric(t)]

    def getSample(self, column, rows=5):
        df = self
        sample = []
        while len(sample) < rows:
            val = df[column][random.randint(0, df[column].shape[0]-1)]
            if val:
                sample.append(val)
        return sample

    def printColumns(self):
        [print(col) for col in self.columns]

    def printTypesSampleColNames(self, sampleLen=5):
        tmp = "{} ----- {} ----- {} ----- {}"
        [print(tmp.format(t, self.getSample(col, sampleLen), col, self[col].shape[0])) for t, col in zip(self.dtypes, self.columns)]

    def printUniqueColTypes(self):
        print(self.dtypes.unique())

    def printValueCountsPerCategoricalColumn(self):
        [print(f"col:{col}\n{toLinesOfStr(self[col].value_counts())}\n") for col in self.categoricalColumns]

    def getUniqueColValues(self, column):
        return self[column].unique()

    def areNotOutlierValues(self, column, outlierToleranceRate=1):
        df = self
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        otl = outlierToleranceRate
        filter = (df[column] >= Q1 - otl*1.5 * IQR * otl) & (df[column] <= Q3 + otl*1.5 * IQR)
        return filter

    def areOutlierValues(self, column, outlierToleranceRate=1):
        return ~self.areNotOutlierValues(column=column, outlierToleranceRate=outlierToleranceRate)

    def plot(self, x=None,y=None, groupBy=None):
        df = self
        fig = px.line(df, x=x, y=y, color=groupBy)
        fig.show()

    @track
    def drop_duplicates(self, *args, **kwargs):
        prev_size = self.size
        prev_rows = self.shape[0]
        out = super().drop_duplicates(*args, **kwargs)
        print(f"drop_duplicates changed size from {prev_size} to {self.size} ({prev_size/self.size} ratio)")
        print(f"drop_duplicates changed rows from {prev_rows} to {self.shape[0]} ({prev_rows/self.shape[0]} ratio)")
        return out

    @track
    def categoricalToNumeric(self, inplace=False) -> "DataFrameOnSteroids":
        df = self if inplace else self.copy()
        for col in self.categoricalColumns:
            le = preprocessing.LabelEncoder()
            #le.fit(df[col].unique())
            df[col] = le.fit_transform(df[col])
            if col not in df.labelEncoders:
                df.labelEncoders[col] = le
            #df[col] = le.transform(df[col])
        return None if inplace else df

    @track
    def numericalToCategorical(self, inplace=False) -> "DataFrameOnSteroids":
        df = self if inplace else self.copy()
        for col,le in df.labelEncoders.items():
            df[col] = le.inverse_transform(df[col])
        return None if inplace else df

    @track
    def normalize(self, inplace=False) -> "DataFrameOnSteroids":
        df = self if inplace else self.copy()
        for col in self.numericColumns:
            normalizer = MinMaxScaler()
            df[col] = normalizer.fit_transform(np.array(df[col]).reshape(-1,1))
            if col not in df.normalizers:
                df.normalizers[col] = normalizer
        return None if inplace else df

    @track
    def denormalize(self, inplace=False) -> "DataFrameOnSteroids":
        df = self if inplace else self.copy()
        for col, normalizer in df.normalizers.items():
            df[col] = normalizer.inverse_transform(df[col])
        return None if inplace else df

    @track
    def clean(self, inplace=False, outlierToleranceRate=1, dropDuplicates=False):
        if dropDuplicates:
            self.drop_duplicates(inplace=True, keep='last')
        df = self if inplace else self.copy()
        df.objectColToString(inplace=True)
        df.nanToMedian(inplace=True)
        #df.removeOutliers(inplace=True, outlierToleranceRate=outlierToleranceRate)
        if dropDuplicates:
            self.drop_duplicates(inplace=True, keep='last')
        return None if inplace else df

    @track
    def nanToMedian(self, inplace=False):
        df = self if inplace else self.copy()
        for t, col in zip(df.dtypes, df.columns):
            if isNumeric(t):
                median = df[col].dropna().median()
                df[col].fillna(median, inplace=True)
        return None if inplace else df

    @track
    def removeOutliers(self, inplace=False, outlierToleranceRate=1):
        df = self if inplace else self.copy()
        for t, col in zip(df.dtypes, df.columns):
            if isNumeric(t):
                df.drop(df[df.areOutlierValues(col, outlierToleranceRate=outlierToleranceRate)].index, inplace=True)
        return None if inplace else df

    @track
    def objectColToString(self, inplace=False):
        df = self if inplace else self.copy()
        for t, col in zip(df.dtypes, df.columns):
            if t in [object]:
                df[col] = df[col].map(lambda v: str(v))
        return None if inplace else df

    @track
    def convertAsImageEntries(self, key:list[str], expectedImageSize:int,
                              columnToScaleUniformlyOn:str, nameOfImageColumn="image", agg:str="median") -> pd.DataFrame:
        key = [key] if type(key) is str else key
        images = dict(tuple(self[self.columns.difference(self.yNames)].groupby(key)))
        aggStrategy_y = {yName: agg if yName not in self.categoricalColumns else "mode" for yName in self.yNames}
        df = self[[*key, *self.yNames]].groupby(key).agg(aggStrategy_y)
        col = columnToScaleUniformlyOn

        for k, im in images.items():
            if expectedImageSize < im.shape[0]:
                im = im[im.columns.difference(key)]
                begin = im[col].min()
                step = (im[col].max() - im[col].min()) / expectedImageSize
                lines = []
                for end in [begin+i*step for i in range(1, 1+expectedImageSize)]:
                    aggStrategy_x = {c:agg if c not in self.categoricalColumns else "mode" for c in im.columns}
                    line = im[(im[col] >= begin) & (im[col] <= end)].agg(aggStrategy_x)
                    if line.size > 0:
                        lines.append(line)
                    begin = end
                [lines.append(0 * im.iloc[0, :]) for _ in range(expectedImageSize-len(lines))]
                images[k] = pd.DataFrame(data=lines).to_numpy()
            else:
                raise Exception("Can only downscale!")
        d = [(*k, v) for k, v in images.items()]
        df = pd.DataFrame(d, columns=[*key, nameOfImageColumn]).set_index(key).join(df).reset_index()
        return df
