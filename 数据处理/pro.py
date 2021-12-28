#coding : utf-8
import pandas as pd
import numpy as np
import re
import math


def match_date(text):
    pattern = r'(\d{4}-\d{1,2}-\d{1,2})'
    pattern = re.compile(pattern)
    result = pattern.findall(text)
    return result

def processing(filename):
    dataset = pd.read_excel(filename)
    return dataset

def time_tiqu(dataset):
    dropnan= dataset[["pubdate","rating"]]
    dict = {"pubdate":[],"rating":[]}
    for i in range(len(dropnan)):
        if dropnan["pubdate"][i] !="" and dropnan["pubdate"][i] is not np.nan:
            if not isinstance(dropnan["rating"][i], str):
                #print(dropnan["pubdate"][i])
                if dropnan["rating"][i]>0:
                    if isinstance(dropnan["pubdate"][i], str):
                        tmp = match_date(dropnan["pubdate"][i])
                        if len(tmp)==0:
                            continue
                        dict["pubdate"].append(int(tmp[0][0:4]))
                    else:
                        dict["pubdate"].append(int(str(dropnan["pubdate"][i])[0:4]))
                    dict["rating"].append(dropnan["rating"][i])
    processed = pd.DataFrame(dict)
    return processed

def years_process(dataset,start,end):
    dict = {"pubdate":[],"rating":[]}
    for i in range(len(dataset)):
        if start<=dataset["pubdate"][i]<=end:
            dict["pubdate"].append(dataset["pubdate"][i])
            dict["rating"].append(dataset["rating"][i])
    tmp_year = pd.DataFrame(dict)
    return tmp_year

def decribe_df(dataset):
    dataDescribe = dataset["rating"].describe()
    #print(dataDescribe[0],dataDescribe[1])
    return dataDescribe

def merge_data(dataset):
    base_df = pd.DataFrame()
    y60 = decribe_df(years_process(dataset,1960,1969))
    base_df =pd.concat([base_df, y60],axis=1)
    y70 = decribe_df(years_process(dataset, 1970, 1979))
    base_df =pd.concat([base_df, y70], axis=1)
    y80 = decribe_df(years_process(dataset, 1980, 1989))
    base_df =pd.concat([base_df, y80], axis=1)
    y90 = decribe_df(years_process(dataset, 1990, 1999))
    base_df =pd.concat([base_df, y90], axis=1)
    y00 = decribe_df(years_process(dataset, 2000, 2009))
    base_df =pd.concat([base_df, y00], axis=1)
    y10 = decribe_df(years_process(dataset, 2010, 2019))
    base_df =pd.concat([base_df, y10], axis=1)
    # y20 = decribe_df(years_process(dataset, 2020, 2029))
    # base_df =pd.concat([base_df, y20], axis=1)
    base_df = base_df.drop(labels=["count"], axis=0)
    base_df.columns = ["1960s","1970s","1980s","1990s","2000s","2010s"]
    return base_df


def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 标准化
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)
    lnf = [[None] * cols for i in range(rows)]
    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = np.array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = np.array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf
    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
        # 计算各样本的综合得分,用最原始的数据
    w = pd.DataFrame(w)
    return w

if __name__ == "__main__":
    filename = input("Please input the excel file name: ")+".xlsx"
    dataset = processing(filename)
    New = time_tiqu(dataset)
    print(New)
    # A = years_process(New,1960,1969)
    # print(decribe_df(A))
    # print(type(decribe_df(A)))
    cal_df = merge_data(New)
    print(cal_df)
    w = cal_weight(cal_df)  # 调用cal_weight
    w.index = cal_df.columns
    w.columns = ['weight']
    print(w)