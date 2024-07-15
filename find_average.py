import pandas as pd

df = pd.read_csv('data.csv')
df.head()

splitBy = input("Enter the column name to split by: ")

mean = {}
numCols = []

for col in df.columns:
    try:
        float(df[col][0])
        if col != splitBy: numCols.append(col)
    except:
        pass

g = df.groupby([splitBy])
unique = df[splitBy].unique()

for i in unique:
    for j in numCols:
        a = 0
        for k in g.get_group(i)[j]:
            if j not in mean: mean[j] = {}
            if a not in mean[j]: mean[j][a] = []
            mean[j][a].append(k)
            a += 1

for col in mean:
    for i in mean[col]:
        mean[col][i] = sum(mean[col][i])/len(mean[col][i])


cols = [list(mean.keys())]
for _ in range(len(mean[list(mean.keys())[0]])):
    c = []
    for col in mean.keys():
        c.append(mean[col][_])
    cols.append(c)

for col in cols:
    print(",".join([str(i) for i in col]), file=open("output.csv", "a"))