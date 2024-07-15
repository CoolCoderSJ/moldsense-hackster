from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data.csv')
df.head()

y = df['expired']
x = df.drop('expired', axis=1)
x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size=0.25, random_state=125
)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train);

from sklearn.metrics import accuracy_score, f1_score
import pickle

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

with open('model.pkl','wb') as f:
	pickle.dump(model,f)

