# %%
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#bulding the dataset
dataset = pd.read_csv("class01.csv", delimiter =",")

#create the feature matrix
Y01 = dataset["target"]
X01 = dataset.drop(["target"], axis = 1)

# Normalization:
X01_normal = (X01 - np.min(X01)) / (np.max(X01) - np.min(X01))

# Splitting the dataset into the Training set and Test set
X01_train, X01_test, y01_train, y01_test = train_test_split(X01_normal, Y01, train_size=350, random_state=1)

#train the model
nb = GaussianNB()
nb.fit(X01_train, y01_train)

#cros validation
scores = []
for i in range(2, 20):
    scores.append(cross_val_score(nb, X01_train, y01_train, cv=i).mean())

text =f"Naive Bayes score Test: {nb.score(X01_test, y01_test):.2f}", f"Naive Bayes score Train: {nb.score(X01_train, y01_train):.2f}"
textstr = "\n".join(text)

#graph cros validation
plt.plot(range(2,20),  scores, 'bo', range(2,20), scores, 'k')
plt.xticks(range(2,20))
plt.xlabel("Number of folds")
plt.ylabel("Accuracy")
plt.title("Cross Validation - 350 lines Train")
plt.legend(["Accuracy"])
plt.text(0.3, 0.1, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.savefig('cross_validation.png')
plt.show()

#variance train dataset
listTest = []
listTrain = []
porcentagem = []

for i in range(100, 800, 10):
    X01_train, X01_test, y01_train, y01_test = train_test_split(X01_normal, Y01, train_size=i, random_state=1)
    nb = GaussianNB()
    nb.fit(X01_train, y01_train)
    listTest.append(nb.score(X01_test, y01_test))
    listTrain.append(nb.score(X01_train, y01_train))
    porcentagem.append((i/dataset.shape[0])*100)

bestPerce = porcentagem[listTest.index(max(listTest))]
bestAcc = max(listTest)

#graph variation of train dataset
plt.plot(porcentagem,  listTest, porcentagem, listTrain)
plt.xlabel("Dataset size (%)")
plt.ylabel("Accuracy")
plt.title("Variation of the train dataset")
plt.legend(["Test", "Train"])
plt.annotate("", xy=(bestPerce, max(listTest)), xytext=(porcentagem[listTest.index(max(listTest))], max(listTest)+0.01), arrowprops=dict(facecolor='pink', shrink=0.05),)
plt.text(0.4, 0.1 , f"Porcentagem: {bestPerce}% \nAccuracy: {bestAcc}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.savefig('data_train_var.png')
plt.show()

# %%
