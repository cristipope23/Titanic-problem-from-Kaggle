import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#citim fisierele
train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")
combine = [train_df,test_df]


#analizam datele care s-ar putea corela

#print(train_df.head())
#print(train_df.tail())
#print(train_df.describe())

#cati oameni au supravietuit din fiecare clasa
print(train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False))

print('\n\n')

#cati barbati si cate femei au supravietuit
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('\n\n')

#cati au supravietuit in functie de cate rude aveau
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#vizualizam datele

#vedem cati oameni au murit si la ce varsta
"""
g=sns.FacetGrid(train_df, col='Survived')
g=g.map(plt.hist,'Age',bins=20)
plt.show()
"""

#rata de mortalitate de la fiecare clasa
"""
plt.figure(0)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.show()
"""

#rata mortalitatii din fiecare oras
"""
plt.figure(0)
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None, order=None)
grid.add_legend()
plt.show()
"""


#rata mortalitatii pe numarul de membrii din familie
print('\n\n')

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#procentul supravietuirii a celor singuri fata de familii
print('\n\n')

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


#dam drop la cabina si la ticket
#nu ne intereseaza si ne usureaza analiza
print('\nDrop la cabina si la ticket\n')
print("Inainte", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("Dupa", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

print('\n\n')


###################    WRANGLE DATA    ###################

#analizam dupa nume (Mr,Mrs,Sir etc.)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

print('\nConvertim numele importante(Mr,Miss,Mrs,Rare etc. in cifre\n')
#convertim numele importante in cifre
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
print(train_df.head())

print('\nDrop la nume pentru ca nu ne mai intereseaza\n')

#dam drop la nume, pentru ca nu ne mai intereseaza
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

#convertim sexele in cifre 0 si 1
print('\nConvertim sexele in cifre ( 0 si 1 ) \n')
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


print(train_df.head())
print('\n\n')
#completam featurile cu valori nule
#incepem cu varsta
guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

print(train_df.head())

print('\nCreez intervale de ani si rata de supravieturie\n')

#creez intervale de ani si rata de supravieturie
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
print('\nTransform anii in cifre pe baza celor 5 intervale\n')
#transform anii in cifre pe baza celor 5 intervale
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

print(train_df.head())
print('\n\n')
#stergem intervalele 
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df] #salvam in combine ambele dataframe-uri
#print(train_df.head())

#Create new feature combining existing features
#drop Parch, SibSp, and FamilySize features in favor of IsAlone.
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
#print(train_df.head())

#facem un nou feature combinand Pclas cu Age
print('\nFacem un nou feature combinand Pclas cu Age\n')
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

#orasul de unde au murit cei mai multi oameni
print('\nOrasul din care au murit cei mai multi oameni\n')
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#convertim locatiile cu cifre
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print('\nConvertim Embarked in cifre\n')

print(train_df.head())

#completam si la Fare elementul care lipseste din tabel
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
#print(test_df.head())

#construim un interval de preturi ale biletelor si rata de supravietuire
print('\nConstruim un interval de preturi ale biletelor si rata de supravietuire')
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print('\n',train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

#Convertim categoria Fare in cifre pe baza intervalului de preturi de mai sus
print('\nConvertim categoria Fare in cifre pe baza intervalului de preturi de mai sus')
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print('\n',train_df.head(10),'\n\n\n  \
         MACHINE LEARNING \n\n\n')

############################### MACHINE LEARNING ###############################

# X-features
# X_train -> training data set
# y- label-ul
# Y_train -> set of labels to all the data in x_train

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print('Liner SVC:',acc_linear_svc,'\n')

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('Logistic Regression:',acc_log,'\n')

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('Decision Tree:',acc_decision_tree,'\n')

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Random Forest',acc_random_forest,'\n')

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print('Perceptron',acc_perceptron,'\n')

# k-Nearest Neighbors (knn)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print('knn:',acc_knn,'\n\n')

models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest', 'Perceptron', 
              'Linear SVC', 'Decision Tree'],
    'Score': [acc_knn, acc_log, 
              acc_random_forest,acc_perceptron, 
              acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
