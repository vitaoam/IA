# 1. Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train.drop('Cabin', axis=1, inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test.drop('Cabin', axis=1, inplace=True)

le = LabelEncoder()
for col in ['Sex', 'Embarked']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

for dataset in [train, test]:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = le.fit_transform(dataset['Title'])

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Sex', data=train)
plt.title('Sobreviventes por Sexo')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title('Distribuição de Idade')
plt.show()


features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = train[features]
y = train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
print("Random Forest:\n", classification_report(y_val, y_pred_rf))

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_val)
print("Decision Tree:\n", classification_report(y_val, y_pred_dt))


scaler = StandardScaler()
X_cluster = scaler.fit_transform(train[features])

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_cluster)
train['Cluster'] = clusters

plt.figure(figsize=(6,4))
sns.scatterplot(x='Age', y='Fare', hue='Cluster', data=train, palette='Set1')
plt.title('Clusters por Idade e Tarifa')
plt.show()

assoc_data = train.copy()
assoc_data['Sex'] = assoc_data['Sex'].map({0: 'male', 1: 'female'})
assoc_data['Survived'] = assoc_data['Survived'].map({0: 'not_survived', 1: 'survived'})
assoc_data['Pclass'] = assoc_data['Pclass'].astype(str)
assoc_data['FamilySize'] = assoc_data['FamilySize'].astype(str)
assoc_data['Title'] = assoc_data['Title'].astype(str)
assoc_data = assoc_data[['Sex', 'Pclass', 'Survived', 'FamilySize', 'Title']]

assoc_bin = pd.get_dummies(assoc_data)

frequent = apriori(assoc_bin, min_support=0.1, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=0.6)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(5))
