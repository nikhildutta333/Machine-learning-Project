# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#del X,y,y_train,X_test,y_test,data
#del one_hot,one_hot1,one_hot2,one_hot3,one_hot4,one_hot5,one_hot6,one_hot7,one_hot8,one_hot9,one_hot9_1,one_hot9_2,one_hot9_3
#data = data.drop(columns="GradeID")

#to see duplicate columns in dataframe
df[df.index.duplicated()]

# Read the data
data1 = pd.read_csv('Data.csv')

#data = data.drop('Class',axis = 1)
#'NationalITy','PlaceofBirth','StageID','GradeID','SectionID','Topic','Semester','Relation','ParentAnsweringSurvery','ParentschoolSatisfaction','StudentAbsenceDays',axis = 1)


#
#one_hot = pd.get_dummies(data['gender'])
#one_hot1= pd.get_dummies(data['Nationality'])
#one_hot2= pd.get_dummies(data['PlaceofBirth'])
#one_hot3= pd.get_dummies(data['StageID'])
#one_hot4= pd.get_dummies(data['GradeID'])
#one_hot5= pd.get_dummies(data['SectionID'])
#one_hot6= pd.get_dummies(data['Topic'])
#one_hot7= pd.get_dummies(data['Semester'])
#one_hot8= pd.get_dummies(data['Relation'])
#one_hot9= pd.get_dummies(data['ParentSurvey'])
#one_hot9_1= pd.get_dummies(data['ParentschoolSatisfaction'])
#one_hot9_2= pd.get_dummies(data['StudentAbsenceDays'])
#one_hot9_3= pd.get_dummies(data['Class'])

df = pd.concat([one_hot,one_hot1,one_hot2,one_hot3,one_hot4,one_hot5,one_hot6,one_hot7,one_hot8,one_hot9,one_hot9_1,one_hot9_2,one_hot9_3], axis=1)

data= data.drop(['gender','Nationality','PlaceofBirth','StageID','GradeID','SectionID','Topic','Semester','Relation','ParentSurvey','ParentschoolSatisfaction','StudentAbsenceDays','Class'],axis = 1)

data = pd.concat([data,df],axis=1,ignore_index=True)

#'gender','NationalITy','PlaceofBirth','StageID','GradeID','SectionID','Topic','Semester','Relation','ParentAnsweringSurvery','ParentschoolSatisfaction','StudentAbsenceDays'])
X = data.iloc[:, 0:72].values
y = data.iloc[:, -3:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

#Creating the classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

confusion_matrix(y_test.values.argmax(axis=1), predictions.argmax(axis=1))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()