#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from IPython.display import display
import random
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression, LogisticRegression
import random
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[2]:


tagsFile = "tags"
recipesFile = "recipes"
nutritionsFile = "nutritions"
ingredientsFile = "ingredients"

tagsdf = pd.read_csv(f"/data/foodboost/{tagsFile}.csv", index_col=0)
recipesdf = pd.read_csv(f"/data/foodboost/{recipesFile}.csv", index_col=0)


# In[3]:


def recepten_bij_tag(tag):
    a = tagsdf.loc[tagsdf['tag'] == tag].recipe.to_list()
    return a
def tags_bij_recept(gerecht):
    b = tagsdf.loc[tagsdf['recipe'] == gerecht].tag.unique()
    return b


# In[4]:


for i in range(10):
    print(random.choices(tagsdf["tag"].to_list()))


# In[5]:


def User_Favo_Random_Tags(randomTag = random.choices(tagsdf["tag"].to_list(), k=1), K=10):
    while len(recepten_bij_tag(randomTag[0])) < K:
        randomTag = random.choices(tagsdf["tag"].to_list(), k=1)
    
    #Lekker
    print('lengte recipes lijst lekker: ' + str(len(recepten_bij_tag(randomTag[0]))))
    RandomReceptenVoorTag = random.choices(recepten_bij_tag(randomTag[0]), k= K)
    Train_Favorieten, Test_Favorieten = RandomReceptenVoorTag[:int(K*0.8)], RandomReceptenVoorTag[int(K*0.8):]
    
    UserList_Tags = [tags_bij_recept(x) for x in Train_Favorieten]
    
    #Niet lekker
    #Niet lekker OFWEL RANDOM DIE NIET MET FAVOTAG TE MAKEN HEBBEN
    #Niet_Favo_df = [x for x in recipesdf["title"] if x not in np.array(recepten_bij_tag(randomTag[0]))]
    Niet_Favo_df = list(set(recipesdf["title"].to_list()).difference(recepten_bij_tag(randomTag[0])))
    Niet_Lekker_Recepten_Voor_Tag = random.choices(Niet_Favo_df, k= K)
    Train_Favorieten_Niet_Lekker, Test_Favorieten_Niet_Lekker = Niet_Lekker_Recepten_Voor_Tag[:int(K*0.8)], Niet_Lekker_Recepten_Voor_Tag[int(K*0.8):]
    
    Train_Favorieten_Niet_Lekker_Tags = [tags_bij_recept(x) for x in Train_Favorieten_Niet_Lekker]
    Test_Favorieten_Niet_Lekker_Tags = [tags_bij_recept(x) for x in Test_Favorieten_Niet_Lekker]
    #RandomRecepten = random.choices(recipesdf["title"].to_list(), k=int(K*0.8))
    
    #Random_Tags = [tags_bij_recept(x) for x in RandomRecepten]


    return Train_Favorieten, UserList_Tags, Train_Favorieten_Niet_Lekker_Tags, Train_Favorieten_Niet_Lekker, randomTag, K, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags


# In[6]:


NumberOfUsers = 500
NumberOfRecipes = 20
UsersList = []
for i in range(NumberOfUsers):
    UserList, UserList_Tags, randomTags, randomRecipes, Usertag, K, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags = User_Favo_Random_Tags(randomTag = random.choices(tagsdf["tag"].to_list(), k=1), K=NumberOfRecipes)
    User = np.array([UserList, UserList_Tags, randomTags, randomRecipes, Usertag, K, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags])
    UsersList.append(User)

ListSize = len(UsersList)
TrainUsers, ValidateUsers, TestUsers = UsersList[:int(ListSize*0.6)], UsersList[int(ListSize*0.6): int(ListSize*0.8)], UsersList[int(ListSize*0.8):]


# In[7]:


list_of_tags = tagsdf['tag'].unique().tolist()


# In[8]:


def fillInMatrix(matrix, column, index_counter, doDoubleRows, isY, columnPrefix = ""):
    size = len(column)
    if(type(column) == str):
        size = 1
    for i in range(size):
        matrix.loc[index_counter, columnPrefix + column[i]] = 1
        if(doDoubleRows):
            if(isY):
                matrix.loc[index_counter+1, columnPrefix + column[i]] = 0
            else:
                matrix.loc[index_counter+1, columnPrefix + column[i]] = 1


# In[9]:


#TRAIN_Matrix
def generateTrainMatrix(UserList, randomTags, randomRecipes):
    
    matrix = pd.DataFrame(columns = list_of_tags)
    
    for i in range(len(UserList*2)):
        matrix.loc[matrix.shape[0]] = 0
    
    columnPrefix = "2-"
    matrix2 = matrix.copy()
    matrix2.columns = [columnPrefix + columnName for columnName in matrix2.columns]
    
    X = np.array(UserList)
    loo = LeaveOneOut()
    index_counter = 0

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]

        randomTags0 = randomTags.pop(0)
        randomgerecht0 = randomRecipes.pop(0)

        X_train_tags_unique = np.unique(np.concatenate([tags_bij_recept(x) for x in X_train]))
        X_test_tags_unique = np.unique(np.concatenate([tags_bij_recept(x) for x in X_test]))

        #----- Matrix met train vullen
        fillInMatrix(matrix, X_train_tags_unique, index_counter, True, False)
        
        #Rij 1
        fillInMatrix(matrix2, X_test_tags_unique, index_counter, False, False, columnPrefix = columnPrefix)
        #print(matrix2)
        
        #Rij 2
        fillInMatrix(matrix2, randomTags0, index_counter+1, False, False, columnPrefix = columnPrefix)
        
        #Put the value for the random tag as 1 and put y at 0, because it should be false
        fillInMatrix(matrix2, 'y', index_counter, True, True)
        
        #Show which random tag is taken and the one out
        matrix2.loc[index_counter+1, 'Randomgerecht'] = str(randomgerecht0)
        matrix2.loc[index_counter, 'one out'] = X_test
        matrix2.loc[index_counter+1, 'one out'] = X_test
        
        index_counter += 2
    return pd.concat([matrix, matrix2], axis=1)


# In[10]:


#TEST_Matrix
def generateTestMatrix(UserList, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags):
    matrix1 = pd.DataFrame(columns = list_of_tags)
    Gerechten = np.array(UserList)
    Test_Gerechten = np.array(Test_Favorieten)
    
    for i in range(len(Test_Gerechten) + len(Test_Favorieten_Niet_Lekker_Tags)):
        matrix1.loc[matrix1.shape[0]] = 0
    
    matrix2 = matrix1.copy()
    columnPrefix = "2-"
    matrix2.columns = [columnPrefix + columnName for columnName in matrix2.columns]
    
    #-----
    for index_counter in range(len(Test_Gerechten)):
        
        Gerecht_Tags = np.unique(np.concatenate([tags_bij_recept(x) for x in Gerechten]))
        Test_Gerecht_Tags = tags_bij_recept(Test_Gerechten[index_counter])

        #----- Matrix1 met Userlist (1-8) invullen
        fillInMatrix(matrix1, Gerecht_Tags, index_counter, False, False)

        #----- Matrix2 met Test_Favorieten (9-10) vullen
        fillInMatrix(matrix2, Test_Gerecht_Tags, index_counter, False, False, columnPrefix = columnPrefix)
    
    for p in range(len(Test_Gerechten), len(Test_Gerechten) + len(Test_Favorieten_Niet_Lekker_Tags)):
        #----- Matrix1 met Userlist (1-8) invullen
        fillInMatrix(matrix1, Gerecht_Tags, p, False, False)
        
        fillInMatrix(matrix2, Test_Favorieten_Niet_Lekker_Tags, p, False, False, columnPrefix = columnPrefix)
    return pd.concat([matrix1, matrix2], axis=1)


# In[11]:


#--Dubbele Matrix Compleet
def testTrainMatrix():
    UserList, UserList_Tags, randomTags, randomRecipes, Usertag, K, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags = User_Favo_Random_Tags(K=NumberOfRecipes)
    VolledigeMatrix = generateTrainMatrix(UserList, randomTags, randomRecipes)
    return VolledigeMatrix


# In[12]:


testTrainMatrix()


# In[13]:


def createTrainMatrix():
    Matrix = pd.DataFrame()
    for User in TrainUsers:
        UserList, randomTags, randomRecipes = User[0], User[2], User[3]
        TrainMatrix = generateTrainMatrix(UserList, randomTags, randomRecipes)
        Matrix = pd.concat([Matrix, TrainMatrix], axis=0, ignore_index=True)
    
    return Matrix


# In[14]:


def createValidateMatrix():
    Matrix = pd.DataFrame()
    y_validate = []
    for User in ValidateUsers:
        UserList, randomTags, randomRecipes, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags = User[0], User[2], User[3], User[6], User[7]
        TestMatrix = generateTestMatrix(UserList, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags)
        Matrix = pd.concat([Matrix, TestMatrix], axis=0, ignore_index=True)
        
        for p in Test_Favorieten:
            y_validate.append(1)
        for t in Test_Favorieten_Niet_Lekker_Tags:
            y_validate.append(0)
    return Matrix, y_validate


# In[15]:


def createTestMatrix():
    Matrix = pd.DataFrame()
    y_test = []
    for User in TestUsers:
        UserList, randomTags, randomRecipes, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags = User[0], User[2], User[3], User[6], User[7]
        TestMatrix = generateTestMatrix(UserList, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags)
        Matrix = pd.concat([Matrix, TestMatrix], axis=0, ignore_index=True)
        
        for p in Test_Favorieten:
            y_test.append(1)
        for t in Test_Favorieten_Niet_Lekker_Tags:
            y_test.append(0)
    return Matrix, y_test


# In[16]:


# for i in range(NumberOfUsers):
#     UserList, UserList_Tags, randomTags, randomRecipes, Usertag, K, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags = User_Favo_Random_Tags(randomTag = random.choices(tagsdf["tag"].to_list(), k=1), randomTagNietLekker = random.choices(tagsdf["tag"].to_list(), k=1), K=NumberOfRecipes)
#     TrainMatrix = generateTrainMatrix(UserList, randomTags, randomRecipes)
#     TestMatrix = generateTestMatrix(UserList, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags)
#     TotalTrainMatrix = pd.concat([TotalTrainMatrix, TrainMatrix], axis=0, ignore_index=True)
#     TotalTestMatrix = pd.concat([TotalTestMatrix, TestMatrix], axis=0, ignore_index=True)
#     for p in Test_Favorieten:
#         y_test.append(1)
#     for t in Test_Favorieten_Niet_Lekker_Tags:
#         y_test.append(0)


# In[17]:


TotalTrainMatrix = createTrainMatrix()
TotalValidateMatrix, y_validate = createValidateMatrix()
TotalTestMatrix, y_test = createTestMatrix()


# In[18]:


TotalTrainMatrix


# In[19]:


#--Dubbele Matrix
X_train = TotalTrainMatrix.drop(['y', 'Randomgerecht', 'one out'], axis=1)
y_train = TotalTrainMatrix['y'].to_list()


# In[20]:


TotalValidateMatrix_Values = TotalValidateMatrix.loc[:, (TotalValidateMatrix != 0).any(axis=0)]
#TotalValidateMatrix_Values
TotalValidateMatrix


# In[21]:


y_train


# In[22]:


AmountOfRows = X_train[X_train.columns[0]].count()
print(AmountOfRows)


# In[23]:


def RFC():
    model_rfc = RandomForestClassifier(min_samples_split=16, min_samples_leaf=4, min_weight_fraction_leaf=0.05)
    model_rfc.fit(X_train, y_train)
    y_pred = model_rfc.predict(TotalValidateMatrix)
    print(recall_score(y_validate, y_pred))
    print(confusion_matrix(y_true = y_validate, y_pred = y_pred))
    print(accuracy_score(y_validate, y_pred))


# In[24]:


RFC()


# In[25]:


def testClassifiers():
    #---Modellen
    model_lr = LogisticRegression(max_iter=AmountOfRows+100)
    model_knn = KNeighborsClassifier()
    model_svm = SVC()
    model_rfc = RandomForestClassifier()
    
    parameters_lr = {'C': np.logspace(-5, 8, 15)}
    parameters_knn = {'n_neighbors': list(range(1, 10)),
                     'leaf_size' : list(range(1, 10)), 
                      'p':[1,2]}
    parameters_svm = {'C': [0.1, 1, 10, 100],  
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                    'gamma':['scale', 'auto']}
                    #'kernel': ['linear']}
    parameters_rfc = {'min_samples_split': [0,5,10,20],
                      'min_samples_leaf': [2,4,6,8,10],
                      'min_weight_fraction_leaf': [0.01, 0.05, 0.1]}
    
    #---Logistic Regression
    print("\nLogistic Regression")
    Grid_lr = GridSearchCV(model_lr, parameters_lr, cv=5)
    Grid_lr.fit(X_train, y_train)

    y_pred = Grid_lr.predict(TotalValidateMatrix)

    print("y_pred: ", y_pred)
    print("Tuned Logistic Regression Parameters: {}".format(Grid_lr.best_params_)) 
    print("Best score is {}".format(Grid_lr.best_score_))
    print("Confusion Matrix: ", confusion_matrix(y_true = y_validate, y_pred = y_pred))
    #print("Classification Report: ", classification_report(y_validate, y_pred))
    print("Accuracy Score: ", accuracy_score(y_validate, y_pred))
    print("Score: ", Grid_lr.score(TotalValidateMatrix, y_validate))

#---K_NearestNeighbors
    print("\nK_NearestNeighbors")
    Grid_knn = GridSearchCV(model_knn, parameters_knn, cv=5)
    Grid_knn.fit(X_train, y_train)

    y_pred = Grid_knn.predict(TotalValidateMatrix)

    print("y_pred: ", y_pred)
    print("Tuned K_NearestNeighbors Parameters: {}".format(Grid_knn.best_params_)) 
    print("Best score is {}".format(Grid_knn.best_score_))
    print("Confusion Matrix: ", confusion_matrix(y_true = y_validate, y_pred = y_pred))
    print("Accuracy Score: ", accuracy_score(y_validate, y_pred))
    print("Score: ", Grid_knn.score(TotalValidateMatrix, y_validate))

#---SVC
    print("\nSVM")
    Grid_svm = GridSearchCV(model_svm, parameters_svm, cv=5)
    Grid_svm.fit(X_train, y_train)

    y_pred = Grid_svm.predict(TotalValidateMatrix)

    print("y_pred: ", y_pred)
    print("Tuned SVM Parameters: {}".format(Grid_svm.best_params_)) 
    print("Best score is {}".format(Grid_svm.best_score_))
    print("Confusion Matrix: ", confusion_matrix(y_true = y_validate, y_pred = y_pred))
    print("Accuracy Score: ", accuracy_score(y_validate, y_pred))
    print("Score: ", Grid_svm.score(TotalValidateMatrix, y_validate))
    
#---RFC
    print("\nRFC")
    Grid_rfc = GridSearchCV(model_rfc, parameters_rfc, cv=5)
    Grid_rfc.fit(X_train, y_train)

    y_pred = Grid_rfc.predict(TotalValidateMatrix)

    print("y_pred: ", y_pred)
    print("Tuned RFC Parameters: {}".format(Grid_rfc.best_params_)) 
    print("Best score is {}".format(Grid_rfc.best_score_))
    print("Confusion Matrix: ", confusion_matrix(y_true = y_validate, y_pred = y_pred))
    print("Accuracy Score: ", accuracy_score(y_validate, y_pred))
    print("Score: ", Grid_rfc.score(TotalValidateMatrix, y_validate))

    print("====================================================================")


# In[26]:


testClassifiers()


# In[27]:


# parameters = 10
# for i in range(1, parameters):
#     print("i ", i)
#     model = KNeighborsClassifier(n_neighbors = i)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(TotalValidateMatrix)
#     #print(y_pred)
#     print(recall_score(y_validate, y_pred))
#     print(confusion_matrix(y_true = y_validate, y_pred = y_pred))
#     print(accuracy_score(y_validate, y_pred))
#     print("====================")


# In[29]:


testClassifier = KNeighborsClassifier(leaf_size= 1, n_neighbors= 7, p= 1)
testClassifier.fit(X_train, y_train)
testYpred = testClassifier.predict(TotalTestMatrix)

print("y_pred: ", testYpred)
print("Confusion Matrix: ", confusion_matrix(y_true = y_test, y_pred = testYpred))
#print("Classification Report: ", classification_report(y_test, testYpred))
print("Accuracy Score: ", accuracy_score(y_test, testYpred))


# In[32]:


# Import Required libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve


# Setting the range for the parameter (from 1 to 10)
parameter_range = np.arange(5, 8, 1)

# Calculate accuracy on training and test set using the
# gamma parameter with 5-fold cross validation
train_score, test_score = validation_curve(KNeighborsClassifier(leaf_size= 1, p= 1), X_train, y_train,
									param_name = "n_neighbors",
									param_range = parameter_range,
										cv = 5, scoring = "accuracy")

# Calculating mean and standard deviation of training score
mean_train_score = np.mean(train_score, axis = 1)
std_train_score = np.std(train_score, axis = 1)

# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis = 1)
std_test_score = np.std(test_score, axis = 1)

# Plot mean accuracy scores for training and testing scores
plt.plot(parameter_range, mean_train_score,
	label = "Training Score", color = 'b')
plt.plot(parameter_range, mean_test_score,
label = "Test score", color = 'g')

# Creating the plot
plt.title("Validation Curve with KNN Classifier")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()


# In[30]:


# testClassifier = SVC(C= 100, gamma= 'auto')
# testClassifier.fit(X_train, y_train)
# testYpred = testClassifier.predict(TotalTestMatrix)

# print("y_pred: ", testYpred)
# print("Confusion Matrix: ", confusion_matrix(y_true = y_test, y_pred = testYpred))
# #print("Classification Report: ", classification_report(y_test, testYpred))
# print("Accuracy Score: ", accuracy_score(y_test, testYpred))


# In[ ]:




