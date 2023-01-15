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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits


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

list_of_tags = tagsdf['tag'].unique().tolist()


# ## Users Definiëren

# In[4]:


def User_Favo_Random_Tags(randomTag = random.choices(tagsdf["tag"].to_list(), k=1), K=10):
    #The Tag must have at least K dishes
    while len(recepten_bij_tag(randomTag[0])) < K:
        randomTag = random.choices(tagsdf["tag"].to_list(), k=1)
    
    #Defining tasty dishes
    #print('lengte recipes lijst lekker: ' + str(len(recepten_bij_tag(randomTag[0]))))
    RandomReceptenVoorTag = random.choices(recepten_bij_tag(randomTag[0]), k= K)
    
    #Split Dataset 80%-20%
    Train_Favorieten, Test_Favorieten = RandomReceptenVoorTag[:int(K*0.8)], RandomReceptenVoorTag[int(K*0.8):]
    
    #fill list with recipes for the User
    UserList_Tags = [tags_bij_recept(x) for x in Train_Favorieten]
    

    #Dislike (Or RANDOM THAT DO NOT HAVE TO DO WITH FAVOTAG)
    Niet_Favo_df = list(set(recipesdf["title"].to_list()).difference(recepten_bij_tag(randomTag[0])))
    Niet_Lekker_Recepten_Voor_Tag = random.choices(Niet_Favo_df, k= K)
    
    #Split Dataset 80%-20%
    Train_Favorieten_Niet_Lekker, Test_Favorieten_Niet_Lekker = Niet_Lekker_Recepten_Voor_Tag[:int(K*0.8)], Niet_Lekker_Recepten_Voor_Tag[int(K*0.8):]
    
    #Save Tags for in the matrix
    Train_Favorieten_Niet_Lekker_Tags = [tags_bij_recept(x) for x in Train_Favorieten_Niet_Lekker]
    Test_Favorieten_Niet_Lekker_Tags = [tags_bij_recept(x) for x in Test_Favorieten_Niet_Lekker]

    return Train_Favorieten, UserList_Tags, Train_Favorieten_Niet_Lekker_Tags, Train_Favorieten_Niet_Lekker, randomTag, K, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags


# In[5]:


NumberOfUsers = 500
NumberOfRecipes = 20
UsersList = []
for i in range(NumberOfUsers):
    UserList, UserList_Tags, randomTags, randomRecipes, Usertag, K, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags = User_Favo_Random_Tags(randomTag = random.choices(tagsdf["tag"].to_list(), k=1), K=NumberOfRecipes)
    User = np.array([UserList, UserList_Tags, randomTags, randomRecipes, Usertag, K, Test_Favorieten, Test_Favorieten_Niet_Lekker_Tags])
    #Make list with all users
    UsersList.append(User)

ListSize = len(UsersList)
#Split dataset in Train- (60%), Validate- (20%) & Test (20%) set
TrainUsers, ValidateUsers, TestUsers = UsersList[:int(ListSize*0.6)], UsersList[int(ListSize*0.6): int(ListSize*0.8)], UsersList[int(ListSize*0.8):]


# ## Matrix Functies

# In[6]:


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


# In[7]:


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

        #----- Fill Trainmatrix
        fillInMatrix(matrix, X_train_tags_unique, index_counter, True, False)
        
        #Row 1
        fillInMatrix(matrix2, X_test_tags_unique, index_counter, False, False, columnPrefix = columnPrefix)
        #print(matrix2)
        
        #Row 2
        fillInMatrix(matrix2, randomTags0, index_counter+1, False, False, columnPrefix = columnPrefix)
        
        #Put the value for the random tag as 1 and put y at 0, because it should be false
        fillInMatrix(matrix2, 'y', index_counter, True, True)
        
        #Show which random tag is taken and the one out
        matrix2.loc[index_counter+1, 'Randomgerecht'] = str(randomgerecht0)
        matrix2.loc[index_counter, 'one out'] = X_test
        matrix2.loc[index_counter+1, 'one out'] = X_test
        
        index_counter += 2
    return pd.concat([matrix, matrix2], axis=1)


# In[8]:


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
    
    
    for index_counter in range(len(Test_Gerechten)):
        
        Gerecht_Tags = np.unique(np.concatenate([tags_bij_recept(x) for x in Gerechten]))
        Test_Gerecht_Tags = tags_bij_recept(Test_Gerechten[index_counter])

        #----- Fill Matrix1 with Userlist (1-8)
        fillInMatrix(matrix1, Gerecht_Tags, index_counter, False, False)

        #----- Fill Matrix2 with Test_Favorieten (9-10)
        fillInMatrix(matrix2, Test_Gerecht_Tags, index_counter, False, False, columnPrefix = columnPrefix)
    
    for p in range(len(Test_Gerechten), len(Test_Gerechten) + len(Test_Favorieten_Niet_Lekker_Tags)):
        #----- Fill Matrix1 with Userlist (1-8)
        fillInMatrix(matrix1, Gerecht_Tags, p, False, False)
        
        fillInMatrix(matrix2, Test_Favorieten_Niet_Lekker_Tags, p, False, False, columnPrefix = columnPrefix)
    return pd.concat([matrix1, matrix2], axis=1)


# In[11]:


def createTrainMatrix():
    Matrix = pd.DataFrame()
    for User in TrainUsers:
        UserList, randomTags, randomRecipes = User[0], User[2], User[3]
        TrainMatrix = generateTrainMatrix(UserList, randomTags, randomRecipes)
        Matrix = pd.concat([Matrix, TrainMatrix], axis=0, ignore_index=True)
    
    return Matrix


# In[12]:


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


# In[13]:


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


# In[14]:


TotalTrainMatrix = createTrainMatrix()
TotalValidateMatrix, y_validate = createValidateMatrix()
TotalTestMatrix, y_test = createTestMatrix()


# In[15]:


TotalTrainMatrix


# ## Train

# In[16]:


X_train = TotalTrainMatrix.drop(['y', 'Randomgerecht', 'one out'], axis=1)
y_train = TotalTrainMatrix['y'].to_list()


# ## Validate

# In[22]:


def testClassifiers():
    #---Models
    model_lr = LogisticRegression(max_iter=len(TotalValidateMatrix)+100)
    model_knn = KNeighborsClassifier()
    model_svm = SVC()
    model_rfc = RandomForestClassifier()
    
    parameters_lr = {'C': np.logspace(-5, 8, 15)}
    parameters_knn = {'n_neighbors': list(range(1, 20)),
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

    #print("y_pred: ", y_pred)
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

    #print("y_pred: ", y_pred)
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

    #print("y_pred: ", y_pred)
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

    #print("y_pred: ", y_pred)
    print("Tuned RFC Parameters: {}".format(Grid_rfc.best_params_)) 
    print("Best score is {}".format(Grid_rfc.best_score_))
    print("Confusion Matrix: ", confusion_matrix(y_true = y_validate, y_pred = y_pred))
    print("Accuracy Score: ", accuracy_score(y_validate, y_pred))
    print("Score: ", Grid_rfc.score(TotalValidateMatrix, y_validate))

    print("====================================================================")
    
    return Grid_lr.best_params_, Grid_knn.best_params_, Grid_svm.best_params_, Grid_rfc.best_params_


# In[23]:


Lr_best_params, Knn_best_params, SVM_best_params, RFC_best_params = testClassifiers()


# ## Visualisatie Validation Curve

# In[50]:


#Models
model_lr = LogisticRegression(C = Lr_best_params['C'], max_iter=len(TotalValidateMatrix)+100)
model_knn = KNeighborsClassifier(leaf_size= Knn_best_params['leaf_size'], n_neighbors= Knn_best_params['n_neighbors'], p= Knn_best_params['p'])
model_svm = SVC(C= SVM_best_params['C'], gamma= SVM_best_params['gamma'], probability= True)
model_rfc = RandomForestClassifier(min_samples_leaf = RFC_best_params['min_samples_leaf'], min_samples_split= RFC_best_params['min_samples_split'], min_weight_fraction_leaf= RFC_best_params['min_weight_fraction_leaf'])

# Instantiate the classfiers and make a list
classifiers = [model_lr, model_knn, model_svm, model_rfc]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(TotalTestMatrix)[::,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

#Plot figure
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


# In[70]:


# Setting the range for the parameter (from 1 to 21)
parameter_range = np.arange(1, 21, 1)

# Calculate accuracy on training and test set using the
# gamma parameter with 5-fold cross validation
train_score, test_score = validation_curve(KNeighborsClassifier(leaf_size= Knn_best_params['leaf_size'], p= Knn_best_params['p']), X_train, y_train,
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

