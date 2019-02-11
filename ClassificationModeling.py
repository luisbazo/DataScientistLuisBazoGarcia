#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'tk')
import pandas as pd
from sklearn import preprocessing as scale
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,recall_score, confusion_matrix, roc_curve, average_precision_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score as AUC


# In[23]:

#loading csv file into python
dataset = pd.read_csv('./device_failure.csv',encoding='unicode_escape')


# In[3]:


dataset.head(10)


# In[4]:


#dataset.isnull().sum()
data = dataset.dropna(axis=0)


# In[5]:


print (dataset.dtypes)


# In[6]:


data.columns.shape
data['failure'].value_counts()


# In[7]:


data.columns.shape
data['failure'].value_counts()
#the data is highly imbalanced with failure rate less than 0.09 %


# In[8]:


data.describe()


# In[9]:


#the installation date does not have much to offer. Its just a daily datetime value for dynamic data
#adding a day row representing age (in days) from inital date of installation
#assumption that first reading for a device is the installation date
data.sort_values(['device','date'],inplace=True)
data['date'] = pd.to_datetime(data['date'])
data['Days'] = data.groupby('device')['date'].rank(method='dense')

print (data.head(10))

# In[43]:


#initial data analysis
data.columns.shape
data['failure'].value_counts()
#data.groupby(['attribute7','attribute8'])['attribute7'].count()
#attributes 7 and 8 have same values so one of them can be discarded


# In[10]:


#Failure rate Vs days from installation
Failure = data[data.failure==1]

#Failure.Days.value_counts().plot(kind='barh')
plt.plot(Failure.Days,Failure.failure,'o')
plt.title("Distribution of Failure Rate as per age(days) from installation")
plt.ylabel("Failure")
plt.xlabel("Age(No. of days since installation)")
plt.show()
#highly unlikely that age has any relation with failure rate


# In[11]:


Corr = data[data.columns].corr()
sns.heatmap(Corr,annot=True)
#attrute 9 and 3 seem to have a good co-relation though not high enough


# In[12]:


#fig = plt.figure(figsize = (14,5))
#plt.plot(data.attribute9,data.attribute3,'o')
#plt.title("Relationship between attributes")
#plt.xlabel('X (Attribute9)')
#plt.ylabel('Y (Attribute3)')
#plt.show()


# In[13]:


#fig = plt.figure(figsize = (15,5))
#plt.subplot2grid((1,2),(0,0))
#Failure vs Attribute1
#Failure.attribute3.value_counts().plot(kind='barh')
#plt.title("Distribution of Failure according to attribute1")
#plt.xlabel("Failures")
#plt.ylabel("Attribute3")

#plt.subplot2grid((1,2),(0,1))
#Failure.attribute7.value_counts().plot(kind='barh')
#plt.title("Distribution of Failure according to attribute2")
#plt.xlabel("Failure")
#plt.ylabel("Attribute9")


# In[14]:


#fig = plt.figure(figsize = (15,5))
#plt.subplot2grid((2,2),(0,0))
#Failure.attribute4.value_counts().plot(kind='barh')
#plt.title("Distribution of Failure according to attribute4")
#plt.xlabel("Failure")
#plt.ylabel("Attribute3")

#plt.subplot2grid((2,2),(0,1))
#Failure.attribute9.value_counts().plot(kind='barh')
#plt.title("Distribution of Failure according to attribute9")
#plt.xlabel("Failure")
#plt.ylabel("Attribute3")


# In[15]:


data.describe()


# In[16]:


#since data is highly imbalanced we would be downsampling data.
#resampling data

df_nonfailure = data[data['failure'] == 0]
df_failure = data[data['failure']==1]
df_nonfailure_downsample = resample(df_nonfailure,replace=False,n_samples = 106,random_state=23)
df_resampled = pd.concat([df_nonfailure_downsample,df_failure])

data_Outcome = df_resampled['failure']
#scaling cat and cont data

#dropping unwanted columns
df_resampled = df_resampled.drop(['failure','date','device','attribute8'],axis = 1)
standard_sc = scale.StandardScaler()
x_std = standard_sc.fit_transform(df_resampled)
data_scaled = pd.DataFrame(x_std)


# In[17]:


data_scaled.head()


# In[18]:


#split data into test and train
xtrain,xtest,ytrain,ytest = train_test_split(data_scaled,data_Outcome,test_size=0.25,random_state =19)


# In[30]:


#defining a metrics function to evaluate a model
def Metrics(ytest,pred):
    print('accuray:', accuracy_score(ytest,pred),',recall score:',recall_score(ytest,pred),'\n ConfusionMatrix: \n',confusion_matrix(ytest,pred))
    #model_rf.feature_importances_
    average_precision = average_precision_score(ytest,pred)
    print('average_precision_score: ',average_precision_score(ytest,pred))
    print('Precision Score:',precision_score(ytest,pred_rf),'F1_score:',f1_score(ytest,pred_rf))

    precision, recall,_ = precision_recall_curve(ytest,pred)
    plt.step(recall,precision, color='b',alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(              average_precision))
    #plt.title(algo)


# In[31]:


#decision tree classifier
DT = DecisionTreeClassifier(random_state=12)
model_dt = DT.fit(xtrain,ytrain)
pred_dt=model_dt.predict(xtest)
Metrics(ytest,pred_dt)
#print('******decision tree*****')
#print('accuray:', accuracy_score(ytest,pred_dt),',recall score:',recall_score(ytest,pred_dt))
#print(confusion_matrix(ytest,pred_dt))
#print(model_dt.feature_importances_ )


# In[28]:


roc_auc_score(ytest,pred_dt)


# In[32]:


#random forest
#rf = RandomForestClassifier(max_depth = 2,random_state=1)
rf = RandomForestClassifier(n_estimators=25, min_samples_split=25,                             max_depth=5,random_state=72)
model_rf = rf.fit(xtrain,ytrain)
pred_rf = model_rf.predict(xtest)
Metrics(ytest,pred_rf)
#print('******random forest*****')
#print('accuray:', accuracy_score(ytest,pred_rf),',recall score:',recall_score(ytest,pred_rf))
#print(confusion_matrix(ytest,pred_rf))
#print(model_rf.feature_importances_)
#print('Precision Score:',precision_score(ytest,pred_rf),'F1_score:',f1_score(ytest,pred_rf))


# In[34]:


#gaussian naive bayes
gnb = GaussianNB()
modelgnb = gnb.fit(xtrain,ytrain)
pred_gnb = modelgnb.predict(xtest)
Metrics(ytest,pred_gnb)


# In[35]:


#svm
modelsvc = SVC(kernel='linear')
modelsvc.fit(xtrain,ytrain)
pred_svm = modelsvc.predict(xtest)
Metrics(ytest,pred_svm)


# In[42]:


def model_comparison(X_train,X_test, y_train,y_test):
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier(n_estimators=100)

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),                      (gnb, 'Naive Bayes'),                      (svc, 'Support Vector Classification'),                      (rfc, 'Random Forest')]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos =                 (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value =             calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name, ))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()


# In[43]:


model_comparison(xtrain,xtest,ytrain,ytest)

# In[ ]:


#Decision tree and random forest are perhabs giving the best results among all.
#Although I think models can be improved if provided with more failure data


def corr_df(x, corr_val):
    '''
    Obj: Drops features that are strongly correlated to other features.
          This lowers model complexity, and aids in generalizing the model.
    Inputs:
          df: features df (x)
          corr_val: Columns are dropped relative to the corr_val input (e.g. 0.8)
    Output: df that only includes uncorrelated features
    '''

    # Creates Correlation Matrix and Instantiates
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterates through Correlation Matrix Table to find correlated columns
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = item.values
            if val >= corr_val:
                # Prints the correlated feature set and the corr val
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(i)

    drops = sorted(set(drop_cols))[::-1]

    # Drops the correlated columns
    for i in drops:
        col = x.iloc[:, (i+1):(i+2)].columns.values
        df = x.drop(col, axis=1)

    return df
