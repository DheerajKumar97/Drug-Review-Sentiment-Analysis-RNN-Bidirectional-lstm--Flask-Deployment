############# Customer Lifetime Value Prediction ################## 
            
             # Exploratory Data Analysis and Data Preprocessing # 

#########################################################################################################
''' Loading Liraries'''
#########################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
#########################################################################################################
''' Loading Data'''
#########################################################################################################

df = pd.read_csv("E:\Downlload\drugsComTest_raw.tsv",error_bad_lines=False, sep='\t')
df
##########################################################################################################
''' DATA PREPROCESSING '''
##########################################################################################################

df.isna().sum()

print("Summary statistics of numerical features : \n", df.describe())

print("=======================================================================")

print("\nTotal number of reviews: ",len(df))

print("=======================================================================")

print("\nTotal number of brands: ", len(list(set(df['drugName']))))

print("=======================================================================")

print("\nTotal number of unique products: ", len(list(set(df['condition']))))

print("=======================================================================")

print("\nPercentage of reviews with neutral sentiment : {:.2f}%".format(df[df['rating']==3]["review"].count()/len(df)*100))

print("=======================================================================")

print("\nPercentage of reviews with positive sentiment : {:.2f}%".format(df[df['rating']>3]["review"].count()/len(df)*100))

print("=======================================================================")

print("\nPercentage of reviews with negative sentiment : {:.2f}%".format(df[df['rating']<3]["review"].count()/len(df)*100))
print("=======================================================================")


import plotly.express as px
fig = px.bar(df['rating'].value_counts().sort_index(), x='rating',title='Distribution of Rating')
fig.show()

drug = df["drugName"].value_counts()
import plotly.express as px
fig = px.bar(drug[:20], x='drugName',title='Number of Reviews for Top 20 Drugs')
fig.show()

conditions = df["condition"].value_counts()
import plotly.express as px
fig = px.bar(conditions[:30], x='condition',title='Number of Reviews for Top 30 conditions')
fig.show()


review_length = df["review"].dropna().map(lambda x: len(x))
review_length = review_length.loc[review_length < 1500]
review_length = pd.DataFrame(review_length)
fig = px.histogram(review_length, x="review")
fig.show()

#########################################################################################################
''' DATA PREPARATION'''
#########################################################################################################

df = df.sample(frac=1, random_state=0) #uncomment to use full set of data

# Drop missing values
df.dropna(inplace=True)

# Encode 4s and 5s as 1 (positive sentiment) and 1s and 2s as 0 (negative sentiment)
df['Sentiment'] = np.where(df['rating'] > 6, 1, 0)
df.head()
#########################################################################################################
''' TRAIN TEST SPLIT'''
########################################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['Sentiment'], \
                                                    test_size=0.1, random_state=0)

print('Load %d training examples and %d validation examples. \n' %(X_train.shape[0],X_test.shape[0]))
print('Show a review in the training set : \n', X_train.iloc[10])
X_train,y_train

########################################################################################################
'''Bag of Words
<br>

**Step 1 : Preprocess raw reviews to cleaned reviews**

**Step 2 : Create BoW using CountVectorizer / Tfidfvectorizer in sklearn**

**Step 3 : Transform review text to numerical representations (feature vectors)**

**Step 4 : Fit feature vectors to supervised learning algorithm (eg. Naive Bayes, Logistic regression, etc.)**

**Step 5 : Improve the model performance by GridSearch**

# Text Preprocessing
<br>

**Step 1 : remove html tags using BeautifulSoup**

**Step 2 : remove non-character such as digits and symbols**

**Step 3 : convert to lower case**

**Step 4 : remove stop words such as "the" and "and" if needed**

**Step 5 : convert to root words by stemming if needed**'''
#########################################################################################################

def cleanData(raw_data, remove_stopwords=False, stemming=False, split_text=False):
    text = BeautifulSoup(raw_data, 'html.parser').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split() 
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        
    if stemming==True:

        stemmer = SnowballStemmer('english') 
        words = [stemmer.stem(w) for w in words]
        
    if split_text==True:
        return (words)
    
    return( " ".join(words))


import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from bs4 import BeautifulSoup 
# import logging
# from wordcloud import WordCloud
# from gensim.models import word2vec
# from gensim.models import Word2Vec
# from gensim.models.keyedvectors import KeyedVectors

X_train_cleaned = []
X_test_cleaned = []

for d in X_train:
    X_train_cleaned.append(cleanData(d))
print('Show a cleaned review in the training set : \n',  X_train_cleaned[10])
    
for d in X_test:
    X_test_cleaned.append(cleanData(d))
    
##########################################################################################################    
'''CountVectorizer with Mulinomial Naive Bayes (Benchmark Model) '''
##########################################################################################################

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
countVect = CountVectorizer() 
X_train_countVect = countVect.fit_transform(X_train_cleaned)
print("Number of features : %d \n" %len(countVect.get_feature_names())) #6378 
print("Show some feature names : \n", countVect.get_feature_names()[::1000])



from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_sample(X_train_countVect,y_train)
from collections import Counter
print("Before smote:",Counter(y_train))
print("After smote:",Counter(y_train_smote))


mnb = MultinomialNB()
mnb.fit(X_train_countVect, y_train)


def modelEvaluation(predictions):
    '''
    Print model evaluation to predicted result 
    '''
    print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("\nAUC score : {:.4f}".format(roc_auc_score(y_test, predictions)))
    print("\nClassification report : \n", metrics.classification_report(y_test, predictions))
    print("\nConfusion Matrix : \n", metrics.confusion_matrix(y_test, predictions))
    
    
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
predictions = mnb.predict(countVect.transform(X_test_cleaned))
modelEvaluation(predictions)  

########################################################################################################
''' # TfidfVectorizer with Logistic Regression'''  
#########################################################################################################
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5) #minimum document frequency of 5
X_train_tfidf = tfidf.fit_transform(X_train)
print("Number of features : %d \n" %len(tfidf.get_feature_names())) #1722
print("Show some feature names : \n", tfidf.get_feature_names()[::1000])

lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)

feature_names = np.array(tfidf.get_feature_names())
sorted_coef_index = lr.coef_[0].argsort()
print('\nTop 10 features with smallest coefficients :\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Top 10 features with largest coefficients : \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

predictions = lr.predict(tfidf.transform(X_test_cleaned))
modelEvaluation(predictions)


from sklearn.model_selection import  GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
estimators = [("tfidf", TfidfVectorizer()), ("lr", LogisticRegression())]
model = Pipeline(estimators)


params = {"lr__C":[0.1, 1, 10], 
          "tfidf__min_df": [1, 3], 
          "tfidf__max_features": [1000, None], 
          "tfidf__ngram_range": [(1,1), (1,2)], 
          "tfidf__stop_words": [None, "english"]} 

grid = GridSearchCV(estimator=model, param_grid=params, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_cleaned, y_train)
print("The best paramenter set is : \n", grid.best_params_)


# Evaluate on the validaton set
predictions = grid.predict(X_test_cleaned)
modelEvaluation(predictions)


import pickle
pickle.dump(mnb,open('Naive_Bayes_model.pkl','wb'))


###############################################################################################################################################################

