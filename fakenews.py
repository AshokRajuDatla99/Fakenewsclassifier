import pandas as pd
from sklearn.utils import shuffle
import string
from nltk.corpus import stopwords
from nltk import tokenize
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import itertools
import numpy as np

# stop_words imported from nltk
stop = stopwords.words('english')

def clean_text(df,col_name):
    
    # convert the text to lowercase
    df[col_name] = df[col_name].apply(lambda x: x.lower())

    # remove punctuation
    df[col_name] = df[col_name].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))

    # remove stopwords
    df[col_name] = df[col_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        
    return df

def frequent_tokens(df, col_name, graph_size):
    token_space = tokenize.WhitespaceTokenizer()
    all_words = ' '.join([text for text in df[col_name]])
    token_phrase = token_space.tokenize(all_words)
    frequency = FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),"Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = graph_size)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()

def data_exploration(data):

    print(data.isna().sum())
    # plotting the number of nan values in the dataframe
    # data.isna().sum().plot(kind="bar")
    # plt.show()

    print(data.groupby(['subject'])['text'].count())
    # plotting the subject types and their sample size
    # data.groupby(['subject'])['text'].count().plot(kind="bar")
    # plt.show()

    print(data.groupby(['target'])['text'].count())
    # plotting the number of samples in both fake and real classes
    # data.groupby(['target'])['text'].count().plot(kind="bar")
    # plt.show()

    # plotting the top 20 frequent words of the fake samples
    frequent_tokens(data[data['target'] == 'fake'], 'text', 20)

    # plotting the top 20 frequent words of the true samples
    frequent_tokens(data[data['target'] == 'true'], 'text', 20)


def plot_confusion_matrix(cm, classes):

    plt.imshow(cm,interpolation='nearest', cmap= plt.cm.Blues)
    plt.title(" Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],ha="center", va="bottom", color="black" )
    plt.tight_layout()
    plt.show()

def logistic_regression(x_train,y_train,x_test,y_test):

    # Vectorizing and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression())])

    # Fitting the model
    model = pipe.fit(x_train, y_train)

    # Accuracy
    prediction = model.predict(x_test)
    print("accuracy: {}%".format(round(metrics.accuracy_score(y_test, prediction) * 100, 2)))

    # confusion matrix
    cm = metrics.confusion_matrix(y_test, prediction)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'])

def decisiontree_classifier(x_train,y_train,x_test,y_test):

    # Vectorizing and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('model', DecisionTreeClassifier(criterion='entropy',max_depth=20,splitter='best'))])

    # Fitting the model
    model = pipe.fit(x_train, y_train)

    # Accuracy
    prediction = model.predict(x_test)
    print("accuracy: {}%".format(round(metrics.accuracy_score(y_test, prediction) * 100, 2)))

    # confusion matrix
    cm = metrics.confusion_matrix(y_test, prediction)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'])


def Main():

    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake['target'] = 'fake'
    true['target'] = 'true'

    data = pd.concat([fake, true]).reset_index(drop=True)

    # arranging the index to order
    data.drop(["date"],axis=1,inplace=True)
    data.drop(["title"],axis=1,inplace=True)

    # shuffle the data to avoid bias
    data = shuffle(data)
    data = data.reset_index(drop=True)

    # clean the text: lower,stop_words and punctuation
    data = clean_text(data,'text')

    # explore the data
    # data_exploration(data)

    x_train, x_test, y_train, y_test = train_test_split(data['text'], data.target, test_size=0.2)

    # logistic_regression(x_train,y_train,x_test,y_test)

    decisiontree_classifier(x_train,y_train,x_test,y_test)

if __name__ == "__main__":
    Main()



