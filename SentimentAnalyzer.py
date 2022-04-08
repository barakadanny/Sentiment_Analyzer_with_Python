import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from sklearn import svm
from sklearn.metrics import classification_report
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

tweetFile = pd.read_csv("Tweets-Data.csv")
dataFrame = pd.DataFrame(tweetFile[['tweet_data']])
tweetData = tweetFile['tweet_data']

tknzr = TweetTokenizer()
stopWords = set(stopwords.words("english"))



cleanedData = []
cleaned = []

for line in tweetData:
    tweet = tknzr.tokenize(str(line))

    for word in tweet:
        if word not in string.punctuation:
            if '@' not in word:
                cleaned.append(word)

    cleanedData.append(cleaned)
    cleaned = []

sentencedData = []

for sentence in cleanedData:
    sentencedData.append(" ".join(sentence))

tweetFile.insert(4, "clean_data", "")

cleanData = tweetFile['clean_data']
i = 0

for row in sentencedData:
    cleanData[i] = sentencedData[i]
    i = i + 1

loopData = [0, 1, 2, 3, 4]
time_linear_train = []
time_linear_predict = []

for loop in loopData:
    t0 = 0
    t1 = 0
    t2 = 0

    tweetDataCopy = tweetFile.copy()

    trainedTweetData = tweetDataCopy.sample(frac=.8, random_state=0)
    testTweetData = tweetDataCopy.drop(trainedTweetData.index)

    sid = SentimentIntensityAnalyzer()
    i = 0
    sentimentData = []

    for sentence in trainedTweetData['clean_data']:
        sentimentData.append(sid.polarity_scores(sentence)['compound'])

    sentimentLabel = []

    for sentiment in sentimentData:
        if sentiment >= 0.05:
            sentimentLabel.append("pos")
        elif sentiment <= -0.05:
            sentimentLabel.append("neg")
        else:
            sentimentLabel.append("neu")

    i = 0
    sentimentTestData = []

    for sentence in testTweetData['clean_data']:
        sentimentTestData.append(sid.polarity_scores(sentence)['compound'])

    sentimentForTestLabel = []

    for sentiment in sentimentTestData:
        if sentiment >= 0.05:
            sentimentForTestLabel.append("pos")
        elif sentiment <= -0.05:
            sentimentForTestLabel.append("neg")
        else:
            sentimentForTestLabel.append("neu")

    data = {'clean_data': testTweetData.clean_data, 'sentiment': sentimentForTestLabel}
    df = pd.DataFrame(data)
    df.to_csv('test-data.csv')

    data = {'clean_data': trainedTweetData.clean_data, 'sentiment': sentimentLabel}
    df = pd.DataFrame(data)
    df.to_csv('train-data.csv')

    testData = pd.read_csv('test-data.csv')
    trainData = pd.read_csv('train-data.csv')

   
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

    train_vectors = vectorizer.fit_transform(trainData['clean_data'].values.astype('U'))
    test_vectors = vectorizer.transform(testData['clean_data'].values.astype('U'))

   
    classifier_linear = svm.SVC(kernel='linear')

    t0 = time.time()

    classifier_linear.fit(train_vectors, trainData['sentiment'])

    t1 = time.time()

    prediction_linear = classifier_linear.predict(test_vectors)

    t2 = time.time()

    time_linear_train.append(t1 - t0)
    time_linear_predict.append(t2 - t1)

    
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train[loop], time_linear_predict[loop]))
    report = classification_report(testData['sentiment'], prediction_linear, output_dict=True)

    print('positive: ', report['pos'])
    print('negative: ', report['neg'])

totalTrainTime = 0
totalPredictTime = 0

for i in loopData:
   
    totalTrainTime = totalTrainTime + time_linear_train[i]
    totalPredictTime = totalPredictTime + time_linear_predict[i]

print("Average training time: %fs" % (totalTrainTime / 5))
print("Average prediction time: %fs" % (totalPredictTime / 5))