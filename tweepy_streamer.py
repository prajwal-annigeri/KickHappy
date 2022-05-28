from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob


import twitter_credentials
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import queries

# Twitter Client


class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets

    def get_hash_tag_tweets(self, num_tweets, hash_tag_list):
        hash_tag_tweets = []
        for tweet in Cursor(self.twitter_client.search, q=hash_tag_list).items(num_tweets):
            hash_tag_tweets.append(tweet)
        return hash_tag_tweets


# Twitter authenticator
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY,
                            twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN,
                              twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class TwitterStreamer():

    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_authenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)

        # Filter Twitter streams to capture data by keywords
        stream.filter(track=hash_tag_list)


# Twitter Stream Listener
class TwitterListener(StreamListener):

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error: %s" % str(e))
        return True

    def on_error(self, status):
        if status == 420:
            return False
        print(status)


class TweetAnalyzer():
    # Analyzing and categorizing content from tweets

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(
            data=[tweet.full_text for tweet in tweets], columns=['tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        # df['len'] = np.array([len(tweet.text) for tweet in tweets])
        # df['date'] = np.array([tweet.created_at for tweet in tweets])
        # df['source'] = np.array([tweet.source for tweet in tweets])
        # df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        # df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df

# Function to get Subjectivity


def getSubjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity

# Function to get Polarity


def getPolarity(twt):
    return TextBlob(twt).sentiment.polarity


def getSentiment(score):
    if score == 0:
        return 'Neutral'
    elif score < 0:
        return 'Negative'
    else:
        return 'Positive'


if __name__ == "__main__":

    # twitter_streamer = TwitterStreamer()
    # twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)

    club = input('Club name:')

    queriesDict = {'manchester united': queries.MAN_UTD, 'arsenal': queries.ARSENAL,
                   'chelsea': queries.CHELSEA, 'manchester city': queries.MAN_CITY, 'spurs': queries.SPURS, 'liverpool': queries.LIVERPOOL}

    if club.lower() in queriesDict.keys():
        club = queriesDict[club.lower()]
    else:
        club = club.lower() + ' owners -filter:retweets'
    # print(club)
    print("Fetching tweets....")

    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    api = twitter_client.get_twitter_client_api()

    # tweets = [tweet for tweet in Cursor(
    #    api.search, q=club, lang='en', tweet_mode='extended', since='2021-01-01').items(500)]

    tweets = [tweet for tweet in Cursor(
        api.search, q=club, lang='en', tweet_mode='extended', since='2021-01-01').items(1000)]

    df = tweet_analyzer.tweets_to_data_frame(tweets)
    df['sentiment'] = np.array(
        [tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])
    df['Subjectivity'] = df['tweets'].apply(getSubjectivity)
    df['Polarity'] = df['tweets'].apply(getPolarity)
    df['Sentiment'] = df['Polarity'].apply(getSentiment)

    print(df.head(10))
    # for i in (df['tweets']):
    # print(i)

    df['sentiment'].value_counts().plot(kind='bar')
    plt.title('Sentiment Analysis Bar Plot')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    for i in range(df.shape[0]):
        plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Purple')
    plt.title('Sentiment Analysis Scatter Plot')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.show()
