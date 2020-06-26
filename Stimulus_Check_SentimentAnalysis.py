#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import GetOldTweets3 as got
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import stopwords
import re, string
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import re # importing regex
import string
nltk.download('stopwords')


# In[2]:


text_query = '#stimuluscheck'
count = 15000
# Creation of query object
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setSince("2020-04-12").setUntil("2020-04-15").setMaxTweets(count)# Creation of list that contains all tweets
tweets = got.manager.TweetManager.getTweets(tweetCriteria)
# Creating list of chosen tweet data
text_tweets = [[tweet.date, tweet.text] for tweet in tweets]
text_tweets


# In[3]:


#storing in dataframe
df = pd.DataFrame(text_tweets)
df.head()
df=df.rename(columns={0: "Date", 1: "Tweet"})
df["Tweet"]


# In[7]:



# Creation of query object
tweetCriteria1 = got.manager.TweetCriteria().setQuerySearch(text_query).setSince("2020-04-20").setUntil("2020-04-23").setMaxTweets(count)# Creation of list that contains all tweets
tweets1 = got.manager.TweetManager.getTweets(tweetCriteria1)
# Creating list of chosen tweet data
text_tweets1 = [[tweet.date, tweet.text] for tweet in tweets1]
text_tweets1


# In[9]:


#storing in dataframe
df1 = pd.DataFrame(text_tweets1)
df1.head()
df1=df1.rename(columns={0: "Date", 1: "Tweet"})
df1["Tweet"]


# In[10]:


# Creation of query object
tweetCriteria2 = got.manager.TweetCriteria().setQuerySearch(text_query).setSince("2020-04-29").setUntil("2020-05-01").setMaxTweets(count)# Creation of list that contains all tweets
tweets2 = got.manager.TweetManager.getTweets(tweetCriteria2)
# Creating list of chosen tweet data
text_tweets2 = [[tweet.date, tweet.text] for tweet in tweets2]
text_tweets2


# In[12]:


#storing in dataframe
df2 = pd.DataFrame(text_tweets2)
df2.head()
df2=df2.rename(columns={0: "Date", 1: "Tweet"})
df2['Tweet']


# In[8]:


#data cleaning
import re # importing regex
import string
def clean_tweet(tweet):
  '''
  Remove unncessary things from the tweet 
  like mentions, hashtags, URL links, punctuations
  '''
  # remove old style retweet text "RT"
  tweet = re.sub(r'^RT[\s]+', '', tweet)
 
  # remove hyperlinks
  tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

  # remove hashtags
  # only removing the hash # sign from the word
  tweet = re.sub(r'#', '', tweet)
  tweet = re.sub(r"stimulus", " ", tweet)
  tweet = re.sub(r"Stimulus", " ", tweet)
  tweet = re.sub(r"stimulus check", " ", tweet)
  tweet = re.sub(r"stimuluscheck", " ", tweet)
  tweet = re.sub(r"check", " ", tweet)
  tweet = re.sub(r"Check", " ", tweet)
  tweet = re.sub(r"money", " ", tweet)
  tweet = re.sub(r"payment", " ", tweet)
  tweet = re.sub(r"Payment", " ", tweet)
  tweet = re.sub(r"COVID19", " ", tweet)
  tweet = re.sub(r"COVID", " ", tweet)
  tweet = re.sub(r"Coronavirus", " ", tweet)
  tweet = re.sub(r"coronavirus", " ", tweet)
  tweet = re.sub(r"bank", " ", tweet)
  tweet = re.sub(r"account", " ", tweet)
  tweet = re.sub(r"people", " ", tweet)
  tweet = re.sub(r"LALATE", " ", tweet)
  tweet = re.sub(r"lalate", " ", tweet)
  tweet = re.sub(r"fuck", " ", tweet)
  tweet = re.sub(r"fucking", " ", tweet)
  tweet = re.sub(r"still", " ", tweet)
  tweet = re.sub(r"will", " ", tweet)
  tweet = re.sub(r"don", " ", tweet)
  tweet = re.sub(r"time", " ", tweet)
  tweet = re.sub(r"getting", " ", tweet)
  tweet = re.sub(r"irs", " ", tweet)
  tweet = re.sub(r"i'm", " ", tweet)
  tweet = re.sub(r"one", " ", tweet)
  # remove mentions
  tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  
 
  # remove punctuations like quote, exclamation sign, etc.
  # we replace them with a space
  tweet = re.sub(r'['+string.punctuation+']+', ' ', tweet)
  return tweet 


# In[15]:


print(df.size)
print(df1.size)
print(df2.size)

df_temp = df
df1_temp = df1
df2_temp = df2


# In[10]:


from textblob import TextBlob, Word, Blobber

def get_tweet_sentiment1(tweet):
    '''
    Get sentiment value of the tweet text
    It can be either positive, negative or neutral
    '''
    sentiment = pd.DataFrame(columns=['Clean_Text','Sentiment'])
    # create TextBlob object of the passed tweet text
    for i in tweet:
        #print(i)
        blob = TextBlob(clean_tweet(i))
        #print(blob)
        # get sentiment
        if blob.sentiment.polarity > 0:
            sentiment = sentiment.append({'Clean_Text':str(blob),'Sentiment':"Positive"},ignore_index=True)
         
        elif blob.sentiment.polarity< 0:
            sentiment = sentiment.append({'Clean_Text':str(blob),'Sentiment':"Negative"},ignore_index=True)
          
        else:
            sentiment = sentiment.append({'Clean_Text':str(blob),'Sentiment':"Neutral"},ignore_index=True)
    #print(sentiment[0])
    return sentiment['Clean_Text'], sentiment['Sentiment']


# In[11]:


df_temp['Clean_Text'],df_temp['Sentiment'] = get_tweet_sentiment1(df_temp['Tweet'])
df1_temp['Clean_Text'],df1_temp['Sentiment'] = get_tweet_sentiment1(df1_temp['Tweet'])
df2_temp['Clean_Text'],df2_temp['Sentiment'] = get_tweet_sentiment1(df2_temp['Tweet'])


# In[12]:


df_temp


# In[13]:


def sentiment_analysis(data):
    positive_count = len(data[data['Sentiment'] == 'Positive'])
    negative_count = len(data[data['Sentiment'] == 'Negative'])
    neutral_count = len(data[data['Sentiment'] == 'Neutral'])
    
    positive_percent = round(100 * positive_count / len(data['Sentiment']),2)
    negative_percent = round(100 * negative_count / len(data['Sentiment']),2)
    neutral_percent  = round(100 * neutral_count  / len(data['Sentiment']),2)

    print ('Postive Tweets  | Count: {} , Percent: {} %' . format(positive_count, positive_percent))
    print ('Negative Tweets | Count: {} , Percent: {} %' . format(negative_count, negative_percent))
    print ('Neutral Tweets  | Count: {} , Percent: {} %' . format(neutral_count, neutral_percent))
    print("\n")


# In[14]:


sentiment_analysis(df_temp)
sentiment_analysis(df1_temp)
sentiment_analysis(df2_temp)


# In[15]:


Positive_tweets1= df_temp[df_temp['Sentiment']=='Positive']
Positive_tweets2= df1_temp[df1_temp['Sentiment']=='Positive']
Positive_tweets3= df2_temp[df2_temp['Sentiment']=='Positive']

Negative_tweets1= df_temp[df_temp['Sentiment']=='Negative']
Negative_tweets2= df1_temp[df1_temp['Sentiment']=='Negative']
Negative_tweets3= df2_temp[df2_temp['Sentiment']=='Negative']


# In[16]:


#plotting word cloud


def plot_wordcloud(words,sentiment):
    comment_words = '' 
    stopwords = set(STOPWORDS) 
# iterate through the csv file 
    for val in words: 
      
    # typecaste each val to string 
        val = str(val) 
  
    # split the value 
        tokens = val.split() 
      
    # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
      
        comment_words += " ".join(tokens)+" "
    if(sentiment == 'positive'):
        wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
    else:
        wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 


# In[30]:



plot_wordcloud(Positive_tweets1['Clean_Text'],"positive")


# In[31]:


plot_wordcloud(Negative_tweets1['Clean_Text'],"negative")


# In[36]:


plot_wordcloud(Negative_tweets2['Clean_Text'],"negative")


# In[38]:


plot_wordcloud(Positive_tweets3['Clean_Text'],"positive")


# In[39]:


plot_wordcloud(Negative_tweets3['Clean_Text'],"negative")


# In[43]:


#plotting word cloud
    
def plot_common_words(text,sentiment):
    stopwords = set(STOPWORDS) 
    total_words=[]
    filtered_sentence=[]
# iterate through the csv file 
    for val in text: 
      
    # typecaste each val to string 
        val = str(val) 
  
    # split the value 
        tokens = val.split() 
      
    # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
        total_words.append(tokens)
    
    for words in total_words: 
        for w in words:
            if w not in stopwords: 
                if len(w)>2:
                    filtered_sentence.append(w) 
    tweet_collection = collections.Counter(filtered_sentence)

    tweet_collection.most_common(25)
    clean_tweets_nsw = pd.DataFrame(tweet_collection.most_common(25),
                             columns=['words', 'count'])

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot horizontal bar graph
    if sentiment== "positive":
        color="lightgreen"
    else:
        color="orange"

    clean_tweets_nsw.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color=color)

    ax.set_title("Common Words Found in Tweets (Without Stop Words)")

    plt.show()


# In[45]:



import collections
First_word= pd.DataFrame()
Second_word= pd.DataFrame()
Third_word= pd.DataFrame()

First_word['Tweet']=Positive_tweets1['Clean_Text'].append(Positive_tweets2['Clean_Text']).append(Positive_tweets3['Clean_Text'])
Second_word['Tweet']=Negative_tweets1['Clean_Text'].append(Negative_tweets2['Clean_Text']).append(Negative_tweets3['Clean_Text'])


plot_common_words(First_word['Tweet'],"positive")
plot_common_words(Second_word['Tweet'],"negative")

