#!/usr/bin/env python
# coding: utf-8

# # Getting data

# In[1]:


import requests
from bs4 import BeautifulSoup
import pickle

def url_to_transcript(url):
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.findAll('p')]
    print(url)
    return text

# URLs
urls = [ 'https://scrapsfromtheloft.com/2021/05/20/tig-notaro-boyish-girl-interrupted-transcript/',
        'https://scrapsfromtheloft.com/2021/04/18/joe-list-i-hate-myself-transcript/',
        'https://scrapsfromtheloft.com/2021/03/27/nate-bargatze-greatest-average-american-transcript/',
        'https://scrapsfromtheloft.com/2021/02/27/brian-regan-on-the-rocks-transcript/',
        'https://scrapsfromtheloft.com/2021/02/18/george-carlin-politically-correct-language/',
        'https://scrapsfromtheloft.com/2021/02/18/doug-stanhope-beer-hall-putsch-transcript/',
        'https://scrapsfromtheloft.com/2021/01/15/chris-rock-total-blackout-the-tamborine-extended-cut-transcript/',
        'https://scrapsfromtheloft.com/2020/12/22/sarah-cooper-everythings-fine-transcript/',
        'https://scrapsfromtheloft.com/2020/12/20/bo-burnham-words-words-words-transcript/',
        'https://scrapsfromtheloft.com/2020/12/17/vir-das-outside-in-the-lockdown-special-transcript/',
        'https://scrapsfromtheloft.com/2020/11/27/larry-the-cable-guy-remain-seated-transcript/',
        'https://scrapsfromtheloft.com/2020/11/24/craig-ferguson-just-being-honest-transcript/',
        'https://scrapsfromtheloft.com/2020/11/12/sam-morril-i-got-this-2020-transcript/',
        'https://scrapsfromtheloft.com/2020/11/08/dave-chappelle-snl-monologue-2020-transcript/',
        'https://scrapsfromtheloft.com/2020/11/06/chris-rock-snl-monologue-2020-transcript/',
        'https://scrapsfromtheloft.com/2020/11/05/bill-burr-snl-monologue-2020-transcript/',
        'https://scrapsfromtheloft.com/2020/11/05/john-mulaney-snl-monologue-2020-transcript/',
        'https://scrapsfromtheloft.com/2020/10/10/ronny-chieng-asian-comedian-destroys-america-transcript/',
        'https://scrapsfromtheloft.com/2020/09/24/craig-ferguson-a-wee-bit-o-revolution-transcript/',
        'https://scrapsfromtheloft.com/2020/09/17/michael-mcintyre-showman-transcript/',
        'https://scrapsfromtheloft.com/2020/08/22/rob-schneider-asian-momma-mexican-kids-transcript/',
        'https://scrapsfromtheloft.com/2020/08/22/sam-jay-3-in-the-morning-transcript/',
        'https://scrapsfromtheloft.com/2020/07/26/jack-whitehall-im-only-joking-transcript/',
        'https://scrapsfromtheloft.com/2020/07/19/urzila-carlson-overqualified-loser-transcript/',
        'https://scrapsfromtheloft.com/2020/07/14/george-lopez-well-do-it-for-half-transcript/',
        'https://scrapsfromtheloft.com/2020/07/08/jim-jefferies-intolerant-transcript/',
        'https://scrapsfromtheloft.com/2020/07/05/bill-hicks-censored-david-letterman-transcript/',
        'https://scrapsfromtheloft.com/2020/06/27/george-carlin-doin-it-againtranscript/',
        'https://scrapsfromtheloft.com/2020/06/25/eric-andre-legalize-everything-transcript/',
        'https://scrapsfromtheloft.com/2020/06/22/roy-wood-jr-father-figure-transcript/',
        'https://scrapsfromtheloft.com/2020/06/19/dave-chappelle-hbo-comedy-half-hour-1998-transcript/',
        'https://scrapsfromtheloft.com/2020/06/13/mark-normand-dont-be-yourself-transcript/',
        'https://scrapsfromtheloft.com/2020/06/09/doug-stanhope-fear-of-an-empty-bed-transcript/',
        'https://scrapsfromtheloft.com/2020/06/02/chris-gethard-career-suicide-transcript/',
        'https://scrapsfromtheloft.com/2020/06/01/ramy-youssef-feelings-transcript/',
        'https://scrapsfromtheloft.com/2020/06/01/kenny-sebastian-dont-be-that-guy-transcript/',
        'https://scrapsfromtheloft.com/2020/05/27/billy-connolly-high-horse-tour-live-transcript/',
        'https://scrapsfromtheloft.com/2020/05/26/hannah-gadsby-douglas-transcript/',
        'https://scrapsfromtheloft.com/2020/05/21/hasan-minhaj-homecoming-king-transcript/',
        'https://scrapsfromtheloft.com/2020/05/20/patton-oswalt-i-love-everything-transcript/',
        'https://scrapsfromtheloft.com/2020/05/10/russell-peters-deported-transcript/',
        'https://scrapsfromtheloft.com/2020/05/10/jimmy-o-yang-good-deal-transcript/',
        'https://scrapsfromtheloft.com/2020/05/09/jo-koy-lights-out-2012-full-transcript/',
        'https://scrapsfromtheloft.com/2020/05/08/lee-mack-going-out-live-transcript/',
        'https://scrapsfromtheloft.com/2020/05/07/lee-mack-live-transcript/',
        'https://scrapsfromtheloft.com/2020/05/06/t-j-miller-no-real-reason-transcript/',
        'https://scrapsfromtheloft.com/2020/05/06/jerry-seinfeld-23-hours-to-kill-transcript/',
        'https://scrapsfromtheloft.com/2020/05/05/bill-burr-late-show-with-david-letterman-2010/',
        'https://scrapsfromtheloft.com/2020/05/02/sincerely-louis-ck-transcript/',
        'https://scrapsfromtheloft.com/2020/03/04/pete-davidson-smd-transcript/']

titles = ['TIG NOTARO: BOYISH GIRL INTERRUPTED', 'JOE LIST: I HATE MYSELF', 'NATE BARGATZE: THE GREATEST AVERAGE AMERICAN',
         'BRIAN REGAN: ON THE ROCKS', 'GEORGE CARLIN: POLITICALLY CORRECT LANGUAGE', 'DOUG STANHOPE: BEER HALL PUTSCH',
         'CHRIS ROCK TOTAL BLACKOUT: THE TAMBORINE EXTENDED CUT ', 'SARAH COOPER: EVERYTHING’S FINE', 'BO BURNHAM: WORDS, WORDS, WORDS',
          'VIR DAS: OUTSIDE IN – THE LOCKDOWN SPECIAL', 'LARRY THE CABLE GUY – REMAIN SEATED', 'CRAIG FERGUSON: JUST BEING HONEST',
         'SAM MORRIL: I GOT THIS', 'DAVE CHAPPELLE: SNL MONOLOGUE', 'CHRIS ROCK: SNL MONOLOGUE', 'BILL BURR: SNL MONOLOGUE',
         'JOHN MULANEY: SNL MONOLOGUE', 'RONNY CHIENG: ASIAN COMEDIAN DESTROYS AMERICA ', 'CRAIG FERGUSON: A WEE BIT O’ REVOLUTION',
         'MICHAEL MCINTYRE: SHOWMAN', 'ROB SCHNEIDER: ASIAN MOMMA, MEXICAN KIDS', 'SAM JAY: 3 IN THE MORNING', 'JACK WHITEHALL: I AM ONLY JOKING',
         'URZILA CARLSON: OVERQUALIFIED LOSER', 'GEORGE LOPEZ: WE’LL DO IT FOR HALF', 'JIM JEFFERIES: INTOLERANT',
         'BILL HICKS: THE CENSORED SEVEN MINUTES ON LATE SHOW WITH DAVID LETTERMAN', 'GEORGE CARLIN: DOIN IT AGAIN',
         'ERIC ANDRE: LEGALIZE EVERYTHING', 'ROY WOOD JR.: FATHER FIGURE', 'DAVE CHAPPELLE: HBO COMEDY HALF-HOUR',
          'MARK NORMAND: DON’T BE YOURSELF', 'DOUG STANHOPE: FEAR OF AN EMPTY BED', 'CHRIS GETHARD: CAREER SUICIDE ',
          'RAMY YOUSSEF: FEELINGS', 'KENNY SEBASTIAN DON’T BE THAT GUY', 'BILLY CONNOLLY: HIGH HORSE TOUR LIVE',
          'HANNAH GADSBY: DOUGLAS', 'HASAN MINHAJ: HOMECOMING KING', 'PATTON OSWALT: I LOVE EVERYTHING',
          'RUSSELL PETERS: DEPORTED', 'JIMMY O. YANG: GOOD DEAL', 'JO KOY: LIGHTS OUT', 'LEE MACK: GOING OUT LIVE',
          'LEE MACK: LIVE', 'T.J. MILLER: NO REAL REASON', 'JERRY SEINFELD: 23 HOURS TO KILL', 'BILL BURR ON THE LATE SHOW WITH DAVID LETTERMAN',
          'SINCERELY LOUIS CK', 'PETE DAVIDSON: SMD']



# In[2]:


#print(len(titles))
#print(len(urls))
transcripts = [url_to_transcript(u) for u in urls]


# In[3]:


#Make a new directory to hold the text files
#!mkdir transcripts

for i, c in enumerate(titles):
    with open("transcripts/" + c + ".txt", "wb") as file:
         pickle.dump(transcripts[i], file)


# In[4]:


#Open data
data = {}
for i, c in enumerate(titles):
    with open("transcripts/" + c + ".txt", "rb") as file:
        data[c] = pickle.load(file)


# In[5]:


#data.keys()


# In[6]:


#Check data key
##next(iter(data.keys()))


# In[7]:


#Check data value
##next(iter(data.values()))


# In[8]:


#Test get data
#data['TIG NOTARO: BOYISH GIRL INTERRUPTED'][:2]


# # Cleaning Data

# In[9]:


# Function change this to key: title, value: string format
def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text


# In[10]:


data_combined = {key: [combine_text(value)] for (key, value) in data.items()}
#data_combined


# In[11]:


# We can either keep it in dictionary format or put it into a pandas dataframe
import pandas as pd
pd.set_option('max_colwidth',200)

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
data_df


# In[12]:


#Check data frame
#data_df.transcript.loc['TIG NOTARO: BOYISH GIRL INTERRUPTED']


# In[13]:


# Apply a first round of text cleaning techniques
#'''Lowercase, remove text in square brackets, remove punctuation, remove words containing numbers.'''
import re
import string

def clean_text_round1(text):
    
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)


# In[14]:


# Shown the updated text
data_clean = pd.DataFrame(data_df.transcript.apply(round1))
data_clean


# In[15]:


# Apply a second round of cleaning
#'''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''

def clean_text_round2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)


# In[16]:


# Let's take a look at the updated text
data_clean = pd.DataFrame(data_clean.transcript.apply(round2))
## Now we have data_clean and data_frame
data_clean


# In[17]:


#Show data frame again.
#data_df


# In[18]:


#Make a corpus
data_df.to_pickle("corpus.pkl")


# In[19]:


#Create document-term matrix using CountVectorizer, exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
data_dtm


# In[20]:


#make a data matrix
data_dtm.to_pickle("dtm.pkl")


# In[21]:


# Pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))


# In[22]:


# Read in the document-term matrix
import pandas as pd

data = pd.read_pickle('dtm.pkl')
data = data.transpose()
data.head()


# In[23]:


# Find the top 30 words by each documents
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict


# In[24]:


# Print the top 15 words by each documents
for title, top_words in top_dict.items():
    print(title)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('---')


# In[25]:


# Look at the most common top words --> add them to the stop word list
from collections import Counter

# Let's first pull out the top 30 words for each comedian
words = []
for title in data.columns:
    top = [word for (word, count) in top_dict[title]]
    for t in top:
        words.append(t)
        
words


# In[26]:


# Let's aggregate this list and identify the most common words along with how many routines they occur in
Counter(words).most_common()


# In[27]:


# If more than half of the documetns have it as a top word, exclude it from the list
add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
add_stop_words


# In[28]:


# Let's update our document-term matrix with the new list of stop words
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

# Read in cleaned data
data_clean = pd.read_pickle('data_clean.pkl')

# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.transcript)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

# Pickle it for later use
import pickle
pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")


# In[29]:


# Word clouds!
# Terminal / Anaconda Prompt: conda install -c conda-forge wordcloud
#from wordcloud import WordCloud

#wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
#               max_font_size=150, random_state=42)


# In[30]:


# Reset the output dimensions
#import matplotlib.pyplot as plt

#plt.rcParams['figure.figsize'] = [64, 24]

#Create subplots for each documents
#for index, titles in enumerate(data.columns):
#    wc.generate(data_clean.transcript[titles])
#    
#    plt.subplot(50, 1, index+1)
#    plt.imshow(wc, interpolation="bilinear")
#    plt.axis("off")
#    plt.title(titles)
#    
#plt.show()


# # Sentiment Analysis

# In[31]:


import pandas as pd

data = pd.read_pickle('corpus.pkl')
data


# In[32]:


# Create quick lambda functions to find the polarity and subjectivity of each routine
# Terminal / Anaconda Navigator: conda install -c conda-forge textblob
from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['transcript'].apply(pol)
data['subjectivity'] = data['transcript'].apply(sub)
data


# # Sentiment Analysis With New Document

# In[68]:


realtime_start = round(time.time() * 1000)
input_url = 'https://scrapsfromtheloft.com/2021/06/01/bo-burnham-inside-transcript/'

input_title = 'BO BURNHAM: INSIDE'


# In[69]:


input_trans = url_to_transcript(input_url)
input_trans


# In[70]:


with open("transcripts/" + input_title + ".txt", "wb") as file:
         pickle.dump(input_trans, file)


# In[71]:


data_input = {}
with open("transcripts/" + input_title + ".txt", "rb") as file:
        data_input[input_title] = pickle.load(file)
data_input


# In[72]:


data_combined_input = {input_title: [combine_text(data_input[input_title])]}
data_combined_input


# In[73]:


import pandas as pd
pd.set_option('max_colwidth',200)

data_df2 = pd.DataFrame.from_dict(data_combined_input).transpose()
data_df2.columns = ['transcript']
data_df2 = data_df2.sort_index()
data_df2


# In[74]:


data_clean_input = pd.DataFrame(data_df2.transcript.apply(round1))
data_clean_input


# In[75]:


data_clean_input = pd.DataFrame(data_clean_input.transcript.apply(round2))
data_clean_input


# In[76]:


#Create document-term matrix using CountVectorizer, exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean_input.transcript)
data_dtm2 = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm2.index = data_clean_input.index
data_dtm2


# In[77]:


#make a data matrix
data_dtm2.to_pickle("dtm_input.pkl")


# In[78]:


# Read in the document-term matrix
import pandas as pd

data_input = pd.read_pickle('dtm_input.pkl')
data_input = data_input.transpose()
data_input.head()


# In[79]:


#data_input.columns


# In[80]:


# Find the top 30 words by each documents
top_dict2 = {}

top = data_input[input_title].sort_values(ascending=False).head(30)
top_dict2[input_title]= list(zip(top.index, top.values))

top_dict2


# In[81]:


#top_dict


# In[82]:


import threading
import time

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, from_index, top_dict, top_dict2, input_title, titles):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.from_index = from_index
        self.top_dict = top_dict
        self.top_dict2 = top_dict2
        self.input_title = input_title
        self.titles = titles
        self.result = (0, 0)
    
    def run(self):
        print ("Starting " + self.name + "\n")
        self.result = cmpVSM(self.name, self.from_index, 0, self.top_dict, self.top_dict2, self.input_title, self.titles)
        print ("Exiting " + self.name)
    
    def getMax(self):
        return self.result[0]
    
    def getMaxVal(self):
        return self.result[1]
        
def cmpVSM(threadName, from_index, delay, top_dict, top_dict2, input_title, titles):
        maximum_index = 0
        maxvalue = 0
        point = 0
        for i in range(10):
            if exitFlag:
                threadName.exit()
            
            time.sleep(delay)
            
            for index in range(30):
                if(top_dict[titles[i + from_index]][index][0] == top_dict2[input_title][index][0]):
                    point +=1
            #print(point)
            if(maxvalue < point):
                maxvalue = point
                maximum_index = i + from_index
            point = 0
        print ("%s: %s" % (threadName, time.ctime(time.time())))
        return (maximum_index, maxvalue)
        

# Create new threads
thread1 = myThread(1, "Thread-1", 0, top_dict, top_dict2, input_title, titles)
thread2 = myThread(2, "Thread-2", 10, top_dict, top_dict2, input_title, titles)
thread3 = myThread(3, "Thread-3", 20, top_dict, top_dict2, input_title, titles)
thread4 = myThread(4, "Thread-4", 30, top_dict, top_dict2, input_title, titles)
thread5 = myThread(5, "Thread-5", 40, top_dict, top_dict2, input_title, titles)

time_start = round(time.time() * 1000)

# Start new Threads
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()

thread1.join()
thread2.join()
thread3.join()
thread4.join()
thread5.join()

print ("Exiting Main Thread")

result = (thread1.getMax(), thread1.getMaxVal())


if(result[1] < thread2.getMaxVal()):
    result = (thread2.getMax(), thread2.getMaxVal())
    
if(result[1] < thread3.getMaxVal()):
    result = (thread3.getMax(), thread3.getMaxVal())
    
if(result[1] < thread4.getMaxVal()):
    result = (thread4.getMax(), thread4.getMaxVal())
    
if(result[1] < thread5.getMaxVal()):
    result = (thread5.getMax(), thread5.getMaxVal())
time_end = round(time.time() * 1000)
realtime_end = round(time.time() * 1000)
print("Time: %s" % (time_end - time_start))
print("Real Time: %s" % (realtime_end - realtime_start))
#print(result)


# In[83]:


### Result
print("The nearest already annotated document:")
print(titles[result[0]])
print("Polarity: ")
print(data.at[titles[result[0]],"polarity"])
print("Subjectivity: ")
print(data.at[titles[result[0]],"subjectivity"])


# In[84]:


#data


# In[85]:


#non-paralel
time_start_non = round(time.time() * 1000)
maximum_index = 0
maxvalue = 0
point = 0

for i in range(50):
    for index in range(30):
        if(top_dict[titles[i]][index][0] == top_dict2[input_title][index][0]):
            point +=1
    if(maxvalue < point):
        maxvalue = point
        maximum_index = i
    point = 0
        
time_end_non = round(time.time() * 1000)
print("Non-P Time: %s" % (time_end_non - time_start_non))
print(titles[maximum_index])

