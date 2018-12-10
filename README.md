# How midterm election candidates (for the Senate) tweeted and how that might have affected the results of the election

During the fall of 2018, as part of my data journalism research for [Storybench](http://www.storybench.org), a "cookbook for digital storytelling" by Northeastern University's School of Journalism, I collected twitter data from 2018 midterm election candidates (with R) to see how it might have affected the results of the election. To better understand the impact of these words, machine learning model was used to predict tweet sentiment, and results are visualized (with Python).

## Collecting twitter data (R)
Data were scraped from Twitter with [rtweet](https://rtweet.info), a package developed by [Michael W. Kearney](https://mikewk.com) from the University of Missouri. Data are either from twitter accounts of candidate themselves or their official campaign accounts.

```R
library("rtweet")
allcandidates <- get_timelines(
  c("SenatorCantwell", "Susan4Senate", "jontester", "MattForMontana", "SenFeinstein", "kdeleon", "RosenforNevada", "SenDeanHeller", "kyrstensinema", "RepMcSally", "MittRomney", "JennyWilsonUT", "SenJohnBarrasso", "MartinHeinrich", "MickRich4Senate", "GovGaryJohnson", "maziehirono", "rcurtis808", "tedcruz", "BetoORourke", "FLGovScott", "SenBillNelson", "RepKevinCramer", "SenatorHeitkamp", "SenatorFischer", "JaneRaybould", "amyklobuchar", "NewbergerJim", "TinaSmithMN", "KarinHousley", "tammybaldwin", "LeahVukmir", "stabenow", "JohnJamesMI", "HawleyMO", "clairecmc", "RogerWicker", "dbaria", "MarshaBlackburn", "PhilBredesen", "braun4indiana", "JoeforIndiana", "SenSherrodBrown", "JimRenacci", "SenBobCasey", "louforsenate", "timkaine", "CoreyStewartVA", "Sen_JoeManchin", "MorriseyWV", "SenatorCardin", "Campbell4MD", "SenatorCarper", "RobArlett", "SenatorMenendez", "BobHugin", "SenGillibrand", "CheleFarley", "ChrisMurphyCT", "MattCoreyCT", "elizabethforma", "RepGeoffDiehl", "SenWhitehouse", "flanders4senate", "SenAngusKing", "RingelsteinME", "SenSanders", "ZupanForSenate"), retryonratelimit = TRUE, n = 3000, include_rts = FALSE
)

write_as_csv(allcandidates, "/your_own_file_directory/all_tweets.csv", prepend_ids = TRUE, na = "", fileEncoding = "UTF-8") #saving data in a csv file
```


## Training a machine learning language model to predict each tweet's sentiment (Python)
The model is developed by [Jeremy Howard](https://twitter.com/jeremyphoward), a deep learning researcher and educator, as part of his course [fast.ai](https://github.com/fastai/fastai). All code below (Python) were run on paperspace.com. Instructions on how to set up a machine on paperspace can be found [here](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/paperspace.md).

To train the model, I ran Jeremy's [code](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb) line by line on paperspace. His model was trained by a [large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) containing 50,000 reviews from IMDB. 

## Predicting each tweet's sentiment (Python)
```python
# loading required packages
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.learner import *

import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling

from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *

import dill as pickle
import spacy
import numpy as np
import pandas as pd

# path of data
PATH='data/aclImdb/'

# defining required variables, parameters and functions, and loading the model
em_sz = 200  # size of each embedding vector
nh = 200     # number of hidden activations per layer
nl = 3       # number of layers
opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
TEXT = data.Field(lower=True, tokenize=spacy_tok)
TEXT = pickle.load(open(f'{PATH}models/TEXT.pkl','rb'))
IMDB_LABEL = data.Field(sequential=False)
bs=64; bptt=70

splits = torchtext.datasets.IMDB.splits(TEXT, IMDB_LABEL, 'data/') #this splits all the words in the model into positive and negative

md2 = TextData.from_splits(PATH, splits, bs)

m3 = md2.get_model(opt_fn, 1500, bptt, emb_sz=em_sz, n_hid=nh, n_layers=nl, 
           dropout=0.1, dropouti=0.4, wdrop=0.5, dropoute=0.05, dropouth=0.3)
m3.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)

m3.load_encoder(f'adam3_10_enc')

m3.load_cycle('imdb2', 4)

accuracy(*m3.predict_with_targs()) #accuracy of my model is 0.90849358974358974

# loading data and storing data in separate panda dataframes
AllCandidates = pd.read_csv("data/all_tweets.csv")

candidates = [1]*63 # creating 63 lists because I am analysing 63 candidates' tweets; 1 is just a randomly chosen number

for i in range (0,63):
    candidates[i] = AllCandidates[AllCandidates['screen_name'].str.contains(names[i])] # by doing this, data for each candidate are stored in separate dataframes. For example, candidates[0] stores all data for "SenatorCantwell"; candidates[1] for "Susan4Senate"; candidates[2] for "jontester"...
 
# predicting sentiment for each candidate's tweets
m = m3.model #this is the model
m[0].bs=1 #this is the batch size. so, analyzing one tweet at a time
prediction = []
for i in range(candidate[j].values[:,4].shape[0]): #replace j with 0, 1, 2, 3...63 for each candidate
    ss = candidate[j]['text'][i+a] #saving the ith tweet here; replace a with 0 (Cantwell's tweets start here), 3030 (Susan's tweets start here)... 
    s = [spacy_tok(ss)] #tokenizing the words
    t = TEXT.numericalize(s) #giving the words a number
    
    m.eval() #evaluates the probability of positives or negatives
    m.reset() #reset it so it can analyze the next tweet
    res,*_ = m(t) #storing the probability value in res
    prediction.append(IMDB_LABEL.vocab.itos[to_np(torch.topk(res[-1],1)[1])[0]])
    
    
# showing results
positive = [i for i,x in enumerate(prediction) if x == 'pos'] # all positive tweets here
negative = [i for i,x in enumerate(prediction) if x == 'neg'] # all negative tweets here
len(positive) #number of positive tweets by each candidate
len(negative) #number of negative tweets by each candidate
```

## Visualizing results with matplotlib (Python)
The closer to the election date, the more candidates tweeted.
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

candidates = pd.read_csv("Updated Candidates.csv")
percentage_of_positive_tweets = [72.77215502,72.89565754,74.76340694,77.25617934,77.09820567,78.4372093,75.56086797,75.48902875,77.11086923,75.85608332,77.11526349,81.2329108]
month = [1,2,3,4,5,6,7,8,9,10,11,12]
fig=plt.figure(figsize=(20,10))
plt.scatter(month,percentage_of_positive_tweets, s=500)
plt.ylabel('% of positive tweets from all candidates', fontsize = 25)
plt.yticks(fontsize =27)
plt.xlabel('month and year', fontsize=25)
plt.xticks(month,
          ['Dec 2017', 'Jan 2018', 'Feb 2018', 'March 2018', 'April 2018', 'May 2018', 'June 2018', 'July 2018', 'Aug 2018', 'Sept 2018', 'Oct 2018', 'Nov 2018'], fontsize = 16)
fig.savefig('positive_tweets.jpg')
```

![ ](https://github.com/FlorisWu/twitter-sentiment-analysis/blob/master/tweets_over_time.jpg)

Candidates were generally more positive as they approached election day.

```python
fig=plt.figure(figsize=(20,10))
plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12],[4051,4859,4121,5219,5183,5375,5438,5879,6431,6337,9526,4023], s=500)
plt.ylabel('number of tweets from all candidates', fontsize = 30)
plt.xlabel('month and year', fontsize=25)
plt.yticks(fontsize =27)
plt.xticks(month,
          ['Dec 2017', 'Jan 2018', 'Feb 2018', 'March 2018', 'April 2018', 'May 2018', 'June 2018', 'July 2018', 'Aug 2018', 'Sept 2018', 'Oct 2018', 'Nov 2018'], fontsize = 16)
fig.savefig('tweets_over_time.jpg')
```
![ ](https://github.com/FlorisWu/twitter-sentiment-analysis/blob/master/positive_tweets.jpg)

In general, the more candidates tweeted, the more votes they got. And Democrats tweeted more and got more votes.
```python
fig=plt.figure(figsize=(20,10))
plt.scatter(np.log(candidates[candidates['Party (D/R/L/I)'] == 'D']['Total number of tweets']),candidates[candidates['Party (D/R/L/I)'] == 'D']['percentage of vote'], s=500, color='blue')
plt.scatter(np.log(candidates[candidates['Party (D/R/L/I)'] == 'R']['Total number of tweets']),candidates[candidates['Party (D/R/L/I)'] == 'R']['percentage of vote'], s=500, color='red')
plt.scatter(np.log(candidates[candidates['Party (D/R/L/I)'] == 'I']['Total number of tweets']),candidates[candidates['Party (D/R/L/I)'] == 'I']['percentage of vote'], s=500, color='purple')
plt.scatter(np.log(candidates[candidates['Party (D/R/L/I)'] == 'L']['Total number of tweets']),candidates[candidates['Party (D/R/L/I)'] == 'L']['percentage of vote'], s=500, color='yellow')
plt.ylabel('% of vote', fontsize = 30)
plt.xlabel('log(number of tweets)', fontsize=25)
plt.yticks(fontsize =27)
plt.xticks(fontsize =25)
plt.legend(('Democrats', 'Republicans', 'Independents', 'Libertarians'), loc='upper left', fontsize=21)
fig.savefig('tweets_and_votes.jpg')
```

![ ](https://github.com/FlorisWu/twitter-sentiment-analysis/blob/master/tweets_and_votes.jpg)

In general, the more followers candidates had, the more votes they got. Again, Democrats had more followers and more votes.
```python
fig=plt.figure(figsize=(20,10))
plt.scatter(np.log(candidates[candidates['Party (D/R/L/I)'] == 'D']['Follower count']),candidates[candidates['Party (D/R/L/I)'] == 'D']['percentage of vote'], s=500, color='blue')
plt.scatter(np.log(candidates[candidates['Party (D/R/L/I)'] == 'R']['Follower count']),candidates[candidates['Party (D/R/L/I)'] == 'R']['percentage of vote'], s=500, color='red')
plt.scatter(np.log(candidates[candidates['Party (D/R/L/I)'] == 'I']['Follower count']),candidates[candidates['Party (D/R/L/I)'] == 'I']['percentage of vote'], s=500, color='purple')
plt.scatter(np.log(candidates[candidates['Party (D/R/L/I)'] == 'L']['Follower count']),candidates[candidates['Party (D/R/L/I)'] == 'L']['percentage of vote'], s=500, color='yellow')
plt.ylabel('% of vote', fontsize = 30)
plt.xlabel('log(number of twitter followers)', fontsize=30)
plt.yticks(fontsize =27)
plt.xticks(fontsize =25)
plt.legend(('Democrats', 'Republicans', 'Independents', 'Libertarians'), loc='upper left', fontsize=30)
fig.savefig('followers_and_votes.jpg')
```
![ ](https://github.com/FlorisWu/twitter-sentiment-analysis/blob/master/followers_and_votes.jpg)

For Democrats, the more positive tweets they had, the less votes they had.

```python
fig=plt.figure(figsize=(20,10))
plt.scatter(candidates[candidates['Party (D/R/L/I)'] == 'D']['% of positives'],candidates[candidates['Party (D/R/L/I)'] == 'D']['percentage of vote'], s=500, color='blue')
plt.scatter(candidates[candidates['Party (D/R/L/I)'] == 'R']['% of positives'],candidates[candidates['Party (D/R/L/I)'] == 'R']['percentage of vote'], s=500, color='red')
plt.scatter(candidates[candidates['Party (D/R/L/I)'] == 'I']['% of positives'],candidates[candidates['Party (D/R/L/I)'] == 'I']['percentage of vote'], s=500, color='purple')
plt.scatter(candidates[candidates['Party (D/R/L/I)'] == 'L']['% of positives'],candidates[candidates['Party (D/R/L/I)'] == 'L']['percentage of vote'], s=500, color='yellow')
plt.ylabel('% of vote', fontsize = 30)
plt.xlabel('% of positive tweets', fontsize=25)
plt.yticks(fontsize =27)
plt.xticks(fontsize =25)
plt.legend(('Democrats', 'Republicans', 'Independents', 'Libertarians'), loc='lower right', fontsize=16)
fig.savefig('positives_and_votes.jpg')
```
![ ](https://github.com/FlorisWu/twitter-sentiment-analysis/blob/master/positives_and_votes.jpg)

