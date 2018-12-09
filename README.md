# How midterm election candidates tweeted and how that might have affected the results of the election

During the fall of 2018, as part of my data journalism research for [Storybench](http://www.storybench.org), a "cookbook for digital storytelling" by Northeastern University's School of Journalism, I collected twitter data from 2018 midterm election candidates (with R) to see how it might have affected the results of the election. To better understand the impact of these words, machine learning model was used to predict tweet sentiment, and results are visualized (with Python).

## Data
Data were scraped from Twitter with [rtweet](https://rtweet.info), a package developed by [Michael W. Kearney](https://mikewk.com) from University of Missouri.

```R
