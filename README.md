# How midterm election candidates tweeted and how that might have affected the results of the election

During the fall of 2018, as part of my data journalism research for [Storybench](http://www.storybench.org), a "cookbook for digital storytelling" by Northeastern University's School of Journalism, I collected twitter data from 2018 midterm election candidates (with R) to see how it might have affected the results of the election. To better understand the impact of these words, machine learning model was used to predict tweet sentiment, and results are visualized (with Python).

## Collecting twitter data with R
Data were scraped from Twitter with [rtweet](https://rtweet.info), a package developed by [Michael W. Kearney](https://mikewk.com) from the University of Missouri. Data are either from twitter accounts of candidate themselves or their official campaign accounts.

```R
library("rtweet")
allcandidates <- get_timelines(
  c("SenatorCantwell", "Susan4Senate", "jontester", "MattForMontana", "SenFeinstein", "kdeleon", "RosenforNevada", "SenDeanHeller", "kyrstensinema", "RepMcSally", "MittRomney", "JennyWilsonUT", "SenJohnBarrasso", "MartinHeinrich", "MickRich4Senate", "GovGaryJohnson", "maziehirono", "rcurtis808", "tedcruz", "BetoORourke", "FLGovScott", "SenBillNelson", "RepKevinCramer", "SenatorHeitkamp", "SenatorFischer", "JaneRaybould", "amyklobuchar", "NewbergerJim", "TinaSmithMN", "KarinHousley", "tammybaldwin", "LeahVukmir", "stabenow", "JohnJamesMI", "HawleyMO", "clairecmc", "RogerWicker", "dbaria", "MarshaBlackburn", "PhilBredesen", "braun4indiana", "JoeforIndiana", "SenSherrodBrown", "JimRenacci", "SenBobCasey", "louforsenate", "timkaine", "CoreyStewartVA", "Sen_JoeManchin", "MorriseyWV", "SenatorCardin", "Campbell4MD", "SenatorCarper", "RobArlett", "SenatorMenendez", "BobHugin", "SenGillibrand", "CheleFarley", "ChrisMurphyCT", "MattCoreyCT", "elizabethforma", "RepGeoffDiehl", "SenWhitehouse", "flanders4senate", "SenAngusKing", "RingelsteinME", "SenSanders", "ZupanForSenate"), retryonratelimit = TRUE, n = 3000, include_rts = FALSE
)

write_as_csv(allcandidates, "/your_own_file_directory/all_tweets.csv", prepend_ids = TRUE, na = "", fileEncoding = "UTF-8") #saving data in a csv file
```


