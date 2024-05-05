# import pandas as pd
# import numpy as np 
# import nltk
# from tqdm.notebook import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns

# # nltk.download('maxent_ne_chunker')
# # nltk.download('averaged_perceptron_tagger')
# # nltk.download('words')
# # nltk.download('vader_lexicon')
# plt.style.use('ggplot')

# import nltk

# df = pd.read_csv('../AI/Reviews.csv',encoding='utf-8')
# # print(df.shape)
# # print(df.head(500))
# # print(df.shape)

# # ax=df['Score'].value_counts().sort_index() \
# #     .plot(kind='bar',
# #           title='Count of Reviwes by Starts',
# #           figsize=(10,5))
# # ax.set_xlabel("Reviwe Starts")
# # plt.show()

# #basci nltk
# example = df['Text'][50]
# # print(example)


# tokens = nltk.word_tokenize(example)
# # print(tokens)
# # print(tokens[:10])

# # pos_tags = nltk.pos_tag(tokens)
# # print(pos_tags)

# tagged = nltk.pos_tag(tokens)
# tagged[:10]

# entities = nltk.chunk.ne_chunk(tagged)
# # entities.pprint()

# #VADER - stand for  Valance Aware Dictionary and sEntiment Reasonerr) -  bag of words approch
# # we will use nltk's sentimentIntensityAnalyzer to get the neg/neu/poss scores of the text

# from nltk.sentiment import SentimentIntensityAnalyzer
# from tqdm.notebook import tqdm

# sia = SentimentIntensityAnalyzer()
# # print(sia)

# # sentiment_scores = sia.polarity_scores("I am down to earth")
# # print(sentiment_scores)

# sentiment_scores = sia.polarity_scores(example)
# # print(sentiment_scores)

# # Run the polarity score on the entrie dataset
# # print(df)
# res = {}

# # Iterate over the DataFrame without tqdm
# for index, row in tqdm(df.iterrows(), total=len(df)):
#     text = row['Text']
#     myid = row['Id']
#     res[myid] = sia.polarity_scores(text)
#     # print(f"ID: {myid}, Sentiment Scores: {res[myid]}")

    
# dataframe = pd.Dataframe(res)
# # print(dataframe)
# vaders = pd.DataFrame(res) 
# vaders = vaders.reset_index().rename(columns={'index':'Id'})
# vaders = vaders.merge(df,how='left')
# # print(vaders)

# # Now we have  sentiment socre and metadata
# vaders.head()
# # print(vaders)

# ax = sns.barplot(data=vaders, x='Score', y='compound')
# ax.set_title("Compound Score by Amazon Star Review")
# plt.show()

# fig, axs = plt.subplots(1, 3, figsize=(15,5))
# sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
# sns.barplot(data=vaders, x='Score', y='neutral', ax=axs[1])  # Assuming 'neutral' is the column name for neutral sentiment
# sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])       # Assuming 'neg' is the column name for negative sentiment

# # Setting titles for each subplot
# axs[0].set_title('Positive')
# axs[1].set_title('Neutral')
# axs[2].set_title('Negative')

# plt.show()


import pandas as pd
import numpy as np 
import nltk
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer

# nltk.download('maxent_ne_chunker')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('words')
# nltk.download('vader_lexicon')
plt.style.use('ggplot')

import nltk

df = pd.read_csv('../AI/Reviews.csv', encoding='utf-8')

# Define function to get sentiment scores
def get_sentiment_scores(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

# Apply sentiment analysis to each review
res = {}
for index, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = get_sentiment_scores(text)

# Convert the sentiment scores into a DataFrame
vaders = pd.DataFrame(res).T.reset_index().rename(columns={'index':'Id'})

# Merge with original DataFrame to get metadata
vaders = vaders.merge(df, how='left')

# Plot Compound Score by Amazon Star Review
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title("Compound Score by Amazon Star Review")
plt.show()

# Plot Positive, Neutral, and Negative scores
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])

# Setting titles for each subplot
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')

plt.show()
