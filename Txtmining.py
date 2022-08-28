import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping.used to scrap specific content 
import re #regular expression
import matplotlib.pyplot as plt #Used for plotting
from wordcloud import WordCloud #Used to form wordcloud
import nltk         #Python library for Natural Language Processing

##Extracting reviews of any product from e-commerce website Amazon.
#Performing sentiment analysis on this extracted data and build a unigram and bigram word cloud.

# creating empty reviews list 
amzn = []

for i in range(1,31):
  ip=[]  
  url="https://www.amazon.in/New-Apple-iPhone-11-64GB/product-reviews/B08L8C1NJ3/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser") #pulling html node
  # creating soup object to iterate over the extracted content 
  reviews = soup.find_all("span",attrs={"class","a-size-base review-text review-text-content"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
 
  amzn = amzn+ip  # adding the reviews of one page to empty list which in future contains all the reviews

# writng reviews in a text file 
with open("amzn.txt","w",encoding='utf8') as output:
    output.write(str(amzn))
	

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(amzn) 

# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower() #Converting the text to lowercase
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string) #Removing numbers

# words that contained in review
ip_reviews_words = ip_rev_string.split(" ")

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer #converting raw document to a tfidftransformer
vectorizer = TfidfVectorizer(ip_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(ip_reviews_words)

#Adding stopwords
with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/stop.txt","r") as sw:
    stop_words = sw.read()
#Spliting the stopwords    
stop_words = stop_words.split("\n")
#Adding extra stop words after looking at the wordplot
stop_words.extend(["iphone","mobile","time","ios","apple","device","screen","dosen","doesn","aren","battery","told","amazon","good","day","price","product","appario","retail"])

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs.
wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip) 

# positive words # Choose the path for +ve words stored in system
with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos) #Forming positive wordcloud

# negative words Choose path for -ve words stored in system
with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)  #Forming negative wordcloud


# wordcloud with bigram
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer() #It gives additional meaning to the model

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords

stopwords_wc = set(STOPWORDS)
customised_words = ['price', 'great'] # If you want to remove any particular word form text which does not contribute much in meaning

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

#Creating bag of words
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#####Extracting reviews for any movie from IMDB and perform sentiment analysis.

# creating empty reviews list 
imdb = []

#text show-more__control
for i in range(1,31):
  im=[]  
  url="https://www.imdb.com/title/tt1477834/reviews?ref_=tt_ov_rt"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser") #pulling html notes
  # creating soup object to iterate over the extracted content 
  reviews = soup.find_all("div",attrs={"class","text"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    im.append(reviews[i].text)  
 
  imdb = imdb+im  # adding the reviews of one page to empty list which in future contains all the reviews

# writng reviews in a text file 
with open("imdb.txt","w",encoding='utf8') as output:
    output.write(str(imdb))
	

# Joinining all the reviews into single paragraph 
i_p_rev_string = " ".join(imdb) 

# Removing unwanted symbols incase if exists
i_p_rev_string = re.sub("[^A-Za-z" "]+"," ", i_p_rev_string).lower() #Converting into lowercase
i_p_rev_string = re.sub("[0-9" "]+"," ", i_p_rev_string) #Removing numbers

# words that contained in reviews
i_p_reviews_words = i_p_rev_string.split(" ")

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer #converting raw document to a tfidftransformer
vectorizer = TfidfVectorizer(i_p_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(i_p_reviews_words)

with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/stop.txt","r") as sw:
    stop_words = sw.read() #Adding stopwords
    
stop_words = stop_words.split("\n") #Spliting stopwords

stop_words.extend(["wan","dafoe","kidman","orm","ii","vulko","manta","yahaya","computer"]) #Adding few extra stopwords

i_p_reviews_words = [w for w in i_p_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(i_p_reviews_words)

# WordCloud can be performed on the string inputs.
wordcloud_i_p = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_i_p)

# positive words # Choose the path for +ve words stored in system
with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
i_p_pos_in_pos = " ".join ([w for w in i_p_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(i_p_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos) #Forming a positive wordcloud

# negative words Choose path for -ve words stored in system
with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
i_p_neg_in_neg = " ".join ([w for w in i_p_reviews_words if w in negwords])

wordcloud_neg_i_n_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(i_p_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_i_n_neg) #Forming a negative wordcloud

###Choosing any other website on the internet and do some research on how to extract text and perform sentiment analysis

#Forming an empty variable
flpkt=[]

#text show-more__control
for i in range(1,31):
  ft=[]  
  url="https://www.flipkart.com/apple-watch-series-3-gps-42-mm-space-grey-aluminium-case-black-sport-band/product-reviews/itm91c560e722cdd?pid=SMWF94AYMNYHTYDJ&lid=LSTSMWF94AYMNYHTYDJFOPINH&marketplace=FLIPKART&page"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser") #pulling html notes
  # creating soup object to iterate over the extracted content 
  reviews = soup.find_all("div",attrs={"class","_27M-vq"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ft.append(reviews[i].text)  
 
  flpkt = flpkt+ft # adding the reviews of one page to empty list which in future contains all the reviews

# writng reviews in a text file 
with open("flpkt.txt","w",encoding='utf8') as output:
    output.write(str(flpkt))
	

# Joinining all the reviews into single paragraph 
_i_p_rev_string = " ".join(flpkt) 

# Removing unwanted symbols incase if exists
_i_p_rev_string = re.sub("[^A-Za-z" "]+"," ", _i_p_rev_string).lower() #Converting thetext to lowercase
_i_p_rev_string = re.sub("[0-9" "]+"," ", _i_p_rev_string) #Removing numbers

# words that contained in reviews
_i_p_reviews_words = _i_p_rev_string.split(" ") 

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer #converting raw document to a tfidftransformer
vectorizer = TfidfVectorizer(_i_p_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(_i_p_reviews_words)

with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/stop.txt","r") as sw:
    stop_words = sw.read() #Adding stopwords
    
stop_words = stop_words.split("\n") #Splitting stopwords

stop_words.extend(["apple","delhioct","sleeping","permalinkreport","read","more","moreflipkart","google","buyer","jul","abuse","don","iphone"]) #Adding extra stopwords

_i_p_reviews_words = [w for w in _i_p_reviews_words if not w in stop_words] 

# Joinining all the reviews into single paragraph 
_i_p_rev_string = " ".join(_i_p_reviews_words)

# WordCloud can be performed on the string inputs.
wo_rdcloud_i_p = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(_i_p_rev_string)

plt.imshow(wo_rdcloud_i_p)

# positive words # Choose the path for +ve words stored in system
with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
_i_p_pos_in_pos = " ".join ([w for w in _i_p_reviews_words if w in poswords])

wordcloud_pos_i_n_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(_i_p_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_i_n_pos) #Forming positive wordcloud

# negative words Choose path for -ve words stored in system
with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
_i_p_neg_in_neg = " ".join ([w for w in _i_p_reviews_words if w in negwords])

wordcloud_ne_g_i_n_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(_i_p_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_ne_g_i_n_neg) #Forming negative wordcloud
