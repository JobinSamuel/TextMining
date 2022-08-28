library(rvest) #Used for easy web scrapping 
library(XML) #To read File format which shares both file format and data on www
library(magrittr) # A forward pipe operator(Forward a value into next)
library(tm)  #Used for text mining
library(wordcloud) #Used to form word cloud
library(wordcloud2) #Also used to form word cloud
library(RWeka) #RWeka is a collection of machine learning algorithms for data mining tasks 
library(syuzhet) #Used to extracts sentiments
#Question -1
#Read data
amzn <- "https://www.amazon.in/New-Apple-iPhone-Pro-128GB/dp/B08L5H1B5P/ref=sr_1_1_sspa?crid=IBFXY67TWAHK&dchild=1&keywords=iphone+13+%2B+pro+max&qid=1634806878&sprefix=iphone+13%2Caps%2C293&sr=8-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUExWVEyMUlORDNaSEFQJmVuY3J5cHRlZElkPUEwNDMxNjcxVUxNUzNIQzJaNVZMJmVuY3J5cHRlZEFkSWQ9QTAwNjA5ODhaSDBDODU0MEZISUkmd2lkZ2V0TmFtZT1zcF9hdGYmYWN0aW9uPWNsaWNrUmVkaXJlY3QmZG9Ob3RMb2dDbGljaz10cnVl#customerReviews"

#Creating an empty variable
amazon_reviews <- NULL

#Using a forloop to read html page using html node
for (i in 1:30){
  murl <- read_html(as.character(paste(amzn,i,sep="=")))
  rev <- murl %>% html_nodes(".review-text") %>% html_text()
  amazon_reviews <- c(amazon_reviews,rev)
}

write.table(amazon_reviews,"Iphone 12.txt")

#### Sentiment Analysis ####
amr <- amazon_reviews
str(amr)
# Convert the character data to corpus type
cp <- Corpus(VectorSource(amr))

cp <- tm_map(cp, function(x) iconv(enc2utf8(x), sub='byte'))

# Data Cleansing
cp_1 <- tm_map(cp, tolower) #Converting to lowercase

cp_1 <- tm_map(cp_1, removePunctuation) #Removing punctuation

cp_1 <- tm_map(cp_1, removeNumbers) #Removing numbers

cp_1 <- tm_map(cp_1, removeWords, stopwords('english')) #Removing stop words

cp_1 <- tm_map(cp_1, stripWhitespace) #Striping white spaces 
inspect(cp_1[1]) #Checking how the cleansed data looks like

# Term document matrix 
Tdm <- TermDocumentMatrix(cp_1)
Tdm
# To remove sparse entries upon a specific value
corpus.dtm.frequent <- removeSparseTerms(Tdm, 0.99) 

Tdm <- as.matrix(Tdm) #Converting to matrix
dim(Tdm)

Tdm[1:20, 1:20]

inspect(cp_1[1])


rs <- rowSums(Tdm)  #Rowsums gives column sums of a Matrix or Data Frame, Based on a Grouping Variable
rs

r_sub <- subset(rs, rs >= 30) #Shows only the words which has value more than 30
r_sub

barplot(r_sub, las=1, col = rainbow(30))  # Bar plot

#Cleaning data again using the words which make no sense in barplot and removing them
cp_1 <- tm_map(cp_1, removeWords, c('thay','sure','way','still','say','saying','ther','call','read','yes','get','also','used','live','one','pre','assbut','eye'))
cp_1 <- tm_map(cp_1, stripWhitespace)

Tdm <- TermDocumentMatrix(cp_1)
Tdm

Tdm <- as.matrix(Tdm)
Tdm[100:109, 1:20]

# Bar plot after removal of the term 'phone'
rs <- rowSums(Tdm)
rs

r_sub <- subset(rs, rs >= 30)
r_sub
#Forming barplot after cleaning again
barplot(r_sub, las=1, col = rainbow(30))

##### Word cloud #####
wordcloud(words = names(r_sub), freq = r_sub)
#Sorting 
r_sub1 <- sort(rowSums(Tdm), decreasing = TRUE)
head(r_sub1)

wordcloud(words = names(r_sub1), freq = r_sub1) # all words are considered

#better visualization
wordcloud(words = names(r_sub1), freq = r_sub1, random.order=F, colors=rainbow(30), scale = c(2,0.5), rot.per = 0.4)

#Forming a triangular wordcloud
r1 <- data.frame(names(r_sub), r_sub)
colnames(r1) <- c('word', 'freq')
wordcloud2(r1, size=0.3, shape = 'triangle')

# Obtaining Sentiment scores 
y <- get_nrc_sentiment(amazon_reviews)
y
#Plotting barplot on sentimental scores 
barplot(colSums(y), las = 2, col = rainbow(10),
        ylab = 'Count',main= 'Sentiment scores for AMAZON  Reviews')

#### Bigram ####
minfreq_bigram <- 2
big <- NGramTokenizer(cp_1, Weka_control(min = 2, max = 2))
two_word <- data.frame(table(big))
sort_two <- two_word[order(two_word$Freq, decreasing = TRUE), ]

wordcloud(sort_two$big, sort_two$Freq, random.order = F, scale = c(2, 0.35), min.freq = minfreq_bigram, colors = brewer.pal(8, "Dark2"), max.words = 150)

#Question -2
imdb <- "https://www.imdb.com/title/tt1477834/reviews?ref_=tt_ov_rt" #Read data

IMDB_reviews <- NULL #Creating a null variable

#Using a forloop to read html page using html node
for (i in 1:30){
  murl <- read_html(as.character(paste(imdb,i,sep="=")))
  rev <- murl %>%
    html_nodes(".show-more__control") %>%
    html_text()
  IMDB_reviews <- c(IMDB_reviews,rev)
}

write.table(IMDB_reviews,"Aquaman.txt")

imbr <- IMDB_reviews

str(imbr) #shows the structure 

# Convert the character data to corpus type
I <- Corpus(VectorSource(imbr))

I <- tm_map(I, function(x) iconv(enc2utf8(x), sub='byte'))

# Data Cleansing
i_1 <- tm_map(I, tolower) #Converting to lower case

i_1 <- tm_map(i_1, removePunctuation) #Removing punctuation

i_1<- tm_map(i_1, removeNumbers) #Removing numbers

i_1 <- tm_map(i_1, removeWords, stopwords('english')) #Removing stop words

i_1 <- tm_map(i_1, stripWhitespace)#Striping white spaces 

inspect(i_1[1]) #Checking the cleansed data

#Forming a term document matrix
tDm <- TermDocumentMatrix(i_1)
tDm

# To remove sparse entries upon a specific value
crpus.dtm.frequent <- removeSparseTerms(tDm, 0.99) 
tDm <- as.matrix(tDm) #Converting to matrix
dim(tDm) #Checking the dimensions

tDm[1:20, 1:20]

inspect(i_1[1])

mv <- rowSums(tDm) #Rowsums gives column sums of a Matrix or Data Frame, Based on a Grouping Variable
mv

mv_sub <- subset(mv, mv >= 40) #Choosing values which have rosums value more or equal to 40 
mv_sub

barplot(mv_sub, las=2, col = rainbow(30))# Bar plot

#Removing words which doesnt make sense 
i_1 <- tm_map(i_1, removeWords, c('already','try','make','reasons','either','says','onif','dude','roll','ive','will','dceu','also','thus','pee'))
i_1 <- tm_map(i_1, stripWhitespace)
 
#Forming termdocument matrix
tDm <- TermDocumentMatrix(i_1)
tDm

tDm <- as.matrix(tDm)
tDm[100:150, 1:20]

# Bar plot after removal of the term 'phone'
mv <- rowSums(tDm)
mv

mv_sub <- subset(mv, mv >= 60)
mv_sub

barplot(mv_sub, las=1, col = rainbow(30)) #Forming barplot after cleaning the data again

#Forming word cloud
wordcloud(words = names(mv_sub), freq = mv_sub)

mv_sub1 <- sort(rowSums(tDm), decreasing = TRUE) #Sorting according to rowsums value
head(mv_sub1)

#Better visualization
wordcloud(words = names(mv_sub1), freq = mv_sub1, random.order=F, colors=rainbow(30), scale = c(2,0.5), rot.per = 0.4)
#This wordcloud might take some time

# Obtaining Sentiment scores 
s <- get_nrc_sentiment(IMDB_reviews)
s
#Plotting bar plot for sentiment scores for imdb reviews
barplot(colSums(s), las = 2, col = rainbow(10),
        ylab = 'Count',main= 'Sentiment scores for IMDB Reviews')

#Question -3
#Read data
flp <- "https://www.flipkart.com/apple-watch-series-3-gps-42-mm-space-grey-aluminium-case-black-sport-band/product-reviews/itm91c560e722cdd?pid=SMWF94AYMNYHTYDJ&lid=LSTSMWF94AYMNYHTYDJFOPINH&marketplace=FLIPKART&page"

#Creating an null variable
flp_reviews <- NULL

#Using a forloop to read html page using html node
for (i in 1:20){
  url <- read_html(as.character(paste(flp,i,sep="")))
  rev <- url %>% html_nodes("._27M-vq") %>% html_text()
  flp_reviews <- c(flp_reviews,rev)
}


write.table(flp_reviews,"Apple Watch.txt")

#### Sentiment Analysis ####
flpkt <- flp_reviews

str(flpkt) #Checking the structure
# Convert the character data to corpus type
cpvs <- Corpus(VectorSource(flpkt))

cpvs <- tm_map(cpvs, function(x) iconv(enc2utf8(x), sub='byte'))

# Data Cleansing
fp_1 <- tm_map(cpvs, tolower) #Converting to lower case

fp_1 <- tm_map(fp_1, removePunctuation) #Removing punctuations

fp_1<- tm_map(fp_1, removeNumbers) #Removing numbers

fp_1 <- tm_map(fp_1, removeWords, stopwords('english')) #Removing stopwords


fp_1 <- tm_map(fp_1, stripWhitespace)#Striping white spaces 
inspect(fp_1[1]) #Checking cleansed data

#TermDocument Matrix
TDm <- TermDocumentMatrix(fp_1)
TDm

crpus.dtm.frequen <- removeSparseTerms(TDm, 0.99) 
#Converting into matrix
TDm <- as.matrix(TDm)
dim(TDm) #Checking the dimensions

TDm[1:20, 1:20]

fpk <- rowSums(TDm)#Rowsums gives column sums of a Matrix or Data Frame, Based on a Grouping Variable
fpk

fp_sub <- subset(fpk, fpk >= 40) #Considering values which has the rowsums value greater than or equal to 40
fp_sub
#Barplot
barplot(fp_sub, las=2, col = rainbow(30))
#Adding new words to be removed
fpk_1 <- tm_map(fp_1, removeWords, c('already',' s','dont','make','reasons','either','says','onif','dude','roll','ive','will','dceu','also','thus','pee','give'))
fpk_1 <- tm_map(fp_1, stripWhitespace)#Stripping whitespaces

#TermDocument Matrix
TDm <- TermDocumentMatrix(fp_1)
TDm
#Converting into matrix
TDm <- as.matrix(TDm)
TDm[100:150, 1:20]

# Bar plot after removal of the term 'phone'
fpk <- rowSums(TDm) ##Rowsums gives column sums of a Matrix or Data Frame, Based on a Grouping Variable
fpk

fp_sub <- subset(fpk, fpk >= 30) #Considering values which has the rowsums value greater than or equal to 30
fp_sub

#Plotting bar plot after cleaning of data again
barplot(fp_sub, las=1, col = rainbow(30))

#Wordcloud
wordcloud(words = names(fp_sub), freq = fp_sub)
#Sorting basedon rowsums values
fp_sub1 <- sort(rowSums(TDm), decreasing = TRUE)
head(fp_sub1)

#Another wordcloud for better visualization
wordcloud(words = names(fp_sub1), freq = fp_sub1, random.order=F, colors=rainbow(30), scale = c(2,0.5), rot.per = 0.4)
#This could take some time
# Obtaining Sentiment scores 
z <- get_nrc_sentiment(flp_reviews)
z
barplot(colSums(z), las = 2, col = rainbow(10),
        ylab = 'Count',main= 'Sentiment scores for Flipkart Reviews')
