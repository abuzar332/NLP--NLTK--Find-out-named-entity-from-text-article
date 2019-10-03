import nltk
import tokenize
from ImmportExportToCSV import penn2morphy
from ImmportExportToCSV import ExportToCSV
import urllib.request
from urllib.request import Request, urlopen
import numpy as np

######You can directly import from txt file ###############################################################

###############################################################

fileRead = open("testDataText.txt","r")
textFromFile=fileRead.read()
fileRead.close()

###############################################################

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

porter_stemmer=PorterStemmer()
lem=WordNetLemmatizer()

word="Multiplying"

print(lem.lemmatize(word,"VB"))
print(porter_stemmer.stem(word))
print("Done Check...")





###########################################################################################################

import re
textSp=textFromFile.split()   #instead of textL 

i=0 
length = len(textSp)  #list length 
while(i<length):
	if((len(textSp[i])>=50) | ('=' in textSp[i])):
		textSp.remove (textSp[i])	
		length = length -1   
		continue
	i = i+1


##JOIN back the text tokens and remove special chars and words associated with special chars
#text12=re.sub('[^a-zA-Z0-9 \n\.]', ' ', " ".join(textSp))
text1 =re.sub(r"[@$%_<>/]", "", " ".join(textSp), flags=re.I)
#text1=re.sub(r"[\n]", "\n\n", text1, flags=re.I)
#Removing Multiple Spaces
text1=re.sub(r"\s+"," ", text1, flags = re.I) 
#Removing Spaces from Start and End
text1=re.sub(r"^\s+","", text1, flags = re.I) 
#Removing a Single Character
text1 = re.sub(r"\s+[a-zA-Z]\s+", " ", text1)
#Replace /u2013 and /u2019



textSp1=text1.split()  #word_tokenize(text1)
file = open("sample.txt","w",encoding='utf-8')
file.write(text1)
file.close()

#print(text)
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize



tokens = [t for t in textSp1]
print(tokens)










#############################################################################################
""" FURTHER HIGH EEND PROCESSING HERE BELOW -  
                                                                                            #
"""                                                                                         #
#############################################################################################
#lemmenting the words
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

"""

from nltk.corpus import stopwords
sr= stopwords.words('english')
clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        
        clean_tokens.remove(token)


#tokens = [token for token in tokens if token not in stop]

# remove words less than three letters

#clean_tokens = [word for word in clean_tokens if len(word) >= 3]

file = open("outputlast.txt","w",encoding='utf-8')
file.write(" ".join(clean_tokens))
file.close()

pos_tag_tokens=nltk.pos_tag(tokens)

pos_tag_tokens_neat=[t for t in pos_tag_tokens if str(t[0]).lower() != str(t[1]).lower()]
ExportToCSV(pos_tag_tokens)
"""

from nltk.tokenize import PunktSentenceTokenizer
custom_sent_tokenizer = PunktSentenceTokenizer()  #can be sent a parameter in constructor to train the tokenizer as needed with train model text
tokenized = custom_sent_tokenizer.tokenize(" ".join(tokens))

def process_content():
    try:
        aa=[]
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked.label())

            
            for j in chunked:
                if type(j)==Tree:
                    aa.append(" ".join([token+'##'+pos for token, pos in j.leaves()]))

                else:
                    aa.append( "##".join(j))

            aa.append("~~~~~~EndC1~~~~~~")


            #chunked.draw()
            file = open("sampletest2.txt","w",encoding='utf-8')
            file.write("\n".join(aa)+"\n")
            file.close()

    except Exception as e:
        print(str(e))

def process_content_Chunking():
    try:
        aa=[]
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked.label())

            
            for j in chunked:
                if type(j)==Tree:
                    aa.append(" ".join([token+'##'+pos for token, pos in j.leaves()]))

                else:
                    aa.append( "##".join(j))

            aa.append("~~~~~~EndC1~~~~~~")


            #chunked.draw()

        file = open("sampletest3.txt","w",encoding='utf-8')
        file.write("\n".join(aa)+"\n")
        file.close() 


            
               

    except Exception as e:
        print(str(e))
def process_content_Find_namedEntity(tokenized):
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=False)
            namedEnt.draw()
    except Exception as e:
        print(str(e))

#process_content_Find_namedEntity(tokenized)
#process_content()



######Actual Usage here ######################################################

def process_content_Find_namedEntity_Save(tokenized):
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=False)
            np.savetxt('test.out', namedEnt) 
    except Exception as e:
        print(str(e))


#process_content_Find_namedEntity_Save(tokenized)


from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg

# sample text
sample = gutenberg.raw("bible-kjv.txt")
sample2="John went to school. But yesterday he was absent."
tok = sent_tokenize(sample)

for x in range(5):
    print(tok[x])

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

def get_namedEntity_continuous_chunks(text):
     tkns=custom_sent_tokenizer.tokenize(text)
     continuous_chunk = []
     current_chunk = []
     named_Entity=[]
     counter=0;

     for i in tkns:
         counter+=1
         words=nltk.word_tokenize(i)
         tagged_words=nltk.pos_tag(words)
         namedEnt = nltk.ne_chunk(tagged_words, binary=False)
         for j in namedEnt:
             if type(j) == Tree:
                     current_chunk.append(" ".join([token for token, pos in j.leaves()])+ "##"+j.label()+"##SentenceNo"+str(counter))
             elif current_chunk:
                     named_entity = " ".join(current_chunk)
                     if named_entity not in continuous_chunk:
                             continuous_chunk.append(named_entity)
                             current_chunk = []
             else:
                     continue
     return continuous_chunk



#testAA=process_content()

testBB=process_content_Chunking()
#bb=get_namedEntity_continuous_chunks(textFromFile)
print(testBB)

from nltk.corpus import stopwords
st_w=stopwords.words('english')
clean_tokens=tokens
#add commad and full stop and other marks
for token in tokens:
    if token in st_w:
        clean_tokens.remove(token)


from nltk.tokenize import RegexpTokenizer

custom_regexp_tokenizer=RegexpTokenizer('\w+')
clean_tokens=custom_regexp_tokenizer.tokenize(" ".join(clean_tokens))


#words = nltk.word_tokenize(s)

#words=[word.lower() for word in words if word.isalpha()]


freq = nltk.FreqDist(clean_tokens)
for key,val in freq.items():
    print(str(key) + ':' + str(val))

print(freq.most_common(2))
freq.plot(20, cumulative=False)

