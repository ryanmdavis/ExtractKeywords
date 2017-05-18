import json, itertools, nltk, string, os, operator
import numpy as np
from pprint import pprint
from collections import Counter
#from idlelib.idle_test import test_textview
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from __builtin__ import file
from math import sqrt
from string import maketrans
from collections import Counter
from summa import keywords

def get_text(json_file_loc):
    # parse all of the JSON objects in the file.  If there are more than one object, then
    # each object will be stored in an array
    with open(json_file_loc) as data_file:
        text=data_file.read()
        seq_num_loc_total=0
        seq_end_list=[0]*100
        seq_num_loc=0
        num_json_objects=-1
        
        # determine number of JSON objects
        while seq_num_loc>=0:
            num_json_objects+=1
            seq_num_loc=text[seq_num_loc_total:].find("sequenceNumber")
            seq_num_loc_total+=seq_num_loc+len("sequenceNumber")
            seq_end_list[num_json_objects]=seq_num_loc_total
        
        # now make an list of json objects
        #seq_num_loc=text.find("sequenceNumber")
        json_obj_list = [{} for _ in range(num_json_objects)]#initialize list of json objects
        end_loc_previous=0
        for json_object_num in range(num_json_objects):
            
            # find the end of the JSON object
            json_object_end_loc=seq_end_list[json_object_num] + text[seq_end_list[json_object_num]:].find("}}")+4
    
            
            # add JSON object to list
            json_obj_list[json_object_num]=json.loads(text[end_loc_previous:json_object_end_loc])
            end_loc_previous=json_object_end_loc

    # now make a list of string object with (1) lowercase only (2) punctuation replaced with spaces (3) newline, carrage returns r (4) only include entries that are not empty
    text_body = [json_obj_list[x]['result']['extractorData']['data'][0]['group'][0]['text_body'][0]['text'].lower().encode('ascii','replace').replace('\n', ' ').replace('\r', ' ').replace('?','').translate(None, string.punctuation) for x in range(num_json_objects) if bool(json_obj_list[x]['result']['extractorData']['data'])]
    # text_body = [json_obj_list[x]['result']['extractorData']['data'][0]['group'][0]['text_body'][0]['text'].lower().encode('ascii','ignore').translate(None, string.punctuation).replace('\n', ' ').replace('\r', ' ') for x in range(num_json_objects)]
    # text_body = [json_obj_list[x]['result']['extractorData']['data'][0]['group'][0]['text_body'][0]['text'].lower().encode('ascii','replace').replace('?s',' ').replace('?',' ').translate(None, string.punctuation).replace('\n', ' ').replace('\r', ' ') for x in range(num_json_objects)]
    text_keywords = [json_obj_list[x]['result']['extractorData']['data'][0]['group'][0]['keywords'][0]['text'].lower().encode('ascii','ignore').replace('\r', '').replace(', ',',').split(',') for x in range(num_json_objects) if bool(json_obj_list[x]['result']['extractorData']['data'])]
        
    #print json_obj_list[2]['result']['extractorData']['data'][0]['group'][0]['Mktofieldwrap'][0]['text']
    #text_list=[]
    return text_body,text_keywords

# tokenize_strings
#    input:
#        text with all lowercase and no punctuation.  Pass from get_text()
def tokenizeAndStemStrings(text):
    
    # turn text to tokens
    tokens = nltk.word_tokenize(text)
    
    # remove stop words
    tokens_no_sw = [word for word in tokens if not word in stopwords.words('english')]
    
    # stem words
    stemmed = []
    stemmer = PorterStemmer()
    for item in tokens_no_sw:
        # this line converts strings to unicode, so here I do it explicitly
        stemmed.append(stemmer.stem(unicode(item)))
    
    return stemmed

def getTextAndKeywords(parent_directory):
    text_list = []
    keyword_list = []
    num_keywords_list=[]
    file_id_list=[]
    for subdir, dirs, files in os.walk(parent_directory):
        for file_name in files:
            file_path = subdir + os.path.sep + file_name
            text_doc_list,keyword_doc_list=get_text(file_path)
            #stemmed=[tokenizeAndStemStrings(text_list[x]) for x in range(len(text_list))]
            for doc_num in range(len(text_doc_list)):
                file_id_list.append(file_name + "##" + str(doc_num))
                text_list.append(text_doc_list[doc_num])
                keyword_list.append(keyword_doc_list[doc_num])
                num_keywords_list.append(len(keyword_doc_list[doc_num]))
                
    return text_list,keyword_list,num_keywords_list

def createTfidfTable(text_list):
                
    tfidf = TfidfVectorizer(tokenizer=tokenizeAndStemStrings, stop_words='english',ngram_range=(1,4), use_idf=True, smooth_idf = False, norm=None)
    tfs = tfidf.fit_transform(text_list)

    return tfs, tfidf

def getRegExps(text_list,grammar):
    cp=nltk.RegexpParser(grammar)
    regexp_list=[[]]*len(text_list)
    num_regexp_list=[0]*len(text_list)
    for doc_num in range(len(text_list)):
        tokens = nltk.word_tokenize(text_list[doc_num])
        tagged = nltk.pos_tag(tokens) # part of speech tagging, required for searching for grammars
        tree=cp.parse(tagged)
        this_list=[]
        for subtree in tree.subtrees():
            if subtree.label() == 'KW': 
                this_list.append(' '.join([subtree.leaves()[x][0] for x in range(len(subtree.leaves()))]))
        regexp_list[doc_num]=this_list
        num_regexp_list[doc_num]=len(this_list)
        
    return regexp_list,num_regexp_list
# getRegExpKeywords(...)
# how to decide how many keywords to generate (specified using num_keywords argument): 
# (1) generate the same number of keywords from the algorithm as the gold standard.  For this option pass a dictionary to keyword_threshold where each value is the number of gold standard keywords for that file.
# (2) Use a fixed number or threshold for each file.  For this option pass just one double (tf-idf threshold or number of keywords depending on weight_type) to keyword_threshold

# weighting schemes (specified using weight_type input argument)
# weigh by number of occurances of regexp (weight_type='count')
# weigh by tf-idf weight_type='tfidf'
def getKeywordsByCount(regexps_list,num_keywords_to_choose_array):
        
    keywords=[[Counter(regexps_list[x]).most_common(num_keywords_to_choose_array[x])[y][0] for y in range(num_keywords_to_choose_array[x])] for x in range(len(regexps_list))]
    
    return keywords

def getKeywordsByTfidf(regexps_list,num_keywords_to_choose_array,tfidf):
    
    # allocate some memory for keywords
    keywords = [[]]*len(regexps_list)
    
    # define stemmer
    stemmer=PorterStemmer()
    
    # get feature names (i.e. potential keywords) and assign idf to a dictionary
    feature_names=tfidf.get_feature_names()
    idf=tfidf.idf_-1
    idf_dict={feature_names[x]:idf[x] for x in range(len(feature_names))}
    
    for doc_num in range(len(regexps_list)):
        # calculate term frequencies
        c=Counter(regexps_list[doc_num])
        term_list=c.most_common()
        # tf_dict={' '.join([stemmer.stem(w) for w in nltk.word_tokenize(term_list[x][0]) if not w in stopwords.words('english')]): term_list[x][1] for x in range(len(term_list))}
        tf_dict={' '.join([stemmer.stem(w) for w in nltk.word_tokenize(term_list[x][0]) if not w in stopwords.words('english')]): term_list[x][1] for x in range(len(term_list)) if term_list[x][0] in idf_dict}
    
        # now calculate tf-idf
        tfidf_dict={key: idf_dict[key]*tf_dict[key] for key in tf_dict.keys()}
        sorted_tfidf_keys=sorted(tfidf_dict.iteritems(),key=operator.itemgetter(1),reverse=True)[0:num_keywords_to_choose_array[doc_num]]
        
        keywords[doc_num]=[sorted_tfidf_keys[x][0] for x in range(len(sorted_tfidf_keys))]
        
    return keywords

def keywordExtractionPerformance(algorithm_keywords,gold_standard_keywords):

    # stem all of the keywords
    pass

    # compute number of terms (i.e. size) of union and intersection
    union_sizes=[len(list(Counter(algorithm_keywords[x])|Counter(gold_standard_keywords[x]))) for x in range(len(algorithm_keywords))]
    intersection_sizes=[len(list(Counter(algorithm_keywords[x])&Counter(gold_standard_keywords[x]))) for x in range(len(algorithm_keywords))]
    
    # compute number of terms (i.e. size) of algorithm (retreived) and gold standard (relevant) keywords
    
    algorithm_keyword_length=[len(list(Counter(algorithm_keywords[x]))) for x in range(len(algorithm_keywords))]
    gold_standard_keyword_length=[len(list(Counter(gold_standard_keywords[x]))) for x in range(len(algorithm_keywords))]
    
    # compute Jaccard index
    jaccard_index_vector=np.asarray(intersection_sizes).astype('float')/np.asarray(union_sizes).astype('float')
    mean_jaccard=np.mean(jaccard_index_vector)

    # compute precision and recall
    precision_vector=np.asarray(intersection_sizes).astype('float')/np.asarray(algorithm_keyword_length).astype('float')
    recall_vector=np.asarray(intersection_sizes).astype('float')/np.asarray(gold_standard_keyword_length).astype('float')
    mean_precision=np.mean(precision_vector)
    mean_recall=np.mean(recall_vector)
    
    return mean_jaccard,mean_precision,mean_recall
    
def getKeywordsByTextRank(text_list, num_keywords_to_generate_array):
    
    keywords_by_TextRank=[[]]*len(text_list)

    #for doc_num in range(len(text_list)):
    for doc_num in [75]:
        print doc_num
        tokenized_and_stemmed=' '.join(tokenizeAndStemStrings(text_list[doc_num]))
        kw=keywords.keywords(tokenized_and_stemmed)
        keywords_by_TextRank[doc_num]=keywords.keywords(tokenized_and_stemmed).split('\n')[0:num_keywords_to_generate_array[doc_num]]
    
    return keywords_by_TextRank
    
if __name__ == '__main__':
    # references: this project was inspired by:
    #    1) http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/
    #    2) Lab #2 and Lecture #2 from https://sites.duke.edu/compsci290_01_s2014/schedule/
    #    3) Introduction to Information Retrieval, https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html

    # read text and keywords from hard drive
    text_list,keyword_list,num_keywords_list=getTextAndKeywords('/home/ryan/Dropbox/Code/Data/FierceBiotech')
    
    # generate tfidf matrix
    # tfs,tfidf = createTfidfTable(text_list)
    
    # generate keywords constrained by the below regular expressions.  The below keyword grammar corresponds to
    # a noun preceeded by any number of adjectives, the whole of which may or may not be preceeded by a
    # preposition + noun + adjective phrase
    regexps_list,num_regexp_list = getRegExps(text_list,'KW: {(<JJ>* <NN.*>+<IN>)? <JJ>* <NN.*>+}')
    
    # determine how many keywords should be generated for each document.  The average number of keywords
    # generated by the algorim will equal the average number of keywords provided in the document.  The
    # number of keywords generated by the algoritm will be proportional to file size.
    num_keywords_to_generate_array=(np.around(np.asarray(num_regexp_list)*np.mean(np.asarray(num_keywords_list))/np.mean(np.asarray(num_regexp_list)))).astype(int)
      
    # keywords by count
    # keyword_by_count_list = getKeywordsByCount(regexps_list, num_keywords_to_generate_array)
    # mean_jaccard,mean_precision,mean_recall=keywordExtractionPerformance(keyword_by_count_list,keyword_list)
    # print "Weighting-by-count performance was " + str(mean_jaccard) + ", " + str(mean_precision) + ", " + str(mean_recall) + "(Jaccard index, precision, recall)"
    
    # generate keywords by searching for regular expressions, and weight the keywords by tfidf
    # keyword_by_tfidf_list = getKeywordsByTfidf(regexps_list, num_keywords_to_generate_array,tfidf)
    # mean_jaccard,mean_precision,mean_recall=keywordExtractionPerformance(keyword_by_tfidf_list,keyword_list)
    # print "Weighting-by-tfidf performance was " + str(mean_jaccard) + ", " + str(mean_precision) + ", " + str(mean_recall) + "(Jaccard index, precision, recall)"
    
    # generate keywords using TextRank (TextRank is PageRank for words instead of webpages)
    keyword_by_TextRank_list = getKeywordsByTextRank(text_list, num_keywords_to_generate_array)
    mean_jaccard,mean_precision,mean_recall=keywordExtractionPerformance(keyword_by_TextRank_list,keyword_list)
    print "textrank performance was " + str(mean_jaccard) + ", " + str(mean_precision) + ", " + str(mean_recall) + "(Jaccard index, precision, recall)"
    
    #keyword_by_TextRank = [str(keywords.keywords(text_list[doc_num])).split() for doc_num in range(len(text_list))]
    
    # generate keywords using a supervised method: A Ranking Approach to Keyphrase Extraction
    
    #query_term='chutes'
    #response = tfidf.transform([query_term])
    #doc_keys = tuple([(text_dict.keys()[x],tfs[x].dot(response.T).todense()[0,0]) for x in range(len(text_dict)) if tfs[x].dot(response.T) > 0])

    print "test"
   