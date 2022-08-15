# -*- coding:utf-8 -*-

import re
import numpy as np
import pandas as pd
import collections
from nltk.stem import PorterStemmer
from tqdm import tqdm


def filter_function(str):
    if str=='' or str=='\n':
        return False
    else:
        return True


def get_Vocabulary(dataframe):
    fake_voc= collections.defaultdict(int)
    real_voc= collections.defaultdict(int)
    fake_name_voc = collections.defaultdict(int)
    real_name_voc = collections.defaultdict(int)

    num_total=len(dataframe)

    for i in range(num_total):
        voc=fake_voc if dataframe['label'][i] else real_voc

        ## build vocabulary based on text and title
        text=dataframe['title'][i]+' '+dataframe['text'][i]
        text=text.lower()
        text=list(filter(filter_function, re.split('[^a-zA-Z]', text)))
        text = [ps.stem(word) for word in text]

        for word in text:
            voc[word]+=1

        ## build name vocabulary based on author's name
        name_voc = fake_name_voc if dataframe['label'][i] else real_name_voc
        name = dataframe['author'][i]
        name=name.lower()
        name_voc[name]+=1

    return fake_voc, real_voc,fake_name_voc, real_name_voc

def Text_Preprocessing(filename):
    text=[]
    with open(filename, 'r', encoding='latin-1') as f1:
        for line in f1.readlines():
            line = line.lower()
            line = list(filter(filter_function, re.split('[^a-zA-Z]', line)))
            text+=line
    text=[ps.stem(word) for word in text]
    f1.close()

    return text

# Building and evaluating a Naive Bayes Classifier
def Naive_Bayes_Classifier(text,name, vocabulary_list, name_vocabulary_list, voc_matrix,name_voc_matrix):
    ## prior score
    score1=np.log10(num_real/num_total)
    score2=np.log10(num_fake/num_total)

    # name score
    if name in name_vocabulary_list:
        index = name_vocabulary_list.index(name)
        score1 += np.log10(name_voc_matrix[index][1])
        score2 += np.log10(name_voc_matrix[index][3])

    # vocabulary score
    for word in text:
        if word in vocabulary_list:
            index=vocabulary_list.index(word)
            score1+=np.log10(voc_matrix[index][1])
            score2+=np.log10(voc_matrix[index][3])

    return 0 if score1>score2 else 1


def get_PMatrix(vocabulary_list,fake_voc, real_voc,sm=0.5):
    # get probability Matrix of real and fake:
    voc_num = len(vocabulary_list)
    voc_maxtix = np.zeros((voc_num, 4), dtype=float)
    for i in range(voc_num):
        word = vocabulary_list[i]
        if word in real_voc:
            voc_maxtix[i][0] = real_voc[word]
        else:
            voc_maxtix[i][0] = 0
        if word in fake_voc:
            voc_maxtix[i][2] = fake_voc[word]
        else:
            voc_maxtix[i][2] = 0
    wn_real = np.sum(voc_maxtix,axis=0)[0]
    print("wn_real:",wn_real)
    wn_fake = np.sum(voc_maxtix,axis=0)[2]
    print("wn_fake:", wn_fake)
    for i in range(voc_num):
        voc_maxtix[i][1] = (voc_maxtix[i][0] + sm) / (wn_real + sm* voc_num)
        voc_maxtix[i][3] = (voc_maxtix[i][2] + sm) / (wn_fake + sm* voc_num)
    return voc_maxtix


def Model_test(test_data,test_label,vocabulary_list, name_vocabulary_list, voc_matrix, name_voc_matrix):
    right_cases = 0
    T_real,F_real,T_fake,F_fake=0,0,0,0
    real_test, fake_test =sum(test_label['label'][:20]==0), sum(test_label['label'][:20]==1)

    num_test=len(test_data)
    for i in tqdm(range(num_test)[:20]):
        ##preprocessing of name
        name = test_data['author'][i]
        name= name.lower()

        ##preprocessing of title+ text
        text = test_data['title'][i] + ' ' + test_data['text'][i]
        text = text.lower()
        text = list(filter(filter_function, re.split('[^a-zA-Z]', text)))

        cls = Naive_Bayes_Classifier(text, name, vocabulary_list,name_vocabulary_list, voc_matrix, name_voc_matrix)

        if cls == test_label['label'][i]:
            ## result = 'right'
            right_cases += 1
            if cls==1:  ## predictive result: fake
                T_fake += 1
            else:     ## predictive result: real
                T_real += 1
        else:
            ## result = 'wrong'
            if cls==1: ## predictive result: fake
                F_fake += 1
            else:
                F_real += 1

    accuracy = right_cases / (real_test + fake_test)
    print("Accuracy:{},T_real:{},F_real:{},T_fake:{},F_fake:{}".format(accuracy, T_real, F_real, T_fake, F_fake))

    p_real = T_real / (T_real + F_real)
    p_fake = T_fake / (T_fake + F_fake)
    r_real = T_real / real_test
    r_fake = T_fake / fake_test
    f1_real = 2 * p_real * r_real / (p_real + r_real)
    f1_fake = 2 * p_fake * r_fake / (p_fake + r_fake)
    return  accuracy,p_real,r_real,f1_real,p_fake,r_fake,f1_fake

if __name__ == '__main__':
    ## data loading
    root='C:/Users/sjun1/PycharmProjects/Natrual Language Processing/authenticity_checking'
    train_path = root+"/data/train.csv"
    test_data_path = root+"/data/test.csv"
    test_label_path=root+"/data/labels.csv"

    train_data = pd.read_csv(train_path,keep_default_na=False)
    test_data =  pd.read_csv(test_data_path,keep_default_na=False)
    test_label = pd.read_csv(test_label_path, keep_default_na=False)

    ## get number of positve and negative cases
    num_fake = sum(train_data['label'] == 1)
    num_real = sum(train_data['label'] == 0)
    num_total=num_fake +num_real

    ## get vocanulary from the training set
    ps = PorterStemmer()
    fake_voc, real_voc, fake_name_voc,real_name_voc = get_Vocabulary(train_data)
    vocabulary = set(fake_voc.keys()).union(real_voc.keys())
    print("Length of original voc:", len(vocabulary))

    name_vocabulary=set(fake_name_voc.keys()).union(real_name_voc.keys())

    while True:
        expm = input("Input the No. of experiment need to implement(1 or 2):")
        if expm not in ['1', '2']:
            print("Wrong input,please try again. Ipnut the No. of experiment( 1 or 2):")
        else:
            expm = int(expm)
            break

    ## Experiment 1: baseline#########################
    if expm==1:
        model_name = 'model.txt'
        result_name = 'baseline-result.txt'
    #
    ## Experiment 2: stop_word filtering and wordlength filtering ##############
    elif expm==2:
        filename_stopw=root+'/data/stopwords.txt'
        stop_word=[]
        with open(filename_stopw,'r') as file1:
            for line in file1.readlines():
                stop_word.append(line[:-1])
            file1.close()

        vocabulary=[x for x in vocabulary if x not in stop_word and len(x)>2 and len(x)<9]

        model_name = 'wordlength-model.txt'
        result_name = 'wordlength-result.txt'

    if expm in [1,2]:
        vocabulary_list = sorted(vocabulary, key=lambda item: item, reverse=False)
        print("vocabulary length:",len(vocabulary_list))
        voc_matrix=get_PMatrix(vocabulary_list, fake_voc, real_voc)

        name_vocabulary_list= sorted(list(name_vocabulary))
        print("name vocabulary length:", len(name_vocabulary_list))
        name_voc_matrix=get_PMatrix(name_vocabulary_list, fake_name_voc,real_name_voc)


        with open(root+'/result/'+ model_name,'w',encoding='latin-1') as file:
            for i in range(len(vocabulary_list)):
                file.write("{}  {}  {}  {}  {}  {}\n".format(i + 1, vocabulary_list[i], int(voc_matrix[i][0]), voc_matrix[i][1], int(voc_matrix[i][2]), voc_matrix[i][3]))
            file.close()

        # Test the model with test data
        accuracy,p_real, r_real, f1_real,p_fake, r_fake,f1_fake = Model_test(test_data, test_label, vocabulary_list,name_vocabulary_list, voc_matrix, name_voc_matrix)

        print("p_real:{0:.3f},r_real:{1:.3f},f1_real:{2:.3f},p_fake:{3:.3f},r_fake:{4:.3f},f1_fake:{5:.3f}".format(p_real, r_real, f1_real,p_fake, r_fake,f1_fake))

