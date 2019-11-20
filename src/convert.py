import pandas as pd
import pickle
import numpy as np

# Read in the file and create sequence list and dna list
def read_file(file_name):
    sequence_list = []
    dna_list = []
    dna_string_list  = []
    with open(file_name, 'r') as f:
        count = 0
        for i in f:
            if count % 2 == 0:
                sequence_list.append(i[1:-1])
            if count % 2 == 1:
                dna_list.append(list(i[:-1]))
                dna_string_list.append(i[:-1])
            count += 1             

    return sequence_list, dna_list, dna_string_list

# create data frame and convert the strings to numbers
def create_df(dna_list):
    dna_df = pd.DataFrame(dna_list)   
    dna_df.to_csv('../HW3STRINGS.csv')
    DNA = {'A': 1, 'C': 2, 'G' : 3, 'T':4 }
    dna_df.replace(to_replace=DNA, inplace=True)    
    return dna_df
    
#caluclate hamming distance
def hamming_df(dna_string_list):
    
    n = len(dna_string_list)
    
    main_dist_list = []
    for i in range(n):
        main_dist_list.append([0]*n)
    
    for i in range(n):
        for j in range(i+1,n):
           ham_dist = hamdist(dna_string_list[i],dna_string_list[j])
           main_dist_list[i][j] = ham_dist
           main_dist_list[j][i] = ham_dist
    hamming_df = pd.DataFrame(main_dist_list)
    return hamming_df
    

def hamdist(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs

# save the data frame and the sequence_list using pickle
def save_df(df, ham_df, sequence_list):
    df.to_csv('../HW3.csv')
    ham_df.to_csv('../HW3_ham.csv')
    with open('seq_list.pkl','wb') as f:
        pickle.dump(sequence_list, f)

def main():
    file_name = "../HW4.fas"
    seq_list, dna_list, dna_string_list = read_file(file_name)
    df = create_df(dna_list)
    ham_df = hamming_df(dna_string_list)
    save_df(df, ham_df, seq_list)

if __name__ == "__main__":
    main()