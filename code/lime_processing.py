import pandas as pd
from nltk.metrics import agreement

path = '../data/Labels/'
#load a given survey part
def load_survey(file):
    full_path = path + file
    df = pd.read_csv(full_path)
    
    #transpose df so that each row contains an answer
    df = df.T.reset_index()
    return df

#process the lime answers and change them to numeric labels
def process_labels(df):
    df.columns.values[0] = 'Question'
    df.columns.values[1] = 'Contains'
    df['Motive'] = ''
    
    #0: None, 1: Anti-Elitism, 2: People-Centrism, 3: People-Sovereignty
    df['Motive'].loc[df['Question'].str.contains('[None]',regex=False)] = 0
    df['Motive'].loc[df['Question'].str.contains('[Anti-Elitism]',regex=False)] = 1
    df['Motive'].loc[df['Question'].str.contains('[People-Centrism]',regex=False)] = 2
    df['Motive'].loc[df['Question'].str.contains('[People-Sovereignty]',regex=False)] = 3
    return df

#only keep one row per question that one hot encodes all labels and adds a populism label that is 1 if one of the motives was selected
def create_pop_df(df):
    final = pd.DataFrame(columns=['Question','Anti-Elitism','People-Centrism','People-Sovereignty','None','Populism'])
    for i in range(200):
        temp = df[:4]
        final = pd.concat([final, pd.DataFrame({'Question':[temp['Question'].iloc[0]],'Anti-Elitism': [0],'People-Centrism': [0],'People-Sovereignty': [0],'None': [0],'Populism': [0]})])
        for j in range(4):
            if temp['Contains'].iloc[j] == 'Yes' and temp['Motive'].iloc[j] == 0:
                final['None'].iloc[i] = 1
            if temp['Contains'].iloc[j] == 'Yes' and temp['Motive'].iloc[j] == 1:
                final['Anti-Elitism'].iloc[i] = 1
            if temp['Contains'].iloc[j] == 'Yes' and temp['Motive'].iloc[j] == 2:
                final['People-Centrism'].iloc[i] = 1
            if temp['Contains'].iloc[j] == 'Yes' and temp['Motive'].iloc[j] == 3:
                final['People-Sovereignty'].iloc[i] = 1
        if final['None'].iloc[i] == 0:
            final['Populism'].iloc[i] = 1
        df = df[4:]
    return final

#return occurence stats for populist motives 
def pop_stats(df):
    print(f"The participant labelled {sum(df['Anti-Elitism'])} as Anti-Elitism")
    print(f"The participant labelled {sum(df['People-Centrism'])} as People-Centrism")
    print(f"The participant labelled {sum(df['People-Sovereignty'])} as People-Sovereignty")
    print(f"The participant labelled {sum(df['Populism'])} as populist in total")

#get amount of populism labels only
def pop_stat(df):
    if 'Populism' in df:
        print(f"There are {sum(df['Populism'])} populist comments in total")
    else:
        print(f"There are {sum(df['Final Label'])} populist comments in total")

#extract gold labels for different parts (example comments to measure the quality of the instructions)
def extract_gold(df, part):
    if part == 1:
        gold = df.iloc[[148,149,150,151,532,533,534,535]]
        df = df.drop([148,149,150,151,532,533,534,535])
    if part == 2:
        gold = df.iloc[[712,713,714,715]]
        df = df.drop([712,713,714,715])     
    if part == 3:
        gold = df.iloc[[316,317,318,319]]
        df = df.drop([316,317,318,319])
    if part == 4:
        gold = df.iloc[[56,57,58,59,588,589,590,591]]
        df= df.drop([56,57,58,59,588,589,590,591])
    if part == 5:
        gold = df.iloc[[368,369,370,371,500,501,502,503]]
        df= df.drop([368,369,370,371,500,501,502,503])
        
    if part == 6:
        gold = df.iloc[[536,537,538,539]]
        df= df.drop([536,537,538,539])
        
    return df, gold 

#get krippendorff's alpha for two participants
def alpha2(df1, df2):
    anti =[[0,str(i),str(list(df1['Anti-Elitism'])[i])] for i in range(0,len(df1))]+[[1,str(i),str(list(df2['Anti-Elitism'])[i])] for i in range(0,len(df2))]
    anti_agg = agreement.AnnotationTask(data=anti)
    anti_alpha = anti_agg.alpha()

    cent =[[0,str(i),str(list(df1['People-Centrism'])[i])] for i in range(0,len(df1))]+[[1,str(i),str(list(df2['People-Centrism'])[i])] for i in range(0,len(df2))]
    cent_agg = agreement.AnnotationTask(data=cent)
    cent_alpha = cent_agg.alpha()

    sov =[[0,str(i),str(list(df1['People-Sovereignty'])[i])] for i in range(0,len(df1))]+[[1,str(i),str(list(df2['People-Sovereignty'])[i])] for i in range(0,len(df2))]
    sov_agg = agreement.AnnotationTask(data=sov)
    sov_alpha = sov_agg.alpha()

    pop =[[0,str(i),str(list(df1['Populism'])[i])] for i in range(0,len(df1))]+[[1,str(i),str(list(df2['Populism'])[i])] for i in range(0,len(df2))]
    pop_agg = agreement.AnnotationTask(data=pop)
    pop_alpha = pop_agg.alpha()

    print(f'The agreement on Anti-Elitism between these two participants has an alpha value of:{anti_alpha}')
    print(f'The agreement on People-Centrism between these two participants has an alpha value of:{cent_alpha}')
    print(f'The agreement on People-Sovereignty between these two participants has an alpha value of:{sov_alpha}')
    print(f'The total agreement between these two participants has an alpha value of:{pop_alpha}')

#get krippendorff's alpha for three participants
def alpha2(df1, df2,df3):
    anti =[[0,str(i),str(list(df1['Anti-Elitism'])[i])] for i in range(0,len(df1))]+[[1,str(i),str(list(df2['Anti-Elitism'])[i])] for i in range(0,len(df2))]+[[1,str(i),str(list(df3['Anti-Elitism'])[i])] for i in range(0,len(df3))]
    anti_agg = agreement.AnnotationTask(data=anti)
    anti_alpha = anti_agg.alpha()

    cent =[[0,str(i),str(list(df1['People-Centrism'])[i])] for i in range(0,len(df1))]+[[1,str(i),str(list(df2['People-Centrism'])[i])] for i in range(0,len(df2))]+[[1,str(i),str(list(df3['People-Centrism'])[i])] for i in range(0,len(df3))]
    cent_agg = agreement.AnnotationTask(data=cent)
    cent_alpha = cent_agg.alpha()

    sov =[[0,str(i),str(list(df1['People-Sovereignty'])[i])] for i in range(0,len(df1))]+[[1,str(i),str(list(df2['People-Sovereignty'])[i])] for i in range(0,len(df2))]+[[1,str(i),str(list(df3['People-Sovereignty'])[i])] for i in range(0,len(df3))]
    sov_agg = agreement.AnnotationTask(data=sov)
    sov_alpha = sov_agg.alpha()

    pop =[[0,str(i),str(list(df1['Populism'])[i])] for i in range(0,len(df1))]+[[1,str(i),str(list(df2['Populism'])[i])] for i in range(0,len(df2))]+[[1,str(i),str(list(df3['Populism'])[i])] for i in range(0,len(df3))]
    pop_agg = agreement.AnnotationTask(data=pop)
    pop_alpha = pop_agg.alpha()

    print(f'The agreement on Anti-Elitism between the participants has an alpha value of:{anti_alpha}')
    print(f'The agreement on People-Centrism between the participants has an alpha value of:{cent_alpha}')
    print(f'The agreement on People-Sovereignty between the has an alpha value of:{sov_alpha}')
    print(f'The total agreement between the participants has an alpha value of:{pop_alpha}')
   

#create the final data set by majority vote
def majority(df1,df2,df3):
    df1['Pop2'] = df2['Populism']
    df1['Pop3'] = df3['Populism']

    df1['Final Label'] = df1['Populism'] + df1['Pop2'] + df1['Pop3']
    df1['Final Label'].loc[df1['Final Label']<2] = 0 
    df1['Final Label'].loc[df1['Final Label']>1] = 1

    return df1[['Question','Final Label']]

#assign label by majority vote for all motives
def majority_all(df1,df2,df3):
    df1['anti2'] = df2['Anti-Elitism']
    df1['anti3'] = df3['Anti-Elitism']
    df1['cent2'] = df2['People-Centrism']
    df1['cent3'] = df3['People-Centrism']
    df1['sov2'] = df2['People-Sovereignty']
    df1['sov3'] = df3['People-Sovereignty']

    df1['Final Anti'] = df1['Anti-Elitism'] + df1['anti2'] + df1['anti3']
    df1['Final Anti'].loc[df1['Final Anti']<2] = 0 
    df1['Final Anti'].loc[df1['Final Anti']>1] = 1

    df1['Final Cent'] = df1['People-Centrism'] + df1['cent2'] + df1['cent3']
    df1['Final Cent'].loc[df1['Final Cent']<2] = 0 
    df1['Final Cent'].loc[df1['Final Cent']>1] = 1

    df1['Final Sov'] = df1['People-Sovereignty'] + df1['sov2'] + df1['sov3']
    df1['Final Sov'].loc[df1['Final Sov']<2] = 0 
    df1['Final Sov'].loc[df1['Final Sov']>1] = 1


    return df1[['Question','Final Anti', 'Final Cent', 'Final Sov']]

#calculate a data frame that holds all labels for each motives by each participant
def all_labels(df1,df2,df3):
    df1['none2'] = df2['None']
    df1['none3'] = df3['None']
    df1['anti2'] = df2['Anti-Elitism']
    df1['anti3'] = df3['Anti-Elitism']
    df1['cent2'] = df2['People-Centrism']
    df1['cent3'] = df3['People-Centrism']
    df1['sov2'] = df2['People-Sovereignty']
    df1['sov3'] = df3['People-Sovereignty']
    df1 = df1.rename(columns={'Anti-Elitism': 'anti1'})
    df1 = df1.rename(columns={'People-Centrism': 'cent1'})
    df1 = df1.rename(columns={'People-Sovereignty': 'sov1'})
    df1 = df1.rename(columns={'None': 'none1'})
    df1 = df1[['Question', 'anti1','anti2', 'anti3', 'cent1', 'cent2', 'cent3', 'sov1', 'sov2', 'sov3','none1',  'none2', 'none3']]
    return df1.reset_index(drop=True)