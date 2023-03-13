# This script is aim to solve Kaggle 2023 GarantiBBVA Datacamp case. There are 7 different files contains;
# Train, Test and Example_Submission files,
# Education, Language, Skill and Work_experience about customers.

# Import Libaries
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import datetime as dt
from helpers import hyp_op, OHE_partial
from helpers import one_hot_encoder, base_models, hyperparameter_optimization, voting_classifier, check_df
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier

from helpers import plot_importance
## Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

pd.set_option('display.width', 10000)
pd.set_option('display.max_rows',1000)
# Import Files
df_train = pd.read_csv('/Users/buraksayilar/Desktop/garanti-bbva-data-camp/train_users.csv')
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_test =  pd.read_csv('/Users/buraksayilar/Desktop/garanti-bbva-data-camp/test_users.csv')
df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_education = pd.read_csv('/Users/buraksayilar/Desktop/garanti-bbva-data-camp/education.csv')
df_education = df_education.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_skill = pd.read_csv('/Users/buraksayilar/Desktop/garanti-bbva-data-camp/skills.csv')
df_skill = df_skill.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_lng = pd.read_csv('/Users/buraksayilar/Desktop/garanti-bbva-data-camp/languages.csv')
df_lng = df_lng.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_workex = pd.read_csv('/Users/buraksayilar/Desktop/garanti-bbva-data-camp/work_experiences.csv')
df_workex = df_workex.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_sub =pd.read_csv('/Users/buraksayilar/Desktop/garanti-bbva-data-camp/submission.csv')


check_df(df_train)
check_df(df_test)
check_df(df_education) # Only valuable info is the school_name. Rest is useless.
check_df(df_skill)
check_df(df_lng) # One_hot_encoder for language, label encoder for proficiency.
check_df(df_workex) # I probably drop location, too much NAs.

# VLOOKUP Function to bring related futures according to user_id.
def VLOOKUP(dataframe1, dataframe2, key, wanted_column_list):
    # Filter dataframe2 to only include rows where key is in dataframe1
    # I already filtered train and test datasets in 'merged-df' part, so this block
    # is not necessary.
    #filtered_df2 = dataframe2[dataframe2[key].isin(dataframe1[key])][wanted_column_list]
    #print("filtered_df2 shape:", filtered_df2.shape)
    #print("filtered_df2 columns:", filtered_df2.columns)

    # Merge filtered_df2 with dataframe1 on the key column
    merged_df = pd.merge(dataframe1, dataframe2, on=key)
    print("merged_df shape:", merged_df.shape)
    print("merged_df columns:", merged_df.columns)
    return merged_df


## EDUCATION ###########################################################################################################
df_education['degree'] = df_education['degree'].astype('str')
df_education['degree'].fillna(0, inplace=True)
# We create 4 features for degrees.
for i in range(len(df_education)):
    if 'Yüksek' in df_education.loc[i, 'degree'] or 'Yuksek' in df_education.loc[i, 'degree'] or 'Master' in df_education.loc[i, 'degree']:
        df_education.loc[i, 'MASTER'] = 1
    elif 'Doctor' in df_education.loc[i, 'degree'] or 'PhD' in df_education.loc[i, 'degree'] or 'Philosophy' in df_education.loc[i, 'degree']:
        df_education.loc[i, 'DOCTOR'] = 1
    elif 'Bachelor' in df_education.loc[i, 'degree'] or 'Lisans Derecesi' in df_education.loc[i, 'degree'] or 'Mühendislik' in df_education.loc[i, 'degree'] or 'B.S' in df_education.loc[i, 'degree'] or 'Lisans' in df_education.loc[i, 'degree']:
        df_education.loc[i, 'BACHELOR'] = 1
    elif (((df_education.loc[i,'degree'] == '0') | (df_education.loc[i,'degree'] == 'nan')) & ('Üniversite' in df_education.loc[i,'school_name'] or 'University' in df_education.loc[i,'school_name'])):
        df_education.loc[i, 'BACHELOR'] = 1
    else:
        df_education.loc[i, 'OTHER'] = 1
# Reputable university
for i in range(len(df_education)):
    if 'Bogazici' in df_education.loc[i, 'school_name'] or \
            'Istanbul Technical' in df_education.loc[i, 'school_name'] or \
            'İstanbul Teknik' in df_education.loc[i, 'school_name'] or \
            'Middle East Technical' in df_education.loc[i, 'school_name'] or \
            'Yıldız Te' in df_education.loc[i, 'school_name'] or \
            'Yildiz Te' in df_education.loc[i, 'school_name']:
        df_education.loc[i, 'REPUTABLE'] = 1

# More then %80 is NaN in start and end dates, so we dropped them.
df_education.drop(['start_year_month', 'end_year_month'], axis=1, inplace=True)

# There is a problem; user_id is unique in train and test sets but in Education set, there are
# multiple rows for different users :)
# For Train Dataset
education_df = pd.DataFrame(columns=['user_id'])
education_df['user_id'] = df_train['user_id']
merged_df_train = education_df.merge(df_education, on='user_id', how='left')
merged_df_train = merged_df_train.groupby('user_id').sum().reset_index()
# For Test Dataset
education_df = pd.DataFrame(columns=['user_id'])
education_df['user_id'] = df_test['user_id']
merged_df_test = education_df.merge(df_education, on='user_id', how='left')
merged_df_test = merged_df_test.groupby('user_id').sum().reset_index()
### Train & Test Concat for Education
dfc_train = VLOOKUP(df_train, merged_df_train, 'user_id', ['REPUTABLE', 'BACHELOR', 'MASTER','OTHER' ,'DOCTOR' ])
dfc_test = VLOOKUP(df_test, merged_df_test, 'user_id', ['REPUTABLE', 'BACHELOR', 'MASTER','OTHER' ,'DOCTOR' ])





# So, now we have main train & test set infos, plus usable Education info.
### SKILL ##############################################################################################################
df_skill_list = df_skill['skill'].value_counts().sort_values(ascending=False)
OHE_partial(df_skill, 'skill', encode_feature_list= [], valuecount_select=True, select_number=70)
df_skill = one_hot_encoder(df_skill, ['encoder_subset'])
df_skill_list = df_skill_list[df_skill_list > 1000].iloc[:60].index

df_skill['skill'].value_counts().head(60)

df_skill = df_skill[df_skill['skill'].isin(df_skill_list)]
df_skill = one_hot_encoder(df_skill, ['skill'], drop_first=True)

skill_df = pd.DataFrame(columns=['user_id'])
skill_df['user_id'] = df_train['user_id']
merged_df_train = skill_df.merge(df_skill, on='user_id', how='left')
merged_df_train = merged_df_train.groupby('user_id').sum().reset_index()
# For Test Dataset
skill_df = pd.DataFrame(columns=['user_id'])
skill_df['user_id'] = df_test['user_id']
merged_df_test = skill_df.merge(df_skill, on='user_id', how='left')
merged_df_test = merged_df_test.groupby('user_id').sum().reset_index()
skill_columns_list = ['skill_AJAX', 'skill_ASP.NET', 'skill_ASP.NET MVC', 'skill_Agile Methodologies', 'skill_Agile Metotları', 'skill_Algorithms', 'skill_Android', 'skill_Android Development', 'skill_Angular', 'skill_AngularJS', 'skill_Araştırma', 'skill_AutoCAD', 'skill_Bootstrap', 'skill_C', 'skill_C#', 'skill_C++', 'skill_CSS', 'skill_Cascading Style Sheets (CSS)', 'skill_Databases', 'skill_Design Patterns', 'skill_Docker', 'skill_Eclipse', 'skill_Ekip Çalışması', 'skill_Engineering', 'skill_English', 'skill_Entity Framework', 'skill_Git', 'skill_HTML', 'skill_HTML5', 'skill_Hibernate', 'skill_Integration', 'skill_JIRA', 'skill_JSON', 'skill_Java', 'skill_JavaScript', 'skill_Jenkins', 'skill_LINQ', 'skill_Liderlik', 'skill_Linux', 'skill_Machine Learning', 'skill_Management', 'skill_Matlab', 'skill_Maven', 'skill_Microsoft Excel', 'skill_Microsoft Office', 'skill_Microsoft SQL Server', 'skill_Microsoft Word', 'skill_MongoDB', 'skill_MySQL', 'skill_Mühendislik', 'skill_Node.js', 'skill_OOP', 'skill_OOP (Nesne Yönelimli Programlama)', 'skill_Object Oriented Design', 'skill_Object-Oriented Programming (OOP)', 'skill_Oracle', 'skill_PHP', 'skill_PL/SQL', 'skill_Photoshop', 'skill_PostgreSQL', 'skill_PowerPoint', 'skill_Programlama', 'skill_Programming', 'skill_Proje Yönetimi', 'skill_Project Management', 'skill_Python', 'skill_Python (Programming Language)', 'skill_React Native', 'skill_React.js', 'skill_Redis', 'skill_Research', 'skill_SOAP', 'skill_SQL', 'skill_Scrum', 'skill_Software Design', 'skill_Software Development', 'skill_Software Engineering', 'skill_Software Project Management', 'skill_Spring Boot', 'skill_Spring Framework', 'skill_T-SQL', 'skill_TCP/IP', 'skill_TFS', 'skill_Teamwork', 'skill_Telecommunications', 'skill_UML', 'skill_Veri Analizi', 'skill_Visual Studio', 'skill_WCF', 'skill_Web Applications', 'skill_Web Development', 'skill_Web Geliştirme', 'skill_Web Services', 'skill_WordPress', 'skill_XML', 'skill_Yazılım Geliştirme', 'skill_Yönetim', 'skill_jQuery', 'skill_İngilizce']
# There is a difference between skill columns (not all skill columns are available in test set), so I took skills only in test set to prevent
# inconvinience in model building/training.
### Train & Test Concat for Education
dfc_train = VLOOKUP(dfc_train, merged_df_train, 'user_id', skill_columns_list)
dfc_test = VLOOKUP(dfc_test, merged_df_test, 'user_id', skill_columns_list)





### LANGUAGE ###########################################################################################################
#check_df(df_lng)
df_lng['language'] = df_lng['language'].astype('str')
#df_lng['degree'].fillna(0, inplace=True)
df_lng['proficiency'].fillna('elementary', inplace=True)
### Elementary proficiecny is spelled wrong. We fix that.
df_lng['proficiency'].replace('elemantary', 'elementary', inplace=True)

#df_lng[df_lng['proficency'] == np.nan]
#df_lng[(df_lng['language'] == 'Türkçe') & (df_lng['proficiency'] == np.nan)]
#df_lng[(df_lng['language'] == 'Türkçe')]
# Creating Class for languages
for i in range(len(df_lng)):
    if 'English' in df_lng.loc[i, 'language'] or \
            'İngilizce' in df_lng.loc[i, 'language'] or \
            'ingiliz' in df_lng.loc[i, 'language'] or \
            'english' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'ENGLISH'] = 1
    elif 'Almanca' in df_lng.loc[i, 'language'] or \
            'German' in df_lng.loc[i, 'language'] or \
            'alman' in df_lng.loc[i, 'language'] or \
            'german' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'GERMAN'] = 1
    elif 'Italiano' in df_lng.loc[i, 'language'] or \
            'İtalyanca' in df_lng.loc[i, 'language'] or \
            'italyan' in df_lng.loc[i, 'language'] or \
            'italiano' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'ITALIANO'] = 1
    elif 'Español' in df_lng.loc[i, 'language'] or \
            'İspanyolca' in df_lng.loc[i, 'language'] or \
            'ispanyol' in df_lng.loc[i, 'language'] or \
            'español' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'ESPANOL'] = 1
    elif 'Français' in df_lng.loc[i, 'language'] or \
            'Fransızca' in df_lng.loc[i, 'language'] or \
            'fransız' in df_lng.loc[i, 'language'] or \
            'français' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'FRANCAIS'] = 1
    elif 'Russain' in df_lng.loc[i, 'language'] or \
            'Rusça' in df_lng.loc[i, 'language'] or \
            'rusça' in df_lng.loc[i, 'language'] or \
            'russian' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'RUSSIAN'] = 1
    elif 'Chinese' in df_lng.loc[i, 'language'] or \
            'Çince' in df_lng.loc[i, 'language'] or \
            'çince' in df_lng.loc[i, 'language'] or \
            'chinese' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'CHINESE'] = 1
    elif 'Arabic' in df_lng.loc[i, 'language'] or \
            'Arapça' in df_lng.loc[i, 'language'] or \
            'Arabish' in df_lng.loc[i, 'language'] or \
            'arab' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'ARABIC'] = 1
    elif 'Turkish' in df_lng.loc[i, 'language'] or \
            'Türkçe' in df_lng.loc[i, 'language'] or \
            'Turkce' in df_lng.loc[i, 'language'] or \
            'türk' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'TURKISH'] = 1
    elif ((df_lng.loc[i, 'language'] != np.nan) & \
          (df_lng.loc[i, 'language'] != 'Turkish') & \
          (df_lng.loc[i, 'language'] != 'Turkish')):
        df_lng.loc[i,'OTHER_LNG'] = 1
for i in range(len(df_lng)):
    if 'English' in df_lng.loc[i, 'language'] or \
            'İngilizce' in df_lng.loc[i, 'language'] or \
            'ingiliz' in df_lng.loc[i, 'language'] or \
            'english' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'ENGLISH'] = 1
    elif 'Almanca' in df_lng.loc[i, 'language'] or \
            'German' in df_lng.loc[i, 'language'] or \
            'alman' in df_lng.loc[i, 'language'] or \
            'german' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'GERMAN'] = 1
    elif 'Español' in df_lng.loc[i, 'language'] or \
            'İspanyolca' in df_lng.loc[i, 'language'] or \
            'ispanyol' in df_lng.loc[i, 'language'] or \
            'español' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'ESPANOL'] = 1
    elif 'Turkish' in df_lng.loc[i, 'language'] or \
            'Türkçe' in df_lng.loc[i, 'language'] or \
            'Turkce' in df_lng.loc[i, 'language'] or \
            'türk' in df_lng.loc[i, 'language']:
        df_lng.loc[i, 'TURKISH'] = 1
    else:
        df_lng.loc[i,'OTHER_LNG'] = 1

### Ordinaly encoded 'proficiency' feature.
oe = OrdinalEncoder(categories=[['elementary', 'limited_working', 'professional_working', 'full_professional', 'native_or_bilingual']])
prof = df_lng[['proficiency']]
df_lng['PROF'] = oe.fit_transform(prof)


# For Train dataset
lng_df = pd.DataFrame(columns=['user_id'])
lng_df['user_id'] = df_train['user_id']
merged_df_train = lng_df.merge(df_lng, on='user_id', how='left')
merged_df_train = merged_df_train.groupby('user_id').sum().reset_index()
# We will divide total proficiency the total languages.
merged_df_train['PROF'] = (merged_df_train['PROF'] / (merged_df_train['ENGLISH'] +
                                                      merged_df_train['OTHER_LNG'] +
                                                      merged_df_train['FRANCAIS'] +
                                                      merged_df_train['GERMAN'] +
                                                      merged_df_train['ESPANOL'] +
                                                      merged_df_train['ARABIC'] +
                                                      merged_df_train['RUSSIAN'] +
                                                      merged_df_train['ITALIANO'] +
                                                      merged_df_train['CHINESE']))
# For Test dataset
lng_df = pd.DataFrame(columns=['user_id'])
lng_df['user_id'] = df_test['user_id']
merged_df_test = lng_df.merge(df_lng, on='user_id', how='left')
merged_df_test = merged_df_test.groupby('user_id').sum().reset_index()
# We will divide total proficiency the total languages.
merged_df_test['PROF'] = (merged_df_test['PROF'] / (merged_df_test['ENGLISH'] +
                                                      merged_df_test['OTHER_LNG'] +
                                                      merged_df_test['FRANCAIS'] +
                                                      merged_df_test['GERMAN'] +
                                                      merged_df_test['ESPANOL'] +
                                                      merged_df_test['ARABIC'] +
                                                      merged_df_test['RUSSIAN'] +
                                                      merged_df_test['ITALIANO'] +
                                                      merged_df_test['CHINESE']))
### Train & Test Concat for Language
dfc_train = VLOOKUP(dfc_train, merged_df_train, 'user_id', ['TURKISH', 'ENGLISH', 'OTHER_LNG', 'FRANCAIS', 'GERMAN' ,'ESPANOL', 'ARABIC', 'RUSSIAN', 'ITALIANO', 'CHINESE', 'PROF' ])
dfc_test = VLOOKUP(dfc_test, merged_df_test, 'user_id', ['TURKISH', 'ENGLISH', 'OTHER_LNG', 'FRANCAIS', 'GERMAN' ,'ESPANOL', 'ARABIC', 'RUSSIAN', 'ITALIANO', 'CHINESE', 'PROF' ])


### WORK EXPERIENCE ####################################################################################################
df_workex = df_workex[df_workex['start_year_month'] < 201901]
df_workex.reset_index(inplace=True)
df_workex['location'] = df_workex['location'].astype('str')
## Region & Counrty classes
for i in range(len(df_workex)):
    if 'İstanbul' in df_workex.loc[i, 'location'] or \
            'Istanbul' in df_workex.loc[i, 'location']:
        df_workex.loc[i,'ISTANBUL'] = 1
    elif 'İzmir' in df_workex.loc[i, 'location'] or \
            'izmir' in df_workex.loc[i, 'location']:
        df_workex.loc[i,'İZMİR'] = 1
    elif 'Antalya' in df_workex.loc[i, 'location'] or \
            'antalya' in df_workex.loc[i, 'location']:
        df_workex.loc[i,'ANTALYA'] = 1
    elif 'Ankara' in df_workex.loc[i, 'location'] or \
            'ankara' in df_workex.loc[i, 'location']:
        df_workex.loc[i,'ANKARA'] = 1
    elif 'Bursa' in df_workex.loc[i, 'location'] or \
            'bursa' in df_workex.loc[i, 'location']:
        df_workex.loc[i,'ANKARA'] = 1
    elif 'Mugla' in df_workex.loc[i, 'location'] or \
            'muğla' in df_workex.loc[i, 'location']:
        df_workex.loc[i,'ANKARA'] = 1
    elif 'Kocaeli' in df_workex.loc[i, 'location'] or \
            'kocaeli' in df_workex.loc[i, 'location']:
        df_workex.loc[i,'ANKARA'] = 1
    elif 'Turkey' in df_workex.loc[i, 'location'] or \
            'Türkiye' in df_workex.loc[i, 'location']:
        df_workex.loc[i, 'OTHER IN TR'] = 1
    elif 'United' in df_workex.loc[i, 'location'] or \
             'Germany' in df_workex.loc[i, 'location'] or \
             'Almanya' in df_workex.loc[i, 'location'] or \
             'France' in df_workex.loc[i, 'location'] or \
             'Amerika' in df_workex.loc[i, 'location'] or \
             'Birleşik' in df_workex.loc[i, 'location'] or \
             'Francis' in df_workex.loc[i, 'location'] or \
             'Poland' in df_workex.loc[i, 'location'] or \
             'Russia' in df_workex.loc[i, 'location'] or \
             'Rusya' in df_workex.loc[i, 'location'] or \
             'Deutschland' in df_workex.loc[i, 'location'] or \
             'Netherlands' in df_workex.loc[i, 'location'] or \
             'Amsterdam' in df_workex.loc[i, 'location'] or \
             'China' in df_workex.loc[i, 'location'] or \
            'Iran' in df_workex.loc[i, 'location']:
        df_workex.loc[i, 'ABROAD'] = 1


## Company ID Encoding
df_workex_companys = df_workex['company_id'].value_counts().iloc[:20].index
df_workex['company_id_subset'] = df_workex['company_id'].apply(lambda x: x if x in df_workex_companys else 'Other')
for i in range(len(df_workex)):
    if df_workex.loc[i, 'company_id'] in df_workex_companys:
        df_workex.loc[i, 'REPUTABLE_COMP'] = 1
    else:
        df_workex.loc[i, 'REPUTABLE_COMP'] = 0

# create the one hot encoder and fit it on the data
encoder = OneHotEncoder(categories='auto', sparse=False)
df_workex['company_id_subset'] = df_workex['company_id_subset'].astype(str)
encoder.fit_transform(df_workex[['company_id_subset']])
# transform the data using the encoder
encoded_data = encoder.transform(df_workex[['company_id_subset']])
df_workex = one_hot_encoder(df_workex, ['company_id_subset'])




df_workex['start_year_month'] = pd.to_datetime(df_workex['start_year_month'], format='%Y%m')
for i in range(len(df_workex)):
    if (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(200701, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(200712, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2007'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(200801, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(200812, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2008'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(200901, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(200912, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2009'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(200912, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(201012, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2010'] = 1
    elif (df_workex.loc[i, 'start_year_month'] > pd.to_datetime(201101, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(201112, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2011'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(201201, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(201212, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2012'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(201301, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(201312, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2013'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(201401, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(201412, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2014'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(201501, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(201512, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2015'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(201601, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(201612, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2016'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(201701, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(201712, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2017'] = 1
    elif (df_workex.loc[i, 'start_year_month'] >= pd.to_datetime(201801, format='%Y%m')) & \
            (df_workex.loc[i, 'start_year_month'] <= pd.to_datetime(201812, format='%Y%m')):
        df_workex.loc[i, 'NEW_JOB_2018'] = 1

columns = ['NEW_JOB_2015', 'NEW_JOB_2012', 'NEW_JOB_2016', 'NEW_JOB_2018', 'NEW_JOB_2017', 'NEW_JOB_2014', 'NEW_JOB_2009', 'NEW_JOB_2007', 'NEW_JOB_2011', 'NEW_JOB_2013', 'NEW_JOB_2010', 'NEW_JOB_2008']
df_workex['TOTAL_EXP'] = 0
for i in range(len(df_workex)):
        df_workex.loc[i, 'TOTAL_EXP'] = df_workex.loc[i, 'NEW_JOB_2015'] + \
                                        df_workex.loc[i, 'NEW_JOB_2012'] + \
                                        df_workex.loc[i, 'NEW_JOB_2016'] + \
                                        df_workex.loc[i, 'NEW_JOB_2018'] + \
                                        df_workex.loc[i, 'NEW_JOB_2017'] + \
                                        df_workex.loc[i, 'NEW_JOB_2014'] + \
                                        df_workex.loc[i, 'NEW_JOB_2009'] + \
                                        df_workex.loc[i, 'NEW_JOB_2007'] + \
                                        df_workex.loc[i, 'NEW_JOB_2011'] + \
                                        df_workex.loc[i, 'NEW_JOB_2013'] + \
                                        df_workex.loc[i, 'NEW_JOB_2010'] + \
                                        df_workex.loc[i, 'NEW_JOB_2008']
df_workex.drop(columns=columns, axis = 1, inplace=True)

workex_df = pd.DataFrame(columns=['user_id'])
workex_df['user_id'] = df_train['user_id']
merged_df_train = workex_df.merge(df_workex, on='user_id', how='left')
merged_df_train = merged_df_train.groupby('user_id').sum().reset_index()
# For Test Dataset
workex_df = pd.DataFrame(columns=['user_id'])
workex_df['user_id'] = df_test['user_id']
merged_df_test = workex_df.merge(df_workex, on='user_id', how='left')
merged_df_test = merged_df_test.groupby('user_id').sum().reset_index()

### Train & Test Concat for Work Experience
dfc_train = VLOOKUP(dfc_train, merged_df_train, 'user_id', ['REPUTABLE_COMP', 'TOTAL_EXP', 'NEW_JOB_2015', 'NEW_JOB_2012', 'NEW_JOB_2016', 'NEW_JOB_2018', 'NEW_JOB_2017', 'NEW_JOB_2014', 'NEW_JOB_2009', 'NEW_JOB_2007', 'NEW_JOB_2011', 'NEW_JOB_2013', 'NEW_JOB_2010', 'NEW_JOB_2008', 'company_id_subset_1074', 'company_id_subset_1343', 'company_id_subset_1456', 'company_id_subset_1562', 'company_id_subset_1607', 'company_id_subset_1647', 'company_id_subset_1805', 'company_id_subset_26', 'company_id_subset_305', 'company_id_subset_34', 'company_id_subset_35', 'company_id_subset_41', 'company_id_subset_518', 'company_id_subset_563', 'company_id_subset_726', 'company_id_subset_740', 'company_id_subset_850', 'company_id_subset_88', 'company_id_subset_89', 'company_id_subset_944', 'company_id_subset_Other', 'ISTANBUL', 'OTHER IN TR', 'İZMİR', 'ANKARA', 'ANTALYA', 'ABROAD'])
dfc_test = VLOOKUP(dfc_test, merged_df_test, 'user_id', ['NEW_JOB_2015', 'NEW_JOB_2012', 'NEW_JOB_2016', 'NEW_JOB_2018', 'NEW_JOB_2017', 'NEW_JOB_2014', 'NEW_JOB_2009', 'NEW_JOB_2007', 'NEW_JOB_2011', 'NEW_JOB_2013', 'NEW_JOB_2010', 'NEW_JOB_2008', 'company_id_subset_1074', 'company_id_subset_1343', 'company_id_subset_1456', 'company_id_subset_1562', 'company_id_subset_1607', 'company_id_subset_1647', 'company_id_subset_1805', 'company_id_subset_26', 'company_id_subset_305', 'company_id_subset_34', 'company_id_subset_35', 'company_id_subset_41', 'company_id_subset_518', 'company_id_subset_563', 'company_id_subset_726', 'company_id_subset_740', 'company_id_subset_850', 'company_id_subset_88', 'company_id_subset_89', 'company_id_subset_944', 'company_id_subset_Other', 'ISTANBUL', 'OTHER IN TR', 'İZMİR', 'ANKARA', 'ANTALYA', 'ABROAD'])











################################################# FREE SPACE ###############################################################################################################################################
df_compare = pd.read_csv('/Users/buraksayilar/Desktop/garanti-bbva-data-camp/SUB2_vs_SUB3.csv')




















### TEST AND TRAIN DATASETS ################################################################################################################################################################################################

############ Creating copy df for TRAIN SET
dfc_train_c = dfc_train
dfc_train_c.fillna(0, inplace=True)
#dfc_train_c.rename({'skill_C#':'skill_C1', 'skill_C++':'skill_C_PLUS'}, axis='columns', inplace=True)
dfc_train_c = OHE_partial(dfc_train_c, 'industry',encode_feature_list=[], valuecount_select=True, select_number=10)


## There are some inf. floats. Fix them
dfc_train_c.replace([np.inf, -np.inf], np.nan, inplace=True)
dfc_train_c.fillna(0, inplace=True)

dfc_train_c.drop(['user_id', 'location', 'industry', 'encoder_subset_İngilizce'], axis=1, inplace=True)

############# Creating copy df for TEST SET
dfc_test_c = dfc_test
dfc_test_c.fillna(0, inplace=True)
#dfc_test_c.rename({'skill_C#':'skill_C1', 'skill_C++':'skill_C_PLUS'}, axis='columns', inplace=True)
dfc_test_c = OHE_partial(dfc_test_c, 'industry',encode_feature_list=[], valuecount_select=True, select_number=10)

## There are some inf. floats. Fix them
dfc_test_c.replace([np.inf, -np.inf], np.nan, inplace=True)
dfc_test_c.fillna(0, inplace=True)

dfc_test_c.drop(['user_id', 'location', 'industry', 'encoder_subset_İngilizce'], axis=1, inplace=True)

## Creating Train and Test Sets
X = dfc_train_c.drop(['moved_after_2019'], axis=1)
X = X[[col for col in X.columns if col in dfc_test_c.columns]] # Industry columns are not the same.
y = dfc_train_c['moved_after_2019']



df_compare.drop(0, axis=0, inplace=True)
# Create a boolean mask by mapping the `user_id` column to a boolean value indicating whether it is in `df_compare['SUB_3']`
mask = dfc_test['user_id'].astype(int).map(lambda x: x in df_compare['SUB_3'].astype(int).values)

# Filter `df_test` based on the boolean mask
filtered_df_test = dfc_test[mask]
filtered_df_test['encoder_subset'].value_counts()









def hyp_op(X, y, model_name, cv=3, scoring="roc_auc"):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    print("Hyperparameter Optimization....")
    best_model = {}
    if model_name == "cart":
        print(f"########## Decision Tree (CART) ##########")
        classifier = DecisionTreeClassifier()
        params = {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        }
    elif model_name == "knn":
        print(f"########## K-Nearest Neighbors ##########")
        classifier = KNeighborsClassifier()
        params = {
            "n_neighbors": [3, 5, 7, 10],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }
    elif model_name == "xgboost":
        print(f"########## XGBoost ##########")
        classifier = XGBClassifier()
        params = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.3],
            "n_estimators": [50, 100, 200],
            "objective": ["binary:logistic"]
        }
    elif model_name == "logistic_regression":
        print(f"########## Logistic Regression ##########")
        classifier = LogisticRegression()
        params = {
            "penalty": ["l1", "l2"],
            "C": [0.1, 0.5, 1, 5, 10],
            "solver": ["liblinear", "saga"]
        }
    elif model_name == "catboost":
        print(f"########## CatBoost ##########")
        classifier = CatBoostClassifier()
        params = {
            "iterations": [100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [3, 5, 7],
            "l2_leaf_reg": [1, 3, 5, 7]
        }
    elif model_name == "lightgbm":
        print(f"########## LightGBM ##########")
        classifier = LGBMClassifier()
        params = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.3],
            "n_estimators": [50, 100, 200],
            "objective": ["binary"],
            "subsample": [1, 0.5, 0.7],
            "metric": ["auc"]
        }
    elif model_name == "random_forest":
        print(f"########## Random Forest ##########")
        classifier = RandomForestClassifier()
        params = {
            "n_estimators": [200, 300, 800],
            "max_depth": [5, 10, 20, None],
            "max_features" : [2, 7, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5, 10]
        }


    cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
    print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

    gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
    final_model = classifier.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
    print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
    print(f"{model_name} best params: {gs_best.best_params_}", end="\n\n")
    return final_model

def hyperparameter_optimization(X, y ,model_list, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name in model_list:
        final_model = hyp_op(X, y, name, cv=cv, scoring=scoring)
        best_models[name] = final_model
    return best_models

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('lightgbm', best_models["lightgbm"]),

                                              ('xgboost', best_models["xgboost"]),

                                              ('random_forest', best_models["random_forest"]),

                                              ],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf
####### VOTING CLASSIFIER ########################################################################################################################################################################################################
model_list = ['lightgbm', 'xgboost', 'random_forest']
best_models = hyperparameter_optimization(X, y, model_list)
voting_clf = voting_classifier(best_models, X, y)



####### RANDOM FORREST ##########################################################################################################################################################################################################

clf = RandomForestClassifier(max_features=7, min_samples_split=5, n_estimators=800)

cv  = StratifiedKFold(shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

hyp_op(X, y, 'random_forest')
clf.fit(X, y)

####### MLPC REGRESSOR ##########################################################################################################################################################################################################
MLPC = MLPClassifier()
mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.001],
             "hidden_layer_sizes": [(10,10), (5,5), (100,100,100)]}

### GridSearchCV
mlpc_cv_model = GridSearchCV(MLPC, mlpc_params, cv = 10, verbose = 1, n_jobs = -1).fit(X, y)
### RandomizeSearchCV
mlpc_cv_model = RandomizedSearchCV(MLPC, mlpc_params, cv=10, verbose= 1, n_jobs= -1).fit(X, y)
### Best Parameters
mlpc_cv_model.best_params_
### Models
mlpc_tuned = MLPClassifier(alpha = 0.01,hidden_layer_sizes =  (100, 100, 100)).fit(X, y)

####### XGBOOST ##########################################################################################################################################################################################################
XGBC = XGBClassifier()
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

xgbc_model = GridSearchCV(XGBC, xgboost_params, cv=10, verbose=1, n_jobs= -1).fit(X, y)
xgbc_tuned = XGBClassifier(learning_rate = 0.1, max_depth = 8, n_estimators = 200,use_label_encoder=False, eval_metric='logloss')

cv  = StratifiedKFold(shuffle=True, random_state=42)
scores = cross_val_score(xgbc_tuned, X, y, cv=cv, scoring='accuracy')

####### LOGISTIC REGRESSION ##########################################################################################################################################################################################################
LR = LogisticRegression()
LR.fit(X, y)


####### LIGHTGBM ##########################################################################################################################################################################################################
X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
lgbm_classifier = LGBMClassifier()
hyp_op(X, y, 'lightgbm')














################################### PREDICTIONS & SUBMISSION SETS ########################################################
X_TEST = dfc_test_c
####### VOTING CLASSIFIER ########################################################################################################################################################################################################

vol_1_results = clf.predict(X_TEST)
df_sub['moved_after_2019'] = vol_1_results
df_sub.to_csv('submisson_22.csv')

####### RANDOM FORREST ##########################################################################################################################################################################################################

rfc_results = clf.predict(X_TEST)
df_sub['moved_after_2019'] = rfc_results
df_sub.to_csv('submisson_22.csv')

####### MLP REGRESSOR ##########################################################################################################################################################################################################

mlpc_tuned_results = mlpc_tuned.predict(X_TEST)
df_sub['moved_after_2019'] = mlpc_tuned_results
df_sub.to_csv('submisson_21.csv')

####### XGBOOST ##########################################################################################################################################################################################################



####### LOGISTIC REGRESSION ##########################################################################################################################################################################################################

LR_results = LR.predict(X_TEST)
df_sub['moved_after_2019'] = LR_results
df_sub.to_csv('submisson_6.csv')

























####### FUTURE IMPORTANCE ##########################################################################################################################################################################################################
def plot_importance(model, features, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=0.5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:(len(features))])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
plot_importance(voting_clf, X)
feature_imp = pd.DataFrame({'Value': clf.feature_importances_, 'Feature': X_TEST.columns})
feature_imp.head(60)