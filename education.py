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

