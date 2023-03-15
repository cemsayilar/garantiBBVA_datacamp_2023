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

