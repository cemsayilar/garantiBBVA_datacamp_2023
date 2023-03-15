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
