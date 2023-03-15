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
