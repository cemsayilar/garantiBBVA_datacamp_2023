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




#################################################################################################################################



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



####### LOGISTIC REGRESSION ###############################################################################################################################

LR_results = LR.predict(X_TEST)
df_sub['moved_after_2019'] = LR_results
df_sub.to_csv('submisson_6.csv')
