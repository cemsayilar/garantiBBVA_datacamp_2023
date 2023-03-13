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
