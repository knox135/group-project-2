# import dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report


""" Format is: 'model = models.many_models(X_train, y_train, X_test, y_test)'
 if you get zero division errors and/or y_pred has values not in y_true,
 check and make sure your test values are expected.
 EX: 'arr([3,4,5,6,7,8,9])'
 while expected is from 0, therefore predictions will have 0,1,2 values that do not 
 exist in the check values.
"""

# run processed data through various models to determine which is best
def many_models_full(x,y,xt,yt):
    
    # Create instance of StandardScalar to fit and transform data
    scaled = StandardScaler()
    
    # Fit and Transform training data
    x_scale = scaled.fit_transform(x)
    
    # Transform test data
    xt_scale = scaled.transform(xt)
    
    # Create initial Random Forest Classifier model
    rf = RandomForestClassifier(random_state=13)
    
    # Fit data to Random forest model
    rf.fit(x,y)
    
    # Make predictions on Random Forest Model
    rf_predict = rf.predict(x)
    rf_test = rf.predict(xt)
    
    # Print scores and classification reports
    print(f'\nRandom Forest \nTest Accuracy: {rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,rf_test)}')
    print(f'classification report: \n {classification_report(yt,rf_test)}')
    
    # Create Gradient Boosting Classifier model
    gr = GradientBoostingClassifier(random_state=13)
    
    # Fit data to Gradient Boosting Classifier model
    gr.fit(x,y)
    
    # Make predictions on Gradient Boosting Classifier model
    gr_predict = gr.predict(x)
    gr_test = gr.predict(xt)
    
    # Print scores and classification reports
    print(f'\nGradient Boost \nTest Accuracy: {gr.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,gr_test)}')
    print(f'\n classification report: \n {classification_report(yt,gr_test)}')
    
    # Create Logistic Regression model
    lr = LogisticRegression(random_state=13)
    
    # Fit data to Logistic Regression model
    lr.fit(x_scale,y)
    
    # Make predictions on Logistic Regression model
    lr_predict = lr.predict(x_scale)
    lr_test = lr.predict(xt_scale)
    
    # Print scores and classification reports    
    print(f'\nLogistic Regression \nTest Accuracy: {lr.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,lr_test)}')
    print(f'classification report: \n {classification_report(yt,lr_test)}')
    
    # Create Poly Support Vector Classifier model
    svc = SVC(kernel='poly',random_state=13)
    
    # Fit data to Poly Support Vector Classifier model
    svc.fit(x_scale,y)
    
    # Make predictions on Poly Support Vector Classifier model
    svc_predict = svc.predict(x_scale)
    svc_test = svc.predict(xt_scale)
    
    # Print scores and classification reports
    print(f'\nPoly Support Vector \nTest Accuracy: {svc.score(xt_scale,yt)}\nBalanced test score: {balanced_accuracy_score(yt,svc_test)}')
    print(f'classification report: \n {classification_report(yt,svc_test)}')

    
    # Adaptive Boosting models(slows the function down quite a lot, comment out or delete if need to speed up for rest)
    # Create Adaptive Boosting models
    ada_low = AdaBoostClassifier(n_estimators=20,random_state=13)
    
    # Fit data to low estimators Adaptive Boosting model 
    ada_low.fit(x_scale,y)
    
    # Make predictions on low estimators Adaptive Boosting model
    ada_low_predict = ada_low.predict(x_scale)
    ada_low_test = ada_low.predict(xt_scale)
    
    # print scores and classification reports
    print(f'\nADA low estimators \nTest Accuracy: {ada_low.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,ada_low_test)}')
    print(f'classification report: \n {classification_report(yt,ada_low_test)}')
    
    # Create Adaptive Boosting model (1500 estimators)
    ada = AdaBoostClassifier(random_state=13)    
    
    # Fit data to Adaptive Boosting model (1500 estimators)
    ada.fit(x_scale,y)
    
    # Make predictions on Adaptive Boosting model (1500 estimators)
    ada_predict = ada.predict(x_scale)
    ada_test = ada.predict(xt_scale)
    
    # print scores and classification reports
    print(f'\nADA \nTest Accuracy: {ada.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,ada_test)}')
    print(f'classification report: \n {classification_report(yt,ada_test)}')
    
    # Create Sigmoid Support Vector Classifier model
    svc_sigmoid = SVC(kernel='sigmoid',random_state=13)
    
    # Fit data to Sigmoid Support Vector Classifier model
    svc_sigmoid.fit(x_scale,y)
    
    # Make predictions on Sigmoid Support Vector Classifier model
    svc_sigmoid_predict = svc_sigmoid.predict(x_scale)
    svc_sigmoid_test = svc_sigmoid.predict(xt_scale)
    
    # print scores and classification reports
    print(f'\nSVC Sigmoid \nTest Accuracy: {svc_sigmoid.score(xt_scale,yt)}\nbalanced test score: {balanced_accuracy_score(yt,svc_sigmoid_test)}')
    print(f'classification report: \n {classification_report(yt,svc_sigmoid_test)}')
    
    # Create comparison dataframe of the models with Model Name as Index
    comparison = pd.DataFrame(
        [
            ['Logistic Regression', lr.score(x_scale,y),lr.score(xt_scale,yt),balanced_accuracy_score(y,lr_predict),balanced_accuracy_score(yt,lr_test)],    
            ['SVC poly',svc.score(x_scale,y), svc.score(xt_scale,yt),balanced_accuracy_score(y,svc_predict),balanced_accuracy_score(yt,svc_test)],
            ['SVC sigmoid',svc_sigmoid.score(x_scale,y),svc_sigmoid.score(xt_scale,yt),balanced_accuracy_score(y,svc_sigmoid_predict),balanced_accuracy_score(yt,svc_sigmoid_test)],
            ['Random Forest',rf.score(x,y),rf.score(xt,yt),balanced_accuracy_score(y,rf_predict),balanced_accuracy_score(yt,rf_test)],
            ['Gradient Boosting',gr.score(x,y),gr.score(xt,yt),balanced_accuracy_score(y,gr_predict),balanced_accuracy_score(yt,gr_test)],
            ['ADA Low Estimators',ada_low.score(x_scale,y),ada_low.score(xt_scale,yt),balanced_accuracy_score(y,ada_low_predict),balanced_accuracy_score(yt,ada_low_test)],
            ['ADA boost', ada.score(x_scale,y),ada.score(xt_scale,yt),balanced_accuracy_score(y,ada_predict),balanced_accuracy_score(yt,ada_test)],
        ],
        columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    
    # Make new column calculating difference between training and test scores
    comparison['Balanced Difference'] = (comparison['Balanced Trained Score'] - comparison['Balanced Test Score'])
    
    # Sort by Balanced Test Score column and print results
    comparison = comparison.sort_values(by='Balanced Test Score', ascending=False)
    
    # Drop Balanced Trained Score for easier readability and print output
    # Examples
    # comparison = comparison[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    # comparison = comparison.drop(columns=['Balanced Trained Score'])
    comparison = comparison.drop('Balanced Trained Score',axis=1)
    print('non-tuned models dataframe')
    print(comparison.head(7))
    
    # Plot the comparison dataframe on a bar graph for visualizations
    comparison.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    
    
    # Create checks for Random Forest Best Parameters
    
    # Check best depth for random forest
    models = {'train_score': [], 'test_score': [], 'max_depth': []}
    
    # Loop through range to see where deviation happens
    for depth in range(1,20):
        
        # Save max depth to models
        models['max_depth'].append(depth)
        
        # Create model for max depth 
        model = RandomForestClassifier(max_depth=depth,random_state=13)
        
        # Fit the random forest model
        model.fit(x, y)
        
        # Make max depth predictions
        y_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Save max depth scores to models
        models['train_score'].append(balanced_accuracy_score(y, y_train_pred))
        models['test_score'].append(balanced_accuracy_score(yt, y_test_pred))
        
    # Save models data to dataframe and sort by test scores    
    models_df = pd.DataFrame(models)
    
    # Show graph of for loop
    models_df.plot(title='Random forest max depth', x='max_depth')
    
    # Set max_depth as index
    models_df_tolist = models_df.set_index('max_depth').sort_values(by='test_score',ascending=False)
    models_df = models_df.sort_values(by='test_score',ascending=False)
    print(f'\nRandom Forest Parameter Tuning:\n \nbest depth balanced test score: \n{models_df.head()}')
    
    # Save top 5 to a list
    best_rf = models_df_tolist.head().index.tolist()
    print(f'max depths: \n{best_rf}\n')
    
    
    
    # Test for min_samples_leafs
    leaf = range(2,10)
    leaves = {'train_score':[],'test_score':[],'min_leaf': []}
    
    # Loop through min_leaf values
    for i in leaf:
        
        # Create Random Forest Classifier instance with depth
        model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],min_samples_leaf=i,random_state=13)
        
        # Fit model
        model.fit(x,y)
        
        # Predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Store scores and best amounts in leaves dictionary
        leaves['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        leaves['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        leaves['min_leaf'].append(i)
        
    # Store dictionary in a dataframe
    leaves_df = pd.DataFrame(leaves)
    
    # Plot best min leaf on graph
    leaves_df.plot(title='Random Forest Best Leaf',x='min_leaf')
    
    # Set leaves_df index to 'min_leaf'
    leaves_df_tolist = leaves_df.set_index('min_leaf').sort_values(by='test_score',ascending=False)
    leaves_df = leaves_df.sort_values(by='test_score',ascending=False)
    print(f'best min leaf balanced test score: \n{leaves_df.head()}')
    
    # Save to list
    best_leaves = leaves_df_tolist.head().index.tolist()
    print(f'best min leaf \n{best_leaves}')
    
    
    
    # Test for best n_estimators
    # Create n_estimators range search
    est = range(2,100,2)
    
    # Create dictionary to store results
    est_models = {'train_score':[],'test_score':[],'n_estimators':[]}
    
    # Loop through range to find best n_estimators
    for e in est:
        
        # Create random forest model
        est_model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],min_samples_leaf=leaves_df['min_leaf'].iloc[0],n_estimators=e,random_state=13)
        
        # Fit model 
        est_model.fit(x,y)
        
        # Make predictions for n_estimators
        X_test_pred = est_model.predict(xt)
        y_train_pred = est_model.predict(x)
        
        # Store results in dictionairy
        est_models['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        est_models['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        est_models['n_estimators'].append(e)
        
    # Save to dataframe
    estimators_df = pd.DataFrame(est_models)
    
    # Graph best n_estimators
    estimators_df.plot(title='best n_estimators',x='n_estimators')
    
    # Set index to n_estimators
    estimators_df_tolist = estimators_df.set_index('n_estimators').sort_values(by='test_score',ascending=False)
    estimators_df = estimators_df.sort_values(by='test_score',ascending=False)
    print(f'\nbest n_estimators balanced test score: \n{estimators_df.head()}')
    
    # Save to list
    best_est = estimators_df_tolist.head().index.tolist()
    print(f'best n_estimators \n{best_est}\n')
    
    
    
    # Test for min_split
    split = range(2,60,2)
    splits = {'train_score':[],'test_score':[],'min_split': []}
    
    # Loop through leaf values
    for s in split:
        
        # Create Random Forest Classifier instance with depth
        model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],n_estimators=estimators_df['n_estimators'].iloc[0],min_samples_leaf=leaves_df['min_leaf'].iloc[0],min_samples_split=s,random_state=13)
        
        # Fit model
        model.fit(x,y)
        
        # Predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Store scores and best amounts in leaves dictionary
        splits['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        splits['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        splits['min_split'].append(s)
        
    # Store dictionary in a dataframe
    splits_df = pd.DataFrame(splits)
    
    # Plot best min_split on graph
    splits_df.plot(title='Random Forest Best min_split',x='min_split')
    
    # Set split_df index to 'min_split'
    splits_df_tolist = splits_df.set_index('min_split').sort_values(by='test_score',ascending=False)
    splits_df = splits_df.sort_values(by='test_score',ascending=False)
    print(f'best min_split balanced test score: \n{splits_df.head()}')
    
    # Save to list
    best_split = splits_df_tolist.head().index.tolist()
    print(f'best min_split \n{best_split}\n')
    
    
    
    # Test for max_leaf_nodes
    max_leaf = range(2,200,3)
    max_leaves = {'train_score':[],'test_score':[],'max_leaf': []}
    
    # Loop through leaf values
    for m in max_leaf:
        
        # Create Random Forest Classifier instance with depth
        model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],n_estimators=estimators_df['n_estimators'].iloc[0],min_samples_leaf=leaves_df['min_leaf'].iloc[0],min_samples_split=splits_df['min_split'].iloc[0],max_leaf_nodes=m,random_state=13)
        
        # Fit model
        model.fit(x,y)
        
        # Predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Store scores and best amounts in leaves dictionary
        max_leaves['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        max_leaves['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        max_leaves['max_leaf'].append(m)
        
    # Store dictionary in a dataframe
    max_leaf_df = pd.DataFrame(max_leaves)
    
    # Plot best max_leaf on graph
    max_leaf_df.plot(title='Random Forest Best max leaf',x='max_leaf')
    
    # Set max_leaf_df index to 'max_leaf'
    max_leaf_df_tolist = max_leaf_df.set_index('max_leaf').sort_values(by='test_score',ascending=False)
    max_leaf_df = max_leaf_df.sort_values(by='test_score',ascending=False)
    print(f'best max_leaf balanced test score: \n{max_leaf_df.head()}')
    
    # Save to list
    best_max_leaf = max_leaf_df_tolist.head().index.tolist()
    print(f'best max leaf \n{best_max_leaf}\n')
    
    
    
    # Create test Random Forest Classifier
    
    test_tuned_rf = RandomForestClassifier(n_estimators=estimators_df['n_estimators'].iloc[0],max_depth=models_df['max_depth'].iloc[0],min_samples_leaf=leaves_df['min_leaf'].iloc[0],min_samples_split=splits_df['min_split'].iloc[0],max_leaf_nodes=max_leaf_df['max_leaf'].iloc[0],random_state=13)
    
    # train the tuned model
    test_tuned_rf.fit(x,y)
    
    # predict with the tuned model
    test_tuned_train = test_tuned_rf.predict(x)
    tuned_test = test_tuned_rf.predict(xt)
    
    # view tuned Random Forest Model scores and classification report
    print(f'\nTest best Random Forest Tuned Parameters scores \n')
    print(f'\nTest Best Random Forest Tuned Parameters Scores \nTest Accuracy: {test_tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,tuned_test)}')
    print(f'classification report: \n {classification_report(yt,tuned_test)}')
    
    # add tuned Random Forest Model to the comparison DataFrame and Plot
    test_df = pd.DataFrame(
        [
            ['Test Random Forest',test_tuned_rf.score(x,y),test_tuned_rf.score(xt,yt),balanced_accuracy_score(y,test_tuned_train),balanced_accuracy_score(yt,tuned_test)]
        ],
            columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    test_df['Balanced Difference'] = (test_df['Balanced Trained Score'] - test_df['Balanced Test Score'])
    test_df = test_df[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    comparison_df = pd.concat([comparison,test_df]).sort_values(by='Balanced Test Score',ascending=False)
    print(f'updated comparison DataFrame \n{comparison_df.head(8)}')
    
    comparison_df.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    # Save best parameters to dictionary to use to find best combination
    best_params = {
        'n_estimators': best_est,
        'max_depth': best_rf,
        'min_samples_leaf': best_leaves,
        'min_split': best_split,
        'max_leaf': best_max_leaf
    }

    # Create dictionary to store results
    best_results = {'n_estimators':[],'max_depth':[],'min_split':[],'max_leaf':[],'min_samples_leaf':[],'train_score':[],'test_score':[]}
    
    # Loop through combinations(refactor in future for efficiency)
    for e in best_params['n_estimators']:
        for m in best_params['max_depth']:
            for l in best_params['min_samples_leaf']:
                for s in best_params['min_split']:
                    for a in best_params['max_leaf']:
                
                        # Create random forest model with best parameters
                        rf_model = RandomForestClassifier(n_estimators=e,max_depth=m,min_samples_split=s,max_leaf_nodes=a,min_samples_leaf=l,random_state=13)
                        
                        # Fit data
                        rf_model.fit(x,y)
                        
                        # Predictions
                        test_pred = rf_model.predict(xt)
                        train_pred = rf_model.predict(x)
                        
                        # Append balanced scores
                        best_results['train_score'].append(balanced_accuracy_score(y,train_pred))
                        best_results['test_score'].append(balanced_accuracy_score(yt,test_pred))
                        best_results['n_estimators'].append(e)
                        best_results['max_depth'].append(m)
                        best_results['min_samples_leaf'].append(l)
                        best_results['min_split'].append(s)
                        best_results['max_leaf'].append(a)
                
    # Save best results to dataframe
    best_results_df = pd.DataFrame(best_results).sort_values(by='test_score',ascending=False)
    
    # Plot best_results
    best_results_plot = best_results_df[['test_score','train_score']].sort_values(by='test_score',ascending=False).reset_index(drop=True)
    best_results_plot.plot(title='Tuned Random Forest Balanced Trained/Balance Test',kind='line')
    
    # Set test_score to index
    best_results_df = best_results_df.set_index('test_score')
    print(f'\nbest results:\n{best_results_df}')
    
    # Save best parameters to print
    best_n = best_results_df['n_estimators'].iloc[0]
    best_d = best_results_df['max_depth'].iloc[0]
    best_l = best_results_df['min_samples_leaf'].iloc[0]
    best_s = best_results_df['min_split'].iloc[0]
    best_a = best_results_df['max_leaf'].iloc[0]
    
    
    # Show best parameters
    print(f'\nTop Best Parameters: \nbest n_estimators: {best_n} \nbest max_depth: {best_d} \nBest min_split: {best_s} \nbest max_leaf: {best_a} \nbest min_leaves: {best_l} \n')
    
    # Create random forest instance with best parameters
    tuned_rf = RandomForestClassifier(n_estimators=best_results_df['n_estimators'].iloc[0], min_samples_leaf=best_results_df['min_samples_leaf'].iloc[0], max_depth=best_results_df['max_depth'].iloc[0],max_leaf_nodes=best_results_df['max_leaf'].iloc[0],min_samples_split=best_results_df['min_split'].iloc[0],random_state=13)    
    
    # Train the tuned model
    tuned_rf.fit(x,y)
    
    # Predict with the tuned model
    best_train = tuned_rf.predict(x)
    best_test = tuned_rf.predict(xt)
    
    # View tuned Random Forest Model scores and classification report
    print(f'\nBest Random Forest Tuned Parameters Scores \nTest Accuracy: {tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,best_test)}')
    print(f'classification report: \n {classification_report(yt,best_test)}')
    
    # Add tuned Random Forest Model to the comparison DataFrame and Plot
    best_df = pd.DataFrame(
        [
            ['Tuned Random Forest',tuned_rf.score(x,y),tuned_rf.score(xt,yt),balanced_accuracy_score(y,best_train),balanced_accuracy_score(yt,best_test)]
        ],
            columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    )
    best_df = best_df.set_index('Model Name')
    best_df['Balanced Difference'] = (best_df['Balanced Trained Score'] - best_df['Balanced Test Score'])
    
    # Show best scores
    print('Balanced Trained Score')
    print(best_df['Balanced Trained Score'])
    print('Balanced Test Score')
    print(best_df['Balanced Test Score'])
    
    # Set columns wanted
    best_df = best_df[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    
    # Combine best results with initial dataframe
    comparison_df = pd.concat([comparison_df,best_df]).sort_values(by='Balanced Test Score',ascending=False)
    
    # View updated dataframe
    print(f'updated comparison DataFrame \n{comparison_df.head(8)}')
    
    # Plot updated comparison model against initial models
    comparison_df.plot(kind='bar')
    return comparison,models_df,estimators_df,best_results_df,comparison_df,best_train,best_test ,leaves_df

""" many_models without min_leaf search to see if 
    efficiency is worth score difference"""


# run processed data through various models to determine which is best
def many_models_max_leaf(x,y,xt,yt):
    
    # Create initial Random Forest Classifier model
    rf = RandomForestClassifier(random_state=13)
    
    # Fit data to Random forest model
    rf.fit(x,y)
    
    # Make predictions on Random Forest Model
    rf_predict = rf.predict(x)
    rf_test = rf.predict(xt)
    
    # Print scores and classification reports
    print(f'\nRandom Forest \nTest Accuracy: {rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,rf_test)}')
    print(f'classification report: \n {classification_report(yt,rf_test)}')
    
    
    # Create comparison dataframe of the models with Model Name as Index
    comparison = pd.DataFrame(
        [
            ['Random Forest',rf.score(x,y),rf.score(xt,yt),balanced_accuracy_score(y,rf_predict),balanced_accuracy_score(yt,rf_test)],
        ],
        columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    
    # Make new column calculating difference between training and test scores
    comparison['Balanced Difference'] = (comparison['Balanced Trained Score'] - comparison['Balanced Test Score'])
    
    # Sort by Balanced Test Score column and print results
    comparison = comparison.sort_values(by='Balanced Test Score', ascending=False)
    
    # Drop Balanced Trained Score for easier readability and print output
    # Examples
    # comparison = comparison[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    # comparison = comparison.drop(columns=['Balanced Trained Score'])
    comparison = comparison.drop('Balanced Trained Score',axis=1)
    print('non-tuned models dataframe')
    print(comparison.head(7))
    
    # Plot the comparison dataframe on a bar graph for visualizations
    comparison.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    
    
    # Create checks for Random Forest Best Parameters
    
    # Check best depth for random forest
    models = {'train_score': [], 'test_score': [], 'max_depth': []}
    
    # Loop through range to see where deviation happens
    for depth in range(1,20):
        
        # Save max depth to models
        models['max_depth'].append(depth)
        
        # Create model for max depth 
        model = RandomForestClassifier(max_depth=depth,random_state=13)
        
        # Fit the random forest model
        model.fit(x, y)
        
        # Make max depth predictions
        y_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Save max depth scores to models
        models['train_score'].append(balanced_accuracy_score(y, y_train_pred))
        models['test_score'].append(balanced_accuracy_score(yt, y_test_pred))
        
    # Save models data to dataframe and sort by test scores    
    models_df = pd.DataFrame(models)
    
    # Show graph of for loop
    models_df.plot(title='Random forest max depth', x='max_depth')
    
    # Set max_depth as index
    models_df_tolist = models_df.set_index('max_depth').sort_values(by='test_score',ascending=False)
    models_df = models_df.sort_values(by='test_score',ascending=False)
    print(f'\nRandom Forest Parameter Tuning:\n \nbest depth balanced test score: \n{models_df.head()}')
    
    # Save top 5 to a list
    best_rf = models_df_tolist.head().index.tolist()
    print(f'max depths: \n{best_rf}\n')
    
    
    # Test for best n_estimators
    # Create n_estimators range search
    est = range(2,100,2)
    
    # Create dictionary to store results
    est_models = {'train_score':[],'test_score':[],'n_estimators':[]}
    
    # Loop through range to find best n_estimators
    for e in est:
        
        # Create random forest model
        est_model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],n_estimators=e,random_state=13)
        
        # Fit model 
        est_model.fit(x,y)
        
        # Make predictions for n_estimators
        X_test_pred = est_model.predict(xt)
        y_train_pred = est_model.predict(x)
        
        # Store results in dictionairy
        est_models['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        est_models['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        est_models['n_estimators'].append(e)
        
    # Save to dataframe
    estimators_df = pd.DataFrame(est_models)
    
    # Graph best n_estimators
    estimators_df.plot(title='best n_estimators',x='n_estimators')
    
    # Set index to n_estimators
    estimators_df_tolist = estimators_df.set_index('n_estimators').sort_values(by='test_score',ascending=False)
    estimators_df = estimators_df.sort_values(by='test_score',ascending=False)
    print(f'\nbest n_estimators balanced test score: \n{estimators_df.head()}')
    
    # Save to list
    best_est = estimators_df_tolist.head().index.tolist()
    print(f'best n_estimators \n{best_est}\n')
    
    
    
    # Test for min_split
    split = range(2,60,2)
    splits = {'train_score':[],'test_score':[],'min_split': []}
    
    # Loop through leaf values
    for s in split:
        
        # Create Random Forest Classifier instance with depth
        model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],n_estimators=estimators_df['n_estimators'].iloc[0],min_samples_split=s,random_state=13)
        
        # Fit model
        model.fit(x,y)
        
        # Predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Store scores and best amounts in leaves dictionary
        splits['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        splits['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        splits['min_split'].append(s)
        
    # Store dictionary in a dataframe
    splits_df = pd.DataFrame(splits)
    
    # Plot best min_split on graph
    splits_df.plot(title='Random Forest Best min_split',x='min_split')
    
    # Set split_df index to 'min_split'
    splits_df_tolist = splits_df.set_index('min_split').sort_values(by='test_score',ascending=False)
    splits_df = splits_df.sort_values(by='test_score',ascending=False)
    print(f'best min_split balanced test score: \n{splits_df.head()}')
    
    # Save to list
    best_split = splits_df_tolist.head().index.tolist()
    print(f'best min_split \n{best_split}\n')
    
    
    
    # Test for max_leaf_nodes
    max_leaf = range(2,200,3)
    max_leaves = {'train_score':[],'test_score':[],'max_leaf': []}
    
    # Loop through leaf values
    for m in max_leaf:
        
        # Create Random Forest Classifier instance with depth
        model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],n_estimators=estimators_df['n_estimators'].iloc[0],min_samples_split=splits_df['min_split'].iloc[0],max_leaf_nodes=m,random_state=13)
        
        # Fit model
        model.fit(x,y)
        
        # Predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Store scores and best amounts in leaves dictionary
        max_leaves['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        max_leaves['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        max_leaves['max_leaf'].append(m)
        
    # Store dictionary in a dataframe
    max_leaf_df = pd.DataFrame(max_leaves)
    
    # Plot best min_split on graph
    max_leaf_df.plot(title='Random Forest Best max leaf',x='max_leaf')
    
    # Set split_df index to 'min_split'
    max_leaf_df_tolist = max_leaf_df.set_index('max_leaf').sort_values(by='test_score',ascending=False)
    max_leaf_df = max_leaf_df.sort_values(by='test_score',ascending=False)
    print(f'best max_leaf balanced test score: \n{max_leaf_df.head()}')
    
    # Save to list
    best_max_leaf = max_leaf_df_tolist.head().index.tolist()
    print(f'best max leaf \n{best_max_leaf}\n')
    
    
    
    # Create test Random Forest Classifier
    test_tuned_rf = RandomForestClassifier(n_estimators=estimators_df['n_estimators'].iloc[0],max_depth=models_df['max_depth'].iloc[0],min_samples_split=splits_df['min_split'].iloc[0],max_leaf_nodes=max_leaf_df['max_leaf'].iloc[0],random_state=13)
    
    # train the tuned model
    test_tuned_rf.fit(x,y)
    
    # predict with the tuned model
    test_tuned_train = test_tuned_rf.predict(x)
    tuned_test = test_tuned_rf.predict(xt)
    
    # view tuned Random Forest Model scores and classification report
    print(f'\nTest Best Random Forest Tuned Parameters Scores "max leaf" \nTest Accuracy: {test_tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,tuned_test)}')
    print(f'classification report: \n {classification_report(yt,tuned_test)}')
    
    # add tuned Random Forest Model to the comparison DataFrame and Plot
    test_df = pd.DataFrame(
        [
            ['Test Random Forest',test_tuned_rf.score(x,y),test_tuned_rf.score(xt,yt),balanced_accuracy_score(y,test_tuned_train),balanced_accuracy_score(yt,tuned_test)]
        ],
            columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    test_df['Balanced Difference'] = (test_df['Balanced Trained Score'] - test_df['Balanced Test Score'])
    test_df = test_df[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    comparison_df = pd.concat([comparison,test_df]).sort_values(by='Balanced Test Score',ascending=False)
    print(f'updated comparison DataFrame \n{comparison_df.head(8)}')
    
    comparison_df.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    # Save best parameters to dictionary to use to find best combination
    best_params = {
        'n_estimators': best_est,
        'max_depth': best_rf,
        'min_split': best_split,
        'max_leaf': best_max_leaf
    }

    # Create dictionary to store results
    best_results = {'n_estimators':[],'max_depth':[],'min_split':[],'max_leaf':[],'train_score':[],'test_score':[]}
    
    # Loop through combinations(refactor in future for efficiency)
    for e in best_params['n_estimators']:
        for m in best_params['max_depth']:
            #for l in best_params['min_samples_leaf']:
            for s in best_params['min_split']:
                for a in best_params['max_leaf']:
                # Create random forest model with best parameters
                    
                    #,min_samples_leaf=l
                    rf_model = RandomForestClassifier(n_estimators=e,max_depth=m,min_samples_split=s,max_leaf_nodes=a,random_state=13)
                    # Fit data
                    rf_model.fit(x,y)
                    # Predictions
                    test_pred = rf_model.predict(xt)
                    train_pred = rf_model.predict(x)
                    # Append balanced scores
                    best_results['train_score'].append(balanced_accuracy_score(y,train_pred))
                    best_results['test_score'].append(balanced_accuracy_score(yt,test_pred))
                    best_results['n_estimators'].append(e)
                    best_results['max_depth'].append(m)
                    best_results['min_split'].append(s)
                    best_results['max_leaf'].append(a)
                    
    # Save best results to dataframe
    best_results_df = pd.DataFrame(best_results).sort_values(by='test_score',ascending=False)
    
    # Plot best_results
    best_results_plot = best_results_df[['test_score','train_score']].sort_values(by='test_score',ascending=False).reset_index(drop=True)
    best_results_plot.plot(title='Tuned Random Forest Balanced Trained/Balance Test',kind='line')
    
    # Set test_score as index
    best_results_df = best_results_df.set_index('test_score')
    print(f'\nbest results:\n{best_results_df}')
    
    # Save best parameters to print
    best_n = best_results_df['n_estimators'].iloc[0]
    best_d = best_results_df['max_depth'].iloc[0]
    best_s = best_results_df['min_split'].iloc[0]
    best_a = best_results_df['max_leaf'].iloc[0]
    
    
    print(f'\nTop Best Parameters "max leaf": \nbest n_estimators: {best_n} \nbest max_depth: {best_d} \nBest min_split: {best_s} \nbest max_leaf: {best_a}')
    
    # Create random forest instance with best parameters
    tuned_rf = RandomForestClassifier(n_estimators=best_results_df['n_estimators'].iloc[0], max_depth=best_results_df['max_depth'].iloc[0],max_leaf_nodes=best_results_df['max_leaf'].iloc[0],min_samples_split=best_results_df['min_split'].iloc[0],random_state=13)    
    
    # Train the tuned model
    tuned_rf.fit(x,y)
    
    # Predict with the tuned model
    best_train = tuned_rf.predict(x)
    best_test = tuned_rf.predict(xt)
    
    # View tuned Random Forest Model scores and classification report
    print(f'\nBest Random Forest Tuned Parameters Scores "max leaf" \nTest Accuracy: {tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,best_test)}')
    print(f'classification report: \n {classification_report(yt,best_test)}')
    
    # Add tuned Random Forest Model to the comparison DataFrame and Plot
    best_df = pd.DataFrame(
        [
            ['Tuned Random Forest',tuned_rf.score(x,y),tuned_rf.score(xt,yt),balanced_accuracy_score(y,best_train),balanced_accuracy_score(yt,best_test)]
        ],
            columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    best_df['Balanced Difference'] = (best_df['Balanced Trained Score'] - best_df['Balanced Test Score'])
    print('Balanced Trained Score')
    print(best_df['Balanced Trained Score'])
    print('Balanced Test Score')
    print(best_df['Balanced Test Score'])
    best_df = best_df[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    comparison_df = pd.concat([comparison_df,best_df]).sort_values(by='Balanced Test Score',ascending=False)
    print(f'updated comparison DataFrame \n{comparison_df.head(8)}')
    
    # Plot updated comparison model against initial models
    comparison_df.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    return comparison,models_df,estimators_df,best_results_df,comparison_df,best_train,best_test




""" many_models without max_leaf search to see if 
    efficiency is worth score difference"""


# run processed data through various models to determine which is best
def many_models_min_leaf(x,y,xt,yt):
    
    
    # Create initial Random Forest Classifier model
    rf = RandomForestClassifier(random_state=13)
    
    # Fit data to Random forest model
    rf.fit(x,y)
    
    # Make predictions on Random Forest Model
    rf_predict = rf.predict(x)
    rf_test = rf.predict(xt)
    
    # Print scores and classification reports
    print(f'\nRandom Forest \nTest Accuracy: {rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,rf_test)}')
    print(f'classification report: \n {classification_report(yt,rf_test)}')
    
    
    # Create comparison dataframe of the models with Model Name as Index
    comparison = pd.DataFrame(
        [
            ['Random Forest',rf.score(x,y),rf.score(xt,yt),balanced_accuracy_score(y,rf_predict),balanced_accuracy_score(yt,rf_test)],
        ],
        columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    
    # Make new column calculating difference between training and test scores
    comparison['Balanced Difference'] = (comparison['Balanced Trained Score'] - comparison['Balanced Test Score'])
    
    # Sort by Balanced Test Score column and print results
    comparison = comparison.sort_values(by='Balanced Test Score', ascending=False)
    
    # Drop Balanced Trained Score for easier readability and print output
    comparison = comparison.drop('Balanced Trained Score',axis=1)
    print('non-tuned models dataframe')
    print(comparison.head(7))
    
    # Plot the comparison dataframe on a bar graph for visualizations
    comparison.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    
    # Create checks for Random Forest Best Parameters 
    # Check best depth for random forest
    models = {'train_score': [], 'test_score': [], 'max_depth': []}
    
    # Loop through range to see where deviation happens
    for depth in range(1,20):
        
        # Save max depth to models
        models['max_depth'].append(depth)
        
        # Create model for max depth 
        model = RandomForestClassifier(max_depth=depth,random_state=13)
        
        # Fit the random forest model
        model.fit(x, y)
        
        # Make max depth predictions
        y_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Save max depth scores to models
        models['train_score'].append(balanced_accuracy_score(y, y_train_pred))
        models['test_score'].append(balanced_accuracy_score(yt, y_test_pred))
        
    # Save models data to dataframe and sort by test scores    
    models_df = pd.DataFrame(models)
    
    # Show graph of for loop
    models_df.plot(title='Random forest max depth', x='max_depth')
    
    # Set max_depth as index
    models_df_tolist = models_df.set_index('max_depth').sort_values(by='test_score',ascending=False)
    models_df = models_df.sort_values(by='test_score',ascending=False)
    
    # View best max_depth
    print(f'\nRandom Forest Parameter Tuning:\n \nbest depth balanced test score: \n{models_df.head()}')
    
    # Save top 5 to a list
    best_rf = models_df_tolist.head().index.tolist()
    print(f'max depths: \n{best_rf}\n')
    
    
    
    
    # Test for min_samples_leafs
    leaf = range(2,30)
    leaves = {'train_score':[],'test_score':[],'min_leaf': []}
    
    # Loop through leaf values
    for i in leaf:
        
        # Create Random Forest Classifier instance with depth
        model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],min_samples_leaf=i,random_state=13)
        
        # Fit model
        model.fit(x,y)
        
        # Predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Store scores and best amounts in leaves dictionary
        leaves['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        leaves['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        leaves['min_leaf'].append(i)
        
    # Store dictionary in a dataframe
    leaves_df = pd.DataFrame(leaves)
    
     # Plot best leaf on graph
    leaves_df.plot(title='Random Forest Best Leaf',x='min_leaf')
    
    # Set leaves_df index to 'min_leaf'
    leaves_df_tolist = leaves_df.set_index('min_leaf').sort_values(by='test_score',ascending=False)
    leaves_df = leaves_df.sort_values(by='test_score',ascending=False)
    
    # View best min_leaf
    print(f'best leaf balanced test score: \n{leaves_df.head()}')
    
    # Save to list
    best_leaves = leaves_df_tolist.head().index.tolist()
    print(f'best leaf \n{best_leaves}')
    

    
    # Test for best n_estimators
    # Create n_estimators range search
    est = range(2,100,2)
    
    # Create dictionary to store results
    est_models = {'train_score':[],'test_score':[],'n_estimators':[]}
    
    # Loop through range to find best n_estimators
    for e in est:
        
        # Create random forest model
        est_model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],min_samples_leaf=leaves_df['min_leaf'].iloc[0],n_estimators=e,random_state=13)
        
        # Fit model 
        est_model.fit(x,y)
        
        # Make predictions for n_estimators
        X_test_pred = est_model.predict(xt)
        y_train_pred = est_model.predict(x)
        
        # Store results in dictionary
        est_models['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        est_models['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        est_models['n_estimators'].append(e)
        
    # Save to dataframe
    estimators_df = pd.DataFrame(est_models)
    
    # Graph best n_estimators
    estimators_df.plot(title='best n_estimators',x='n_estimators')
    
    # Set index to n_estimators
    estimators_df_tolist = estimators_df.set_index('n_estimators').sort_values(by='test_score',ascending=False)
    estimators_df = estimators_df.sort_values(by='test_score',ascending=False)
    
    # View best n_estimators
    print(f'\nbest n_estimators balanced test score: \n{estimators_df.head()}')
    
    # Save to list
    best_est = estimators_df_tolist.head().index.tolist()
    print(f'best n_estimators \n{best_est}\n')
    
    
    
    # Test for min_split
    split = range(2,60,2)
    
    # Create dictionary to store results
    splits = {'train_score':[],'test_score':[],'min_split': []}
    
    # Loop through leaf values
    for s in split:
        
        # Create Random Forest Classifier instance with depth
        model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],min_samples_leaf=leaves_df['min_leaf'].iloc[0],n_estimators=estimators_df['n_estimators'].iloc[0],min_samples_split=s,random_state=13)
        
        # Fit model
        model.fit(x,y)
        
        # Predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Store scores and best amounts in leaves dictionary
        splits['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        splits['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        splits['min_split'].append(s)
        
    # Store dictionary in a dataframe
    splits_df = pd.DataFrame(splits)
    
     # Plot best min_split on graph
    splits_df.plot(title='Random Forest Best min_split',x='min_split')
    
    # Set split_df index to 'min_split'
    splits_df_tolist = splits_df.set_index('min_split').sort_values(by='test_score',ascending=False)
    splits_df = splits_df.sort_values(by='test_score',ascending=False)
    
    # View best min_split
    print(f'best min_split balanced test score: \n{splits_df.head()}')
    
    # Save to list
    best_split = splits_df_tolist.head().index.tolist()
    
    # View best min_split list
    print(f'best min_split \n{best_split}\n')
    
    
    
    # Save best parameters to dictionary to use to find best combination
    best_params = {
        'n_estimators': best_est,
        'max_depth': best_rf,
        'min_samples_leaf': best_leaves,
        'min_split': best_split,
    }

    # Create dictionary to store results
    best_results = {'n_estimators':[],'max_depth':[],'min_samples_leaf':[],'min_split':[],'train_score':[],'test_score':[]}
    
    # Loop through combinations(refactor in future for efficiency)
    for e in best_params['n_estimators']:
        for m in best_params['max_depth']:
            for l in best_params['min_samples_leaf']:
                for s in best_params['min_split']:
                    
                    
                    # Create random forest model with best parameters
                    rf_model = RandomForestClassifier(n_estimators=e,max_depth=m,min_samples_leaf=l,min_samples_split=s,random_state=13)
                    
                    # Fit data
                    rf_model.fit(x,y)#
                    
                    # Predictions
                    test_pred = rf_model.predict(xt)
                    train_pred = rf_model.predict(x)#
                    
                    # Append balanced scores
                    best_results['train_score'].append(balanced_accuracy_score(y,train_pred))
                    best_results['test_score'].append(balanced_accuracy_score(yt,test_pred))
                    best_results['n_estimators'].append(e)
                    best_results['max_depth'].append(m)
                    best_results['min_samples_leaf'].append(l)
                    best_results['min_split'].append(s)
                    
    # Save best results to dataframe
    best_results_df = pd.DataFrame(best_results).sort_values(by='test_score',ascending=False)
    
    # Plot best_results
    best_results_plot = best_results_df[['test_score','train_score']].sort_values(by='test_score',ascending=False).reset_index(drop=True)
    best_results_plot.plot(title='Tuned Random Forest Balanced Trained/Balance Test',kind='line')
    
    # Set test_score as index
    best_results_df = best_results_df.set_index('test_score')
    
    # View best results
    print(f'\nbest results:\n{best_results_df}')
    
    # Save best parameters to print
    best_n = best_results_df['n_estimators'].iloc[0]
    best_d = best_results_df['max_depth'].iloc[0]
    best_l = best_results_df['min_samples_leaf'].iloc[0]
    best_s = best_results_df['min_split'].iloc[0]
    print(f'best n_estimators: {best_n} \nbest max_depth: {best_d} \nbest leaves: {best_l} \nBest min_split: {best_s} \n')
    
    # Create random forest instance with best parameters
    tuned_rf = RandomForestClassifier(n_estimators=best_results_df['n_estimators'].iloc[0], max_depth=best_results_df['max_depth'].iloc[0], min_samples_leaf=best_results_df['min_samples_leaf'].iloc[0],min_samples_split=best_results_df['min_split'].iloc[0],random_state=13)    
    
    # Train the tuned model
    tuned_rf.fit(x,y)
    
    # Predict with the tuned model
    best_train = tuned_rf.predict(x)
    best_test = tuned_rf.predict(xt)
    
    # View tuned Random Forest Model scores and classification report
    #print(f'best Random Forest Tuned Parameters scores "min leaf" \n')
    print(f'\nBest Random Forest Tuned Parameters Scores "min leaf" \nTest Accuracy: {tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,best_test)}')
    print(f'classification report: \n {classification_report(yt,best_test)}')
    
    # Add tuned Random Forest Model to the comparison DataFrame and Plot
    best_df = pd.DataFrame(
        [
            ['Tuned Random Forest',tuned_rf.score(x,y),tuned_rf.score(xt,yt),balanced_accuracy_score(y,best_train),balanced_accuracy_score(yt,best_test)]
        ],
            columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    best_df['Balanced Difference'] = (best_df['Balanced Trained Score'] - best_df['Balanced Test Score'])
    
    # Print tuned 'Balance Trained Score' and 'Balance Test Score' to show Difference column 
    print('Balanced Trained Score')
    print(best_df['Balanced Trained Score'])
    print('Balanced Test Score')
    print(best_df['Balanced Test Score'])
    
    # Drop Balanced Trained Score for easier readability
    best_df = best_df[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    
    # Add tuned parameters to initial comparison dataframe
    comparison_df = pd.concat([comparison,best_df]).sort_values(by='Balanced Test Score',ascending=False)
    print(f'updated comparison DataFrame \n{comparison_df.head(8)}')
    
    # Plot updated comparison model against initial models
    comparison_df.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    return comparison,models_df,leaves_df,estimators_df,best_results_df,comparison_df,best_train,best_test



""" many_models without min/max leaf search to see if 
    efficiency is worth score difference"""



# run processed data through various models to determine which is best
def many_models_no_leaf(x,y,xt,yt):
    
    # Create initial Random Forest Classifier model
    rf = RandomForestClassifier(random_state=13)
    
    # Fit data to Random forest model
    rf.fit(x,y)
    
    # Make predictions on Random Forest Model
    rf_predict = rf.predict(x)
    rf_test = rf.predict(xt)
    
    # Print scores and classification reports
    print(f'\nRandom Forest \nTest Accuracy: {rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,rf_test)}')
    print(f'classification report: \n {classification_report(yt,rf_test)}')
    
    # Create comparison dataframe of the models with Model Name as Index
    comparison = pd.DataFrame(
        [
            ['Random Forest',rf.score(x,y),rf.score(xt,yt),balanced_accuracy_score(y,rf_predict),balanced_accuracy_score(yt,rf_test)],
        ],
        columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    
    # Make new column calculating difference between training and test scores
    comparison['Balanced Difference'] = (comparison['Balanced Trained Score'] - comparison['Balanced Test Score'])
    
    # Sort by Balanced Test Score column and print results
    comparison = comparison.sort_values(by='Balanced Test Score', ascending=False)
    
    # Drop Balanced Trained Score for easier readability and print output
    # Examples
    # comparison = comparison[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    # comparison = comparison.drop(columns=['Balanced Trained Score'])
    comparison = comparison.drop('Balanced Trained Score',axis=1)
    print(comparison.head(7))
    
    # Plot the comparison dataframe on a bar graph for visualizations
    comparison.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    
    
    # Create checks for Random Forest Best Parameters
    
    # Check best depth for random forest
    models = {'train_score': [], 'test_score': [], 'max_depth': []}
    
    # Loop through range to see where deviation happens
    for depth in range(1,20):
        
        # Save max depth to models
        models['max_depth'].append(depth)
        
        # Create model for max depth 
        model = RandomForestClassifier(max_depth=depth,random_state=13)
        
        # Fit the random forest model
        model.fit(x, y)
        
        # Make max depth predictions
        y_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Save max depth scores to models
        models['train_score'].append(balanced_accuracy_score(y, y_train_pred))
        models['test_score'].append(balanced_accuracy_score(yt, y_test_pred))
        
    # Save models data to dataframe and sort by test scores    
    models_df = pd.DataFrame(models)
    
    # Show graph of for loop
    models_df.plot(title='Random forest max depth', x='max_depth')
    
    # Set max_depth as index
    models_df_tolist = models_df.set_index('max_depth').sort_values(by='test_score',ascending=False)
    models_df = models_df.sort_values(by='test_score',ascending=False)
    print(f'\nRandom Forest Parameter Tuning:\n \nbest depth balanced test score: \n{models_df.head()}')
    
    # Save top 5 to a list
    best_rf = models_df_tolist.head().index.tolist()
    print(f'max depths: \n{best_rf}\n')
    
    
    # Test for best n_estimators
    # Create n_estimators range search
    est = range(2,100,2)
    
    # Create dictionary to store results
    est_models = {'train_score':[],'test_score':[],'n_estimators':[]}
    
    # Loop through range to find best n_estimators
    for e in est:
        
        # Create random forest model
        est_model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],n_estimators=e,random_state=13)
        
        # Fit model 
        est_model.fit(x,y)
        
        # Make predictions for n_estimators
        X_test_pred = est_model.predict(xt)
        y_train_pred = est_model.predict(x)
        
        # Store results in dictionairy
        est_models['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        est_models['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        est_models['n_estimators'].append(e)
        
    # Save to dataframe
    estimators_df = pd.DataFrame(est_models)
    
    # Graph best n_estimators
    estimators_df.plot(title='best n_estimators',x='n_estimators')
    
    # Set index to n_estimators
    estimators_df_tolist = estimators_df.set_index('n_estimators').sort_values(by='test_score',ascending=False)
    estimators_df = estimators_df.sort_values(by='test_score',ascending=False)
    print(f'\nbest n_estimators balanced test score: \n{estimators_df.head()}')
    
    # Save to list
    best_est = estimators_df_tolist.head().index.tolist()
    print(f'best n_estimators \n{best_est}\n')
    
    
    
    # Test for min_split
    split = range(2,60,2)
    splits = {'train_score':[],'test_score':[],'min_split': []}
    
    # Loop through min_split values
    for s in split:
        
        # Create Random Forest Classifier instance with depth
        model = RandomForestClassifier(max_depth=models_df['max_depth'].iloc[0],n_estimators=estimators_df['n_estimators'].iloc[0],min_samples_split=s,random_state=13)
        
        # Fit model
        model.fit(x,y)
        
        # Predict
        X_test_pred = model.predict(xt)
        y_train_pred = model.predict(x)
        
        # Store scores and best amounts in leaves dictionary
        splits['train_score'].append(balanced_accuracy_score(y,y_train_pred))
        splits['test_score'].append(balanced_accuracy_score(yt,X_test_pred))
        splits['min_split'].append(s)
        
    # Store dictionary in a dataframe
    splits_df = pd.DataFrame(splits)
    
    # Plot best min_split on graph
    splits_df.plot(title='Random Forest Best min_split',x='min_split')
    
    # Set split_df index to 'min_split'
    splits_df_tolist = splits_df.set_index('min_split').sort_values(by='test_score',ascending=False)
    splits_df = splits_df.sort_values(by='test_score',ascending=False)
    print(f'best min_split balanced test score: \n{splits_df.head()}')
    
    # Save to list
    best_split = splits_df_tolist.head().index.tolist()
    print(f'best min_split \n{best_split}\n')
    
    
    
    # Create test Random Forest Classifier
    test_tuned_rf = RandomForestClassifier(n_estimators=estimators_df['n_estimators'].iloc[0],max_depth=models_df['max_depth'].iloc[0],min_samples_split=splits_df['min_split'].iloc[0],random_state=13)
    
    # train the tuned model
    test_tuned_rf.fit(x,y)
    
    # predict with the tuned model
    test_tuned_train = test_tuned_rf.predict(x)
    tuned_test = test_tuned_rf.predict(xt)
    
    # view tuned Random Forest Model scores and classification report
    print(f'\nTest best Random Forest Tuned Parameters scores "no leaf" \n')
    print(f'\nTest Best Random Forest Tuned Parameters Scores \nTest Accuracy: {test_tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,tuned_test)}')
    print(f'classification report: \n {classification_report(yt,tuned_test)}')
    
    # add tuned Random Forest Model to the comparison DataFrame and Plot
    test_df = pd.DataFrame(
        [
            ['Test Random Forest',test_tuned_rf.score(x,y),test_tuned_rf.score(xt,yt),balanced_accuracy_score(y,test_tuned_train),balanced_accuracy_score(yt,tuned_test)]
        ],
            columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    test_df['Balanced Difference'] = (test_df['Balanced Trained Score'] - test_df['Balanced Test Score'])
    test_df = test_df[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    comparison_df = pd.concat([comparison,test_df]).sort_values(by='Balanced Test Score',ascending=False)
    print(f'updated comparison DataFrame \n{comparison_df.head(8)}')
    
    comparison_df.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    
    # Save best parameters to dictionary to use to find best combination
    best_params = {
        'n_estimators': best_est,
        'max_depth': best_rf,
        'min_split': best_split,
    }

    # Create dictionary to store results
    best_results = {'n_estimators':[],'max_depth':[],'min_split':[],'train_score':[],'test_score':[]}
    
    # Loop through combinations(refactor in future for efficiency)
    for e in best_params['n_estimators']:
        for m in best_params['max_depth']:
            #for l in best_params['min_samples_leaf']:
            for s in best_params['min_split']:
                # Create random forest model with best parameters
                
                #,min_samples_leaf=l
                rf_model = RandomForestClassifier(n_estimators=e,max_depth=m,min_samples_split=s,random_state=13)
                # Fit data
                rf_model.fit(x,y)
                # Predictions
                test_pred = rf_model.predict(xt)
                train_pred = rf_model.predict(x)
                # Append balanced scores
                best_results['train_score'].append(balanced_accuracy_score(y,train_pred))
                best_results['test_score'].append(balanced_accuracy_score(yt,test_pred))
                best_results['n_estimators'].append(e)
                best_results['max_depth'].append(m)
                best_results['min_split'].append(s)
                    
    # Save best results to dataframe
    best_results_df = pd.DataFrame(best_results).sort_values(by='test_score',ascending=False)
    
    # Plot best_results
    best_results_plot = best_results_df[['test_score','train_score']].sort_values(by='test_score',ascending=False).reset_index(drop=True)
    best_results_plot.plot(title='Tuned Random Forest Balanced Trained/Balance Test',kind='line')
    
    # Set test_score as index
    best_results_df = best_results_df.set_index('test_score')
    print(f'\nbest results:\n{best_results_df}')
    
    # Save best parameters to print
    best_n = best_results_df['n_estimators'].iloc[0]
    best_d = best_results_df['max_depth'].iloc[0]
    best_s = best_results_df['min_split'].iloc[0]
    
    
    print(f'\nTop Best Parameters: \nbest n_estimators: {best_n} \nbest max_depth: {best_d} \nBest min_split: {best_s} \n')
    
    # Create random forest instance with best parameters
    tuned_rf = RandomForestClassifier(n_estimators=best_results_df['n_estimators'].iloc[0], max_depth=best_results_df['max_depth'].iloc[0],min_samples_split=best_results_df['min_split'].iloc[0],random_state=13)    
    
    # Train the tuned model
    tuned_rf.fit(x,y)
    
    # Predict with the tuned model
    best_train = tuned_rf.predict(x)
    best_test = tuned_rf.predict(xt)
    
    # View tuned Random Forest Model scores and classification report
    print(f'best Random Forest Tuned Parameters scores "no leaf" \n')
    print(f'\nBest Random Forest Tuned Parameters Scores \nTest Accuracy: {tuned_rf.score(xt,yt)}\nbalanced test score: {balanced_accuracy_score(yt,best_test)}')
    print(f'classification report: \n {classification_report(yt,best_test)}')
    
    # Add tuned Random Forest Model to the comparison DataFrame and Plot
    best_df = pd.DataFrame(
        [
            ['Tuned Random Forest',tuned_rf.score(x,y),tuned_rf.score(xt,yt),balanced_accuracy_score(y,best_train),balanced_accuracy_score(yt,best_test)]
        ],
            columns=['Model Name','Trained Score', 'Test Score','Balanced Trained Score','Balanced Test Score']
    ).set_index('Model Name')
    best_df['Balanced Difference'] = (best_df['Balanced Trained Score'] - best_df['Balanced Test Score'])
    print('Balanced Trained Score')
    print(best_df['Balanced Trained Score'])
    print('Balanced Test Score')
    print(best_df['Balanced Test Score'])
    best_df = best_df[['Trained Score','Test Score','Balanced Test Score','Balanced Difference']]
    comparison_df = pd.concat([comparison_df,best_df]).sort_values(by='Balanced Test Score',ascending=False)
    print(f'updated comparison DataFrame \n{comparison_df.head(8)}')
    
    # Plot updated comparison model against initial models
    comparison_df.plot(kind='bar').tick_params(axis='x',rotation=45)
    
    return comparison,models_df,estimators_df,best_results_df,comparison_df,best_train,best_test