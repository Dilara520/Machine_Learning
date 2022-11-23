from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus


clf = DecisionTreeClassifier()

# fit the model
clf.fit(X_train, y_train)

y_pred_dt = clf.predict(X_test)

print('Model accuracy score on test with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_dt)))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

#One way to prevent overfitting - max depth

#Decision Tree model with default criterion but max_depth=3
#ModelDefinition
clf2 = DecisionTreeClassifier(max_depth=3)

#fit the model and show the accuracy 
clf2.fit(X_train,y_train)

y_pred_dt2 = clf2.predict(X_test)

print('Model accuracy : {0:0.4f}'. format(accuracy_score(y_test, y_pred_dt2)))

#Plot the tree
dot_data = StringIO()
export_graphviz(clf2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

#Decision Tree model with max_depth 5
#ModelDefinition
clf3 = DecisionTreeClassifier(max_depth=5)

#fit the model and show the accuracy 
clf3.fit(X_train,y_train)

y_pred_dt3 = clf3.predict(X_test)

print('Model accuracy : {0:0.4f}'. format(accuracy_score(y_test, y_pred_dt3)))

#Plot the tree
dot_data = StringIO()
export_graphviz(clf3, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

#Decision Tree model with max_depth is 5 and random_state is 5
clf4 = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=5)

clf4.fit(X_train,y_train)

y_pred_dt4 = clf4.predict(X_test)

print('Model accuracy : {0:0.4f}'. format(accuracy_score(y_test, y_pred_dt4)))

#Plot the tree
dot_data = StringIO()
export_graphviz(clf4, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

#cross validation methods
model = DecisionTreeClassifier()

scores = cross_val_score(model, X, y, cv=5)
print(scores)

print("%0.2f accuracy" % scores.mean())