from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
