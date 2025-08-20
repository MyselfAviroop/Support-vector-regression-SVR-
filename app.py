# # # import numpy as np
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns

# # # from sklearn.datasets import make_classification
# # # from sklearn.model_selection import train_test_split, GridSearchCV
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.svm import SVC
# # # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # # # Generate dataset
# # # X, y = make_classification(
# # #     n_samples=1000, n_features=2, n_classes=2, 
# # #     n_clusters_per_class=2, n_redundant=0, random_state=42
# # # )

# # # # Train-test split
# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # # Standardization
# # # scaler = StandardScaler()
# # # X_train = scaler.fit_transform(X_train)
# # # X_test = scaler.transform(X_test)

# # # # =============================
# # # # Linear Kernel
# # # model = SVC(kernel='linear')
# # # model.fit(X_train, y_train)
# # # y_pred = model.predict(X_test)
# # # print("\n=== Linear Kernel ===")
# # # print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# # # print(classification_report(y_test, y_pred))
# # # print(confusion_matrix(y_test, y_pred))

# # # # =============================
# # # # RBF Kernel
# # # rbf = SVC(kernel='rbf')
# # # rbf.fit(X_train, y_train)
# # # y_pred_rbf = rbf.predict(X_test)
# # # print("\n=== RBF Kernel ===")
# # # print(f"Accuracy: {accuracy_score(y_test, y_pred_rbf):.2f}")
# # # print(classification_report(y_test, y_pred_rbf))
# # # print(confusion_matrix(y_test, y_pred_rbf))

# # # # =============================
# # # # Polynomial Kernel
# # # poly = SVC(kernel='poly')
# # # poly.fit(X_train, y_train)
# # # y_pred_poly = poly.predict(X_test)
# # # print("\n=== Polynomial Kernel ===")
# # # print(f"Accuracy: {accuracy_score(y_test, y_pred_poly):.2f}")
# # # print(classification_report(y_test, y_pred_poly))
# # # print(confusion_matrix(y_test, y_pred_poly))

# # # # =============================
# # # # Sigmoid Kernel
# # # sigmoid = SVC(kernel='sigmoid')
# # # sigmoid.fit(X_train, y_train)
# # # y_pred_sigmoid = sigmoid.predict(X_test)
# # # print("\n=== Sigmoid Kernel ===")
# # # print(f"Accuracy: {accuracy_score(y_test, y_pred_sigmoid):.2f}")
# # # print(classification_report(y_test, y_pred_sigmoid))
# # # print(confusion_matrix(y_test, y_pred_sigmoid))

# # # # =============================
# # # # Hyperparameter tuning for RBF
# # # param_grid = {
# # #     'C': [0.1, 1, 10, 100],
# # #     'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
# # #     'kernel': ['rbf']
# # # }
# # # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, n_jobs=1, cv=5)

# # # grid.fit(X_train, y_train)

# # # print("\n=== Grid Search Results ===")
# # # print("Best parameters found:", grid.best_params_)
# # # print("Best estimator found:", grid.best_estimator_)

# # # # Evaluate best model on test data
# # # best_model = grid.best_estimator_
# # # y_pred_best = best_model.predict(X_test)
# # # print("\n=== Best Model on Test Data ===")
# # # print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")
# # # print(classification_report(y_test, y_pred_best))
# # # print(confusion_matrix(y_test, y_pred_best))


# # # #prediction
# # # y_pred2=grid.predict(X_test)
# # # print("\n=== Predictions on Test Data ===")
# # # print(y_pred2)
# # # print("claasification report:",classification_report(y_test,y_pred2))
# # # print("confusion matrix:",confusion_matrix(y_test,y_pred2))
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # from sklearn.datasets import make_classification
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.svm import SVC
# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # # Generate dataset
# # X, y = make_classification(
# #     n_samples=1000, n_features=2, n_classes=2, 
# #     n_clusters_per_class=2, n_redundant=0, random_state=42
# # )

# # # Train-test split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Standardization
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)

# # # =============================
# # # Train SVM with Linear Kernel
# # linear_model = SVC(kernel='linear')
# # linear_model.fit(X_train, y_train)

# # # =============================
# # # Helper function to plot decision boundary
# # def plot_decision_boundary(model, X, y, title):
# #     plt.figure(figsize=(8,6))
# #     # Scatter plot of the data
# #     plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", s=30, edgecolors="k")

# #     # Create grid
# #     ax = plt.gca()
# #     xlim = ax.get_xlim()
# #     ylim = ax.get_ylim()
# #     xx, yy = np.meshgrid(
# #         np.linspace(xlim[0], xlim[1], 200),
# #         np.linspace(ylim[0], ylim[1], 200)
# #     )
# #     grid = np.c_[xx.ravel(), yy.ravel()]
# #     Z = model.decision_function(grid).reshape(xx.shape)

# #     # Plot decision boundary and margins
# #     ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")        # hyperplane
# #     ax.contour(xx, yy, Z, levels=[-1, 1], linestyles=["--","--"], colors="black")  # margins

# #     plt.title(title)
# #     plt.show()

# # # =============================
# # # Plot Linear Kernel Hyperplane
# # plot_decision_boundary(linear_model, X_train, y_train, "SVM with Linear Kernel")


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# x=np.linspace(-5,5,100)
# y=np.sqrt(10**2-x**2)
# y=np.hstack((y,-y))
# x=np.hstack((x,-x))

# x1=np.linspace(-5,5,100)
# y1=np.sqrt(5**2-x1**2)
# y1=np.hstack((y1,-y1))
# x1=np.hstack((x1,-x1))

# plt.scatter(y,x)
# plt.scatter(y1,x1)
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Scatter Plot of Points")
# # plt.grid()
# # plt.show()

# import pandas as pd
# df1=pd.DataFrame(np.vstack([y,x]).T,columns=["X1","X2"])
# df1['Y']=0
# df2=pd.DataFrame(np.vstack([y1,x1]).T,columns=["X1","X2"])
# df2['Y']=1
# df = pd.concat([df1, df2], ignore_index=True)
# print(df.head())

# #dependent and independent variables
# X=df.iloc[:, :-2]
# y=df.Y


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #polynomial kernel
# df['X1_square']=df['X1']**2
# df['X2_square']=df['X2']**2
# df['X1*X2']=df['X1']*df['X2']
# print(df.head())

# #dependent and independent variables
# X = df[['X1', 'X2', 'X1_square', 'X2_square', 'X1*X2']]

# y=df['Y']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# import plotly.express as px
# fig = px.scatter_3d(
#     df, x='X1', y='X2', z='X1*X2',
#     color='Y',
#     title='3D Scatter Plot of Points'
# )
# fig.show()
# # Option 2 (backup): save as HTML file
# fig.write_html("scatter_plot.html")
# print("Plot saved as scatter_plot.html – open this file in your browser.")

# fig = px.scatter_3d(
#     df, x='X1_square', y='X1_square', z='X1*X2',
#     color='Y',
#     title='3D Scatter Plot of Points'
# )
# fig.show()
# fig.write_html("scatter_plot2.html")
# print("Plot saved as scatter_plot.html – open this file in your browser.")



# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# classifier=SVC(kernel='linear')




