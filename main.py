import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA

# Load dataset
df = sns.load_dataset('tips')

X = df[['tip','sex','smoker','day','time','size']]
y = df['total_bill']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Label encode binary categorical columns
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_time = LabelEncoder()

for col, le in zip(['sex', 'smoker', 'time'], [le_sex, le_smoker, le_time]):
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Column transformer: OneHotEncode 'day' and scale numeric features
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), [3]),  # 'day' column index
        ('num', StandardScaler(), [0, 5])  # 'tip' and 'size' columns
    ],
    remainder='passthrough'  # keep other columns (already label encoded) as-is
)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# SVR model
svr = SVR()
svr.fit(X_train, y_train)

# Predictions and evaluation
y_pred = svr.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"SVR R2: {r2:.3f}, MAE: {mae:.3f}")

# GridSearch for best parameters
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}
grid = GridSearchCV(SVR(), param_grid, refit=True)
grid.fit(X_train, y_train)
best_params = grid.best_params_
print("Best parameters from GridSearch:", best_params)

grid_predictions = grid.predict(X_test)
r2_grid = r2_score(y_test, grid_predictions)
mae_grid = mean_absolute_error(y_test, grid_predictions)
print(f"GridSearch SVR R2: {r2_grid:.3f}, MAE: {mae_grid:.3f}")

# PCA for 2D visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# Train SVR on 2D projection
svr_2d = SVR(kernel='linear')
svr_2d.fit(X_train_2d, y_train)
y_pred_2d = svr_2d.predict(X_test_2d)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(X_test_2d[:,0], X_test_2d[:,1], c=y_test, cmap='viridis', label='Actual', s=50)
plt.scatter(X_test_2d[:,0], X_test_2d[:,1], c=y_pred_2d, cmap='cool', marker='x', label='Predicted', s=50)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('SVR Hyperplane (2D PCA projection)')
plt.legend()
plt.show()
