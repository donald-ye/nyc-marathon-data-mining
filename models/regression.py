!pip install catboost

import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score


first_half_pace_cols = [
    'split_pace_5K', 'split_pace_10K', 'split_pace_15K',
    'split_pace_20K', 'split_pace_HALF'
]

df = final_df.copy()

# fade and ratio between consecutive splits
for i in range(len(first_half_pace_cols)-1):
    col_prev = first_half_pace_cols[i]
    col_next = first_half_pace_cols[i+1]
    df[f'fade_{col_prev}_{col_next}'] = df[col_next] - df[col_prev]
    df[f'ratio_{col_next}_{col_prev}'] = df[col_next] / df[col_prev]

fade_cols = [c for c in df.columns if c.startswith('fade_')]
ratio_cols = [c for c in df.columns if c.startswith('ratio_')]

# adding some more statistical features to try and improve model performance
df['avg_pace_first_half'] = df[first_half_pace_cols].mean(axis=1)
df['cv_first_half'] = df[first_half_pace_cols].std(axis=1) / df['avg_pace_first_half']
df['min_pace_first_half'] = df[first_half_pace_cols].min(axis=1)
df['max_pace_first_half'] = df[first_half_pace_cols].max(axis=1)
df['range_pace_first_half'] = df['max_pace_first_half'] - df['min_pace_first_half']

feature_cols = first_half_pace_cols + fade_cols + ratio_cols + \
               ['avg_pace_first_half','cv_first_half','min_pace_first_half',
                'max_pace_first_half','range_pace_first_half',
                'Age','Gender','Country']

target_col = 'OverallTime'

# assign X and y
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) # we use test size 0.2

# A. GradientBoostingRegressor
# encode the variables
numeric_cols = [c for c in first_half_pace_cols] + ['Age']
categorical_cols = ['Gender', 'Country']

numeric_transformer_grad = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer_grad = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_grad = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_grad, numeric_cols),
        ("cat", categorical_transformer_grad, categorical_cols)
    ]
)

# initialize the model pipeline!

model_grad = Pipeline(steps=[
    ('preprocess', preprocessor_grad),
    ('regressor', GradientBoostingRegressor())
])

# fit the model
model_grad.fit(X_train, y_train)
preds_grad = model_grad.predict(X_test)
rmse_grad = np.sqrt(mean_squared_error(y_test, preds_grad))

print("RMSE for GradientBoostingRegressor (seconds):", rmse_grad)
print("RMSE for GBR (minutes):", rmse_grad / 60)

model_grad['regressor'].feature_importances_
# now we can see the feature importances.
feature_names_grad = model_grad.named_steps['preprocess'].get_feature_names_out()
importances_grad = model_grad.named_steps['regressor'].feature_importances_
importances_grad_percent = (importances_grad / importances_grad.sum()) * 100

for name, val in sorted(zip(feature_names_grad, importances_grad_percent), key=lambda x: x[1], reverse=True):
    print(f"{name}: {val:.4f}")

# B. HistGradientBoostingRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# reinitialize the transformers.

numeric_transformer_hist = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer_hist = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_hist = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_hist, numeric_cols),
        ('cat', categorical_transformer_hist, categorical_cols)
    ]
)

model_hist = Pipeline(steps=[
    ('preprocessor', preprocessor_hist),
    ('regressor', HistGradientBoostingRegressor())
])

# fit model
model_hist.fit(X_train, y_train)

# transform data, get regressor and feature names
X_train_trans = model_hist.named_steps['preprocessor'].transform(X_train)
X_test_trans  = model_hist.named_steps['preprocessor'].transform(X_test)

reg = model_hist.named_steps['regressor']

feature_names = model_hist.named_steps['preprocessor'].get_feature_names_out()

print("matching lengths check:")
print("X_test_trans:", X_test_trans.shape)
print("len(feature_names):", len(feature_names))

# compute permutation importance
perm = permutation_importance(
    reg,
    X_test_trans,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

print("Permutation result count:", perm.importances_mean.shape[0])


# make sure importances match number of names
assert perm.importances_mean.shape[0] == len(feature_names), \
    "importances do not equal features. double check the model initialization."

# build labeled series
feature_importances_hist = pd.Series(
    perm.importances_mean,
    index=feature_names
).sort_values(ascending=False)

print(feature_importances_hist)

# convert permutation importances to percentages
positive_importances = feature_importances_hist[feature_importances_hist > 0]
total_importance = positive_importances.sum()
feature_importances_hist_percent = (positive_importances / total_importance) * 100
feature_importances_hist_percent = feature_importances_hist_percent.sort_values(ascending=False)

print(feature_importances_hist_percent)
preds_hist = model_hist.predict(X_test)
rmse_hist = np.sqrt(mean_squared_error(y_test, preds_hist))
print("RMSE for Hist. (seconds):", rmse_hist)
print("RMSE for Hist. (minutes):", rmse_hist / 60)

# CatBoostRegressor 
cat_features = ['Gender','Country']

for col in cat_features:
    most_freq = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(most_freq).astype(str)
    X_test[col] = X_test[col].fillna(most_freq).astype(str)

# create catboost pool (handles categorical features)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# initialize model
model_cat = CatBoostRegressor(
    iterations=1000,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

# Fit model
model_cat.fit(train_pool)

# predict and evaluate. here we see that this is the best-performing model of the three!
preds_cat = model_cat.predict(test_pool)
rmse_cat = np.sqrt(mean_squared_error(y_test, preds_cat))
print("RMSE for Cat. (seconds):", rmse_cat)
print("RMSE for Cat. (minutes):", rmse_cat/60)

# feature importances
feature_importances_cat = model_cat.get_feature_importance(train_pool)
importance_df_cat = pd.DataFrame({'feature': X_train.columns, 'importance': feature_importances_cat})\
                    .sort_values('importance', ascending=False)
print(importance_df_cat.head(15))
