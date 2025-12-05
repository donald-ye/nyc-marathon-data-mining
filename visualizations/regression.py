import matplotlib.pyplot as plt
import seaborn as sns

model_importances = {
    'HistGB': feature_importances_hist_percent,
    'CatBoost': feature_importances_cat,
    'GradientBoosting': importances_grad_percent
}

# Create a pandas Series for GradientBoostingRegressor feature importances
grad_fi_series = pd.Series(importances_grad_percent, index=feature_names_grad)

# Create a pandas Series for CatBoostRegressor feature importances
cat_fi_series = pd.Series(feature_importances_cat, index=X_train.columns)

# Update the model_importances dictionary with correctly formatted Series
model_importances = {
    'HistGB': feature_importances_hist_percent,
    'CatBoost': cat_fi_series,
    'GradientBoosting': grad_fi_series
}
print("Feature importances dictionary updated successfully.")

top_n = 5

for model_name, fi in model_importances.items():
    # Sort the feature importances and take the top N
    fi_sorted = fi.sort_values(ascending=False).head(top_n)
    # remove prefixes for clarity
    fi_sorted.index = fi_sorted.index.str.replace('num__', '').str.replace('cat__', '')

    plt.figure(figsize=(10, max(6, 0.5 * top_n))) # Adjust figure size dynamically
    fi_sorted.plot(kind='barh', color='skyblue')
    plt.gca().invert_yaxis() # Display the most important feature at the top

    for i, value in enumerate(fi_sorted.values):
      plt.text(value, i, ' {:.2f}%'.format(value), va='center', fontsize=9)

    plt.xlim(0, fi_sorted.max() * 1.1)
    plt.title(f"Top {top_n} Features — {model_name}", fontsize=14)
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()

# plot actual vs. predicted finish time for each model

model_preds = {
    "CatBoost": preds_cat,
    "HistGradientBoosting": preds_hist,
    "GradientBoosting": preds_grad
}


for model_name, preds in model_preds.items():
    plt.figure(figsize=(8,6))
    # plot actual vs predicted finish time
    plt.scatter(y_test, preds, alpha=0.5)
    # make the "perfect prediction" line
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2)
    plt.xlabel("Actual Finish Time (seconds)")
    plt.ylabel("Predicted Finish Time (seconds)")
    plt.title(f"Actual vs Predicted Finish Time — {model_name}")
    plt.tight_layout()
    plt.show()

# plot the same thing as above, but separated by gender

for model_name, preds in model_preds.items():
  sns.scatterplot(x=y_test, y=preds, hue=X_test['Gender'])
  plt.plot([y_test.min(), y_test.max()],
          [y_test.min(), y_test.max()],
          'r--', lw=2)
  plt.xlabel("Actual Finish Time (seconds)")
  plt.ylabel("Predicted Finish Time (seconds)")
  plt.title(f"Actual vs Predicted Finish Time — {model_name}")
  plt.show()
