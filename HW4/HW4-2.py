import os
import pandas as pd
from pycaret.classification import setup, compare_models, blend_models, tune_model, finalize_model, predict_model

# 設置 Joblib 的臨時目錄
os.environ["JOBLIB_TEMP_FOLDER"] = "C:\\Temp"
os.makedirs("C:\\Temp", exist_ok=True)

# Step 1: Load the dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Step 2: Feature Engineering
train_data_cleaned = train_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
train_data_cleaned['Age'].fillna(train_data_cleaned['Age'].median(), inplace=True)
train_data_cleaned['Embarked'].fillna(train_data_cleaned['Embarked'].mode()[0], inplace=True)
train_data_cleaned = pd.get_dummies(train_data_cleaned, drop_first=True)

test_data_cleaned = test_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
test_data_cleaned['Age'].fillna(test_data_cleaned['Age'].median(), inplace=True)
test_data_cleaned['Embarked'].fillna(test_data_cleaned['Embarked'].mode()[0], inplace=True)
test_data_cleaned = pd.get_dummies(test_data_cleaned, drop_first=True)

# 確保測試數據與訓練數據的特徵一致
missing_cols = set(train_data_cleaned.columns) - set(test_data_cleaned.columns)
for col in missing_cols:
    test_data_cleaned[col] = 0
test_data_cleaned = test_data_cleaned[train_data_cleaned.columns.drop("Survived")]

# Step 3: PyCaret Setup
clf_setup = setup(
    data=train_data_cleaned,
    target="Survived",
    session_id=123,
    preprocess=True,
    n_jobs=1
)

# Step 4: Model Selection and Blending
best_models = compare_models(n_select=5)
blended_model = blend_models(best_models)

# Step 5: Hyperparameter Tuning (Optimize Precision)
tuned_model = tune_model(blended_model, optimize="Precision")

# Finalize the model
final_model = finalize_model(tuned_model)

# Step 6: Prediction on Test Data
predictions = predict_model(final_model, data=test_data_cleaned)

# Save Predictions to CSV
output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions["prediction_label"]
})
output.to_csv("HW4-2.csv", index=False)

print("預測結果已保存為 HW4-2.csv")
