import os
from pycaret.classification import setup, create_model, tune_model, predict_model
import pandas as pd

# 設置 Joblib 的臨時目錄
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:\\Temp'
os.makedirs('C:\\Temp', exist_ok=True)

# 載入數據
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 數據預處理
train_data_cleaned = train_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
train_data_cleaned['Age'].fillna(train_data_cleaned['Age'].median(), inplace=True)
train_data_cleaned['Embarked'].fillna(train_data_cleaned['Embarked'].mode()[0], inplace=True)
train_data_cleaned = pd.get_dummies(train_data_cleaned, drop_first=True)

test_data_cleaned = test_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
test_data_cleaned['Age'].fillna(test_data_cleaned['Age'].median(), inplace=True)
test_data_cleaned['Embarked'].fillna(test_data_cleaned['Embarked'].mode()[0], inplace=True)
test_data_cleaned = pd.get_dummies(test_data_cleaned, drop_first=True)

missing_cols = set(train_data_cleaned.columns) - set(test_data_cleaned.columns)
for col in missing_cols:
    test_data_cleaned[col] = 0
test_data_cleaned = test_data_cleaned[train_data_cleaned.columns.drop("Survived")]

clf_setup = setup(data=train_data_cleaned, target='Survived', session_id=123, preprocess=True)

model = create_model('gbc')
tuned_model = tune_model(model)
predictions = predict_model(tuned_model, data=test_data_cleaned)

# 檢查列名
print(predictions.columns)

# 將預測結果保存為 CSV
output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions["prediction_label"]  # 替換為正確的列名
})
output.to_csv("HW4-1.csv", index=False)

print("預測結果已保存為 HW4-1.csv")
