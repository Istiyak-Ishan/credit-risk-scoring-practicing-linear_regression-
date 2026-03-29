from src.data_preprocessing import preprocess_data, load_data, split_data
from src.feature_engineering import create_transformer
from src.train import train_model
from src.evaluate import evaluate
df = load_data('data/raw/german_credit.csv')
df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df)
transformer = create_transformer(X_train)
model = train_model(transformer, X_train, y_train)
results = evaluate(model, X_test, y_test)
print(results)