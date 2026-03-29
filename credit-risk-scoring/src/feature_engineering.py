from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
def create_transformer(X):
    categorical_cols = [
        'checking_status', 'credit_history', 'purpose',
        'savings_status', 'employment', 'personal_status',
        'other_parties', 'property_magnitude',
        'other_payment_plans', 'housing', 'job',
        'own_telephone', 'foreign_worker'
    ]
    numerical_cols = [
        'duration', 'credit_amount', 'installment_commitment',
        'residence_since', 'age', 'existing_credits',
        'num_dependents'
    ]
    transformer = ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num','passthrough',numerical_cols)
    ])
    return transformer
