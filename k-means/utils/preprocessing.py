from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_features():
    """Definisce quali sono le feature numeriche e categoriche."""
    categorical_features = ['age', 'career_field', 'communication_style']
    numeric_features = [
        'education', 'career_ambition', 'openness', 'extraversion', 
        'agreeableness', 'conscientiousness', 'chronotype', 
        'spontaneity', 'emotional_intelligence'
    ]
    return categorical_features, numeric_features

def build_preprocessor():
    """Crea e restituisce il trasformatore delle colonne."""
    cat_feats, num_feats = get_features()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_feats)
        ])
    return preprocessor