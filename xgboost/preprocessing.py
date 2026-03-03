import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost_config import get_xgb_model 

base_dir = os.path.dirname(os.path.abspath(__file__))

resource_dir = os.path.join(base_dir, "resources")

os.makedirs(resource_dir, exist_ok=True)

# Percorsi completi per input e output
FILENAME = os.path.join(resource_dir, 'social_matcher.csv')
MODEL_NAME = os.path.join(resource_dir, 'social_matcher_model.pkl')
X_TEST_NAME = os.path.join(resource_dir, 'X_test.pkl')
Y_TEST_NAME = os.path.join(resource_dir, 'y_test.pkl')

# 1. Funzione per separare le feature numeriche da quelle categoriche
def get_features():
    categorical_features = [
        'a_age', 'b_age', 
        'a_career_field', 'b_career_field', 
        'a_communication_style', 'b_communication_style'
    ]
    
    numeric_features = [
        'a_education', 'b_education',
        'a_career_ambition', 'b_career_ambition', 
        'a_openness', 'b_openness', 
        'a_extraversion', 'b_extraversion', 
        'a_agreeableness', 'b_agreeableness', 
        'a_conscientiousness', 'b_conscientiousness', 
        'a_chronotype', 'b_chronotype', 
        'a_spontaneity', 'b_spontaneity', 
        'a_emotional_intelligence', 'b_emotional_intelligence'
    ]
    return categorical_features, numeric_features

# 2. Funzione per creare il preprocessore
def build_preprocessor():
    cat_feats, num_feats = get_features()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_feats)
        ],
        remainder='drop' # Assicura che altre colonne non specificate vengano rimosse
    )
    return preprocessor
