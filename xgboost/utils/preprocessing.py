from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

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

def build_xgb_preprocessor():
    cat_feats, num_feats = get_features()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_feats),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_feats)
        ],
        remainder='drop'
    )
    return preprocessor