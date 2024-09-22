import matplotlib.pyplot as plt
import pandas as pd

def plot_importance(model, feature_names: list):
    feature_importances = model.get_feature_importance()
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('CatBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()