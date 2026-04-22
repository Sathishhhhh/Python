import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import pickle
from preprocessing import CreditRiskPreprocessor

class CreditRiskModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.predictions = {}
    
    def train_logistic_regression(self, X_train, y_train, class_weights):
        """Train logistic regression model"""
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight=class_weights
        )
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        print("Logistic Regression trained")
        return model
    
    def train_random_forest(self, X_train, y_train, class_weights):
        """Train random forest classifier"""
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight=class_weights
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        print("Random Forest trained")
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train gradient boosting classifier"""
        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = model
        print("Gradient Boosting trained")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate classification model"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        cm = confusion_matrix(y_test, y_pred)
        
        self.metrics[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Confusion_Matrix': cm
        }
        
        self.predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        return self.metrics[model_name]
    
    def get_best_model(self, metric='ROC-AUC'):
        """Get best performing model"""
        best_model_name = max(self.metrics, key=lambda x: self.metrics[x][metric])
        best_model = self.models[best_model_name]
        print(f"\nBest Model: {best_model_name} (by {metric}: {self.metrics[best_model_name][metric]:.4f})")
        return best_model, best_model_name
    
    def save_model(self, model, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")
    
    def print_model_comparison(self):
        """Print comparison table of all models"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        metrics_df = pd.DataFrame(self.metrics).T
        print(metrics_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']])

def main():
    # Preprocess data
    preprocessor = CreditRiskPreprocessor('data/credit_risk_data.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Get class weights
    class_weights_dict = preprocessor.class_weight_dict
    
    # Train models
    trainer = CreditRiskModelTrainer()
    trainer.train_logistic_regression(X_train, y_train, class_weights_dict)
    trainer.train_random_forest(X_train, y_train, class_weights_dict)
    trainer.train_gradient_boosting(X_train, y_train)
    
    # Evaluate models
    for model_name, model in trainer.models.items():
        trainer.evaluate_model(model, X_test, y_test, model_name)
    
    # Print comparison
    trainer.print_model_comparison()
    
    # Save best model
    best_model, best_model_name = trainer.get_best_model()
    trainer.save_model(best_model, 'models/best_model.pkl')
    
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()
