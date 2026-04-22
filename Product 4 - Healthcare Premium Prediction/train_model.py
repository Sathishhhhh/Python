import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from preprocessing import HealthcarePreprocessor

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
    
    def train_linear_regression(self, X_train, y_train):
        """Train linear regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['Linear Regression'] = model
        print("Linear Regression trained")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train random forest model"""
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        print("Random Forest trained")
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train gradient boosting model"""
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = model
        print("Gradient Boosting trained")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  R² Score: {r2:.4f}")
        
        return self.metrics[model_name]
    
    def get_best_model(self):
        """Get best performing model based on R² score"""
        best_model_name = max(self.metrics, key=lambda x: self.metrics[x]['R2'])
        best_model = self.models[best_model_name]
        print(f"\nBest Model: {best_model_name}")
        return best_model, best_model_name
    
    def save_model(self, model, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")

def main():
    # Preprocess data
    preprocessor = HealthcarePreprocessor('data/healthcare_data.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_linear_regression(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    trainer.train_gradient_boosting(X_train, y_train)
    
    # Evaluate models
    for model_name, model in trainer.models.items():
        trainer.evaluate_model(model, X_test, y_test, model_name)
    
    # Save best model
    best_model, best_model_name = trainer.get_best_model()
    trainer.save_model(best_model, 'models/best_model.pkl')
    
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()
