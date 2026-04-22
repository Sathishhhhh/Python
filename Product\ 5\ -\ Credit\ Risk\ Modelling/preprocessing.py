import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import pickle

class CreditRiskPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.encoders = {}
        self.class_weights = None
        
    def load_data(self):
        """Load credit risk data"""
        self.df = pd.read_csv(self.filepath)
        print(f"Data loaded: {self.df.shape}")
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values"""
        self.df = self.df.dropna()
        print(f"After removing missing values: {self.df.shape}")
        return self.df
    
    def encode_categorical(self):
        """Encode categorical variables"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le
            print(f"Encoded {col}")
        
        return self.df
    
    def split_data(self, test_size=0.2):
        """Split data into train and test sets"""
        X = self.df.drop('Default', axis=1)
        y = self.df['Default']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Calculate class weights for imbalanced data
        self.class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        self.class_weight_dict = dict(enumerate(self.class_weights))
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Class weights: {self.class_weight_dict}")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Features scaled")
        return X_train_scaled, X_test_scaled
    
    def preprocess(self, test_size=0.2):
        """Complete preprocessing pipeline"""
        self.load_data()
        self.handle_missing_values()
        self.encode_categorical()
        X_train, X_test, y_train, y_test = self.split_data(test_size)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, filepath):
        """Save scaler and encoders"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'encoders': self.encoders,
                'class_weights': self.class_weight_dict
            }, f)
        print(f"Preprocessor saved to {filepath}")

if __name__ == "__main__":
    preprocessor = CreditRiskPreprocessor('data/credit_risk_data.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
