import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

class HealthcarePreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def load_data(self):
        """Load healthcare data"""
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
            if col != 'charges':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le
                print(f"Encoded {col}")
        
        return self.df
    
    def split_data(self, test_size=0.2):
        """Split data into train and test sets"""
        X = self.df.drop('charges', axis=1)
        y = self.df['charges']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
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
            pickle.dump({'scaler': self.scaler, 'encoders': self.encoders}, f)
        print(f"Preprocessor saved to {filepath}")

if __name__ == "__main__":
    preprocessor = HealthcarePreprocessor('data/healthcare_data.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
