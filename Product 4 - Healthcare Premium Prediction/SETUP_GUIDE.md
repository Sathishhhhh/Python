# 🚀 Quick Start Guide - Healthcare Premium Prediction

Follow these steps to get the project running on your machine.

## Step 1️⃣: Install Dependencies

Open your terminal and navigate to the project directory:

```bash
cd "Project 4 - Healthcare Premium Prediction"
pip install -r requirements.txt
```

## Step 2️⃣: Generate Sample Data

Create a sample healthcare dataset:

```bash
python data/generate_sample_data.py
```

**Expected Output:**
```
Sample data generated: 300 records
   age     sex    bmi  children smoker      region      charges
0   45    male  25.3         2     no    northeast    15234.50
...
```

This creates `data/healthcare_data.csv` with 300 sample records.

---

## Step 3️⃣: Perform Exploratory Data Analysis (Optional)

Visualize the data and generate plots:

```bash
python eda.py
```

**Output:** Creates `plots/` folder with:
- `01_distributions.png` - Feature distributions
- `02_categorical.png` - Categorical variables
- `03_correlation.png` - Correlation heatmap
- `04_features_vs_target.png` - Features vs Premium

---

## Step 4️⃣: Train Machine Learning Models

Train all three regression models:

```bash
python train_model.py
```

**Expected Output:**
```
Data loaded: (300, 6)
After removing missing values: (300, 6)
Encoded sex
Encoded smoker
Encoded region
Features scaled
Linear Regression trained
Random Forest trained
Gradient Boosting trained

Linear Regression Performance:
  RMSE: $3,456.23
  MAE: $2,123.45
  R² Score: 0.7532

Random Forest Performance:
  RMSE: $2,234.12
  MAE: $1,456.78
  R² Score: 0.8934

Gradient Boosting Performance:
  RMSE: $2,012.45
  MAE: $1,234.56
  R² Score: 0.9123

Best Model: Gradient Boosting

✓ Training complete!
```

**Files Created:**
- `models/best_model.pkl` - Best performing model
- `models/preprocessor.pkl` - Scaler and encoders

---

## Step 5️⃣: Launch Streamlit Web Application

Start the interactive web interface:

```bash
streamlit run app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open your browser and navigate to `http://localhost:8501`

---

## 🎮 Using the Web Application

1. **Adjust Patient Parameters:**
   - Drag sliders for age and BMI
   - Select options from dropdowns
   
2. **View Prediction:**
   - The estimated annual premium updates in real-time
   
3. **Read Insights:**
   - Get health recommendations based on input values
   - See risk categorization (smoker, BMI, age)

---

## 📊 Project Workflow

```
┌─────────────────────────────┐
│  Generate Sample Data       │ (data/generate_sample_data.py)
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Exploratory Data Analysis  │ (eda.py) [OPTIONAL]
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Preprocess Data            │ (preprocessing.py - auto)
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Train Models               │ (train_model.py)
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Save Best Model            │ (models/best_model.pkl)
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Launch Web Application     │ (app.py)
└─────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Problem: Module not found error
**Solution:** Ensure all packages are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Problem: Model files not found
**Solution:** Run training first:
```bash
python train_model.py
```

### Problem: Streamlit app won't load
**Solution:** Make sure you're in the correct directory:
```bash
cd "Project 4 - Healthcare Premium Prediction"
streamlit run app.py
```

### Problem: Data file not found
**Solution:** Generate sample data first:
```bash
python data/generate_sample_data.py
```

---

## 📈 Model Performance

The project trains three regression models:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | ~$3,500 | ~$2,100 | ~0.75 |
| Random Forest | ~$2,200 | ~$1,500 | ~0.89 |
| **Gradient Boosting** | ~$2,000 | ~$1,200 | ~**0.91** |

*Note: Exact values depend on random data generation*

---

## 💾 Project Files Structure

```
Project 4 - Healthcare Premium Prediction/
├── data/
│   ├── generate_sample_data.py
│   └── healthcare_data.csv (created after Step 2)
├── models/
│   ├── best_model.pkl (created after Step 4)
│   └── preprocessor.pkl (created after Step 4)
├── plots/ (created after Step 3 if EDA run)
│   ├── 01_distributions.png
│   ├── 02_categorical.png
│   ├── 03_correlation.png
│   └── 04_features_vs_target.png
├── preprocessing.py
├── train_model.py
├── eda.py
├── app.py
├── requirements.txt
├── SETUP_GUIDE.md (this file)
└── README.md
```

---

## 📚 Next Steps After Setup

1. **Customize Input Ranges:** Edit `app.py` to adjust slider min/max values
2. **Add More Features:** Modify `data/generate_sample_data.py` to include new variables
3. **Try Different Models:** Edit `train_model.py` to add more algorithms
4. **Deploy Online:** Use Streamlit Cloud for free hosting
5. **Real Data:** Replace sample data with actual healthcare dataset

---

## ✅ Verification Checklist

After completing all steps:

- [ ] Dependencies installed successfully
- [ ] Sample data generated (300 records)
- [ ] EDA plots created (optional)
- [ ] Models trained and saved
- [ ] Streamlit app running at localhost:8501
- [ ] Can input patient data and get predictions
- [ ] Health insights display correctly

---

## 🎓 What You've Learned

✓ Machine Learning Pipeline
✓ Data Preprocessing & Feature Engineering
✓ Multiple Regression Algorithms
✓ Model Evaluation & Comparison
✓ Web Application Development
✓ Model Serialization & Deployment

---

**Enjoy! 🎉 Happy Predicting!**
