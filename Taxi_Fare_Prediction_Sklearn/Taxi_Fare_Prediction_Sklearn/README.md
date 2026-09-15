# Taxi Fare Prediction using Scikit-learn

A beginner-friendly machine learning project that predicts taxi fares from trip distance, passenger count, and pickup hour.

## Tech stack
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

## How to Run

For complete instructions with Windows tips and customization options, see [HOW_TO_RUN.md](HOW_TO_RUN.md).

### Quick Start:
1. **Navigate to project folder**:
   ```powershell
   cd "c:\Users\user\Desktop\jj college\Taxi_Fare_Prediction_Sklearn"
   ```
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Train the model**:
   ```powershell
   python train_model.py
   ```
   *(Saves model to `models/taxi_fare_model.pkl` and chart to `outputs/feature_importance.png`)*
4. **Make a prediction**:
   ```powershell
   python predict.py
   ```

## Dataset
The project includes a small synthetic dataset (`data/taxi_fares.csv`) so it can run immediately without downloading an external dataset.

