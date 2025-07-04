import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your data (make sure flooddata.csv is in your project folder)
df = pd.read_csv('flooddata.csv')
X = df[['total_rainfall', 'max_daily_rainfall', 'mean_daily_rainfall', 'duration']]
y = df['is_flood']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model
with open('flood_model.pkl', 'wb') as f:
    pickle.dump(model, f)
