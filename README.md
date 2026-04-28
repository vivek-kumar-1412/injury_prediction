# Football Injury Prediction System

A machine-learning powered web application that predicts whether a football player
is at **High Risk** or **Low Risk** of injury based on 10 biometric and performance
features.

## Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **scikit-learn** — Pre-trained Gradient Boosting model
- **joblib** — Model serialisation
- **pandas / numpy** — Data handling
- **plotly** — Feature importance visualisation

## Project Structure

```
/project
  models/
    injury_model.pkl      # Pre-trained model (do NOT retrain)
  app.py                  # Streamlit application
  create_model.py         # One-time model generation script
  requirements.txt        # Python dependencies
  README.md               # This file
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

## Input Features

| # | Feature             | Range     | Unit       |
|---|---------------------|-----------|------------|
| 1 | Age                 | 18 - 40   | years      |
| 2 | BMI                 | 18 - 32   | kg/m2      |
| 3 | Total Distance      | 4 - 15    | km/match   |
| 4 | Sprint Count        | 5 - 70    | per match  |
| 5 | Acceleration Load   | 40 - 350  | AU         |
| 6 | ACWR                | 0.4 - 2.2 | ratio      |
| 7 | Yo-Yo Score         | 13 - 24   | level      |
| 8 | Jump Height         | 20 - 60   | cm         |
| 9 | Previous Injuries   | 0 - 8     | count      |
|10 | Minutes Played      | 0 - 90    | min/match  |

## Output

- **Injury Risk**: HIGH RISK or LOW RISK
- **Probability Score**: 0 - 100 %
- **Feature Importance Chart**: Shows which inputs matter most

## Design

Built with the **Swiss International Typographic Style** — strict grid layouts,
Inter typeface, monochrome palette with Swiss Red (#FF3000) accent, zero border
radius, and visible structural borders.
