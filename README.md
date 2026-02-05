# Algerian Laptop Price Predictor

A Django web application that predicts laptop prices in the Algerian market using machine learning.

## Overview

This project uses a trained Linear Regression model to predict laptop prices based on specifications like brand, series, CPU, GPU, RAM, storage, and condition. The model achieves 88.1% accuracy (R² score) with a Mean Absolute Error of 19,316 DZD.

## Features

- Interactive web form for laptop specifications input
- Dynamic dropdowns (brand→series, CPU brand→family)
- Real-time price prediction with confidence intervals
- Modern, responsive design with Algerian theme
- Model performance: R² = 0.881, MAPE = 14.82%, RMSE = 34,390 DZD

## Usage

1. Navigate to the homepage
2. Fill in laptop specifications:
   - Basic info: Brand, Series, Condition, Year
   - CPU: Brand, Family, Generation, Suffix, Professional flag
   - GPU: Model, Suffix
   - Memory: RAM size, SSD size, HDD size
3. Click "Predict Price" to get estimation
4. View predicted price with min/max range based on model uncertainty

## Model Information

The machine learning model was trained on Algerian laptop market data. While a Gradient Boosting model achieved higher performance during testing, we chose Linear Regression for production deployment due to its computational efficiency and suitability for CPU-only backend environments.

**Linear Regression Model Performance:**
- **R² Score**: 0.8810 (88.1% variance explained)
- **MAPE**: 14.82%
- **MAE**: 19,316 DZD
- **RMSE**: 34,390 DZD

The Linear Regression model provides an excellent balance between prediction accuracy and computational simplicity, making it ideal for real-time web applications without requiring GPU acceleration.

Price ranges are calculated using ±RMSE to provide confidence intervals.

## Project Structure

```
predict_price/
├── manage.py
├── predict_price/          # Django project settings
├── predictor/              # Main application
│   ├── forms.py           # Form definitions
│   ├── views.py           # View logic
│   ├── models.py          # Database models
│   ├── model_utils.py     # ML model utilities
│   └── templates/         # HTML templates
├── models/                # Trained ML models
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.pkl
└── static/               # CSS, JS, images
```

## Dataset

This project uses the Algeria Laptop Price Prediction Dataset available on Kaggle:
https://www.kaggle.com/datasets/kadouciabdelhak/algeria-laptop-price-prediction-dataset-cleaned

## License

This project is developed for educational purposes.

## Requirements

See `requirements.txt` for detailed package versions. Main dependencies:

## Notes

- Ensure model files are present in the `models/` directory
- Static files are served automatically in development mode
