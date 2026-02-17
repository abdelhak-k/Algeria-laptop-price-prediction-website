# Algerian Laptop Price Predictor

A Django web application that predicts laptop prices in the Algerian market using machine learning.

**License:** [MIT](LICENSE) — feel free to use this project; we ask that you keep the copyright and license notice.

The website is working and is deployed on: https://laptopdz.leapcell.app/
## Overview

This project uses a trained Linear Regression model to predict laptop prices based on specifications like brand, series, CPU, GPU, RAM, storage, and condition. The model achieves 88.1% accuracy (R² score) with a Mean Absolute Error of 19,316 DZD.

## Features

 
## Model Information

## Usage

 
**Gradient Boosting Model Performance:**
- **R² Score**: ≈ 0.90 (90% variance explained)
- **MAPE**: Lower than 14.82%
- **MAE**: Lower than 19,316 DZD
- **RMSE**: Lower than 34,390 DZD
 
The Gradient Boosting model provides high prediction accuracy and robust performance, making it ideal for real-time web applications. Price ranges are calculated using ±RMSE to provide confidence intervals.
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

This project is developed for educational purposes, by: CHERDOUH Yassir, KADOUCI Abdelhak, GUENDOUZ Ahmed Fateh, SOUACI Abdennour.

## Requirements

See `requirements.txt` for detailed package versions.

## Notes

- Ensure model files are present in the `models/` directory
- Static files are served automatically in development mode
