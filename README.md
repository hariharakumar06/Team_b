# 📡 Telecom Churn Intelligence Dashboard

An advanced AI-powered customer churn prediction system with a beautiful, modern frontend built with Streamlit. This application uses machine learning (Logistic Regression + PCA) to predict customer churn probability in the telecom industry.

## ✨ Features

- 🎨 **Modern Glassmorphism UI** - Beautiful, premium interface with gradient animations
- 🤖 **AI-Powered Predictions** - Machine learning model for accurate churn prediction
- 📊 **Real-time Risk Analysis** - Instant customer churn probability assessment
- 🧠 **Actionable Insights** - AI-generated recommendations for customer retention
- 📈 **Interactive Visualizations** - Dynamic progress bars and metrics
- 🎯 **User-Friendly Interface** - Intuitive input forms with helpful tooltips

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hariharakumar06/Team_b.git
cd Team_b
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to:
```
http://localhost:8501
```

## 📋 Usage

1. **Enter Customer Information:**
   - Customer Age
   - Tenure (months with the company)
   - Monthly Charges (₹)
   - Contract Type (Month-to-Month, One Year, Two Year)
   - Internet Service Type (DSL or Fiber)

2. **Click "Analyze Customer Churn Risk"** to get:
   - Churn probability percentage
   - Risk classification (High Risk / Low Risk)
   - AI-generated actionable insights
   - Retention recommendations

## 🛠️ Technology Stack

- **Frontend Framework:** Streamlit
- **Machine Learning:** scikit-learn
- **Data Processing:** pandas, numpy
- **Model Serialization:** joblib
- **Styling:** Custom CSS with glassmorphism effects

## 📦 Project Structure

```
Team_b/
├── app.py                 # Main Streamlit application
├── churn_model.pkl        # Trained Logistic Regression model
├── scaler.pkl             # StandardScaler for feature normalization
├── pca.pkl                # PCA transformer for dimensionality reduction
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # Project documentation
```

## 🎯 Model Information

The prediction model uses:
- **Algorithm:** Logistic Regression
- **Feature Engineering:** Principal Component Analysis (PCA)
- **Preprocessing:** StandardScaler normalization
- **Input Features:**
  - Age
  - Tenure
  - Monthly Charges
  - Contract Type (One-hot encoded)
  - Internet Service Type (One-hot encoded)

## 🎨 UI Features

- **Glassmorphism Design** - Frosted glass effect with blur
- **Gradient Animations** - Smooth color transitions
- **Responsive Layout** - Adapts to different screen sizes
- **Interactive Elements** - Hover effects and transitions
- **Google Fonts** - Modern typography with Inter font family
- **Dark Theme** - Easy on the eyes with vibrant accents

## 📊 Insights & Recommendations

The application provides intelligent insights based on:
- Customer tenure patterns
- Contract type analysis
- Pricing tier evaluation
- Service type correlations
- Age demographic factors
- Risk severity assessment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

- **Team B** - *Frontend Enhancement & Deployment*
- **Original Model** - Based on Logistic Assessment project

## 🙏 Acknowledgments

- Original project inspiration from [Beni-18/Logistic_Assesment](https://github.com/Beni-18/Logistic_Assesment)
- Streamlit for the amazing framework
- scikit-learn for ML capabilities

## 📞 Support

For support, email your team or open an issue in the repository.

---

<div align="center">
  <p>Built with ❤️ by Team B</p>
  <p>🤖 Powered by AI & Machine Learning</p>
</div>
