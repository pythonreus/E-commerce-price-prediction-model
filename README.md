# 📈 Linear Regression Feature Selection and Evaluation

This project explores **linear regression** with **feature selection**, implementing both the **closed-form solution (normal equation)** and **gradient descent optimization** to train and evaluate models on a custom dataset.

---

## 🧠 Project Overview

The project includes:
- Linear regression using **two optimization techniques**:
  - Closed-form solution
  - Gradient descent *(coming soon)*
- Exhaustive feature selection using combinations of input features
- Model evaluation using standard metrics
- Data visualization for deeper insights
- Automatic identification of the **best-performing model**

---

## 🛠️ Features & Functionality

### 📊 Data Preprocessing
- Reads data from `dataset.csv`
- Drops irrelevant columns (`Email`, `Address`, `Avatar`)
- Converts all values to numeric
- Cleans missing or duplicate rows

### 🔁 Feature Selection
- Uses all possible combinations of 4 features:
  - `Avg. Session Length`
  - `Time on App`
  - `Time on Website`
  - `Length of Membership`

### 📈 Linear Regression Implementations
- **Closed-form solution** (normal equation)
- **Gradient descent** *(manual implementation coming soon)*:
  - Configurable learning rate and iterations
  - Tracks convergence through cost history

### ✅ Evaluation Metrics
Each model is evaluated using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score** (Coefficient of Determination)

### 📉 Visualizations
- Predicted vs Actual scatter plot
- Residuals vs Predicted plot
- Error distribution histogram

### 🏆 Best Model Insights
- Selects and displays the **best model** using highest R² score
- Lists **top 3 models** for comparison

---

## 🔄 Gradient Descent (Coming Soon)

A custom gradient descent implementation will be added to:
- Compare against the closed-form solution
- Offer more control over the learning process
- Handle large datasets more efficiently

Planned features:
- Learning rate tuning
- Cost function tracking
- Convergence visualization

---

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/linear-regression-feature-selection.git
