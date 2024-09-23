# Evaluating Linear Regression, SVR, and ANN Models for California Housing Price Prediction

## Project Overview

This project aims to predict California housing prices by exploring and comparing the performance of various machine learning techniques. I have delved into linear regression, support vector regression (SVR), and artificial neural networks (ANNs) to understand their strengths and limitations in this context. The project also investigates the impact of hyperparameter tuning on SVR, evaluating if it leads to significant performance improvements. The primary goal is to identify the best-performing model for this specific regression task.

## Motivation

This project is motivated by a desire to gain hands-on experience with different machine learning algorithms for regression analysis. I specifically chose to work with the California Housing dataset to:

* **Compare and contrast:**  Evaluate the predictive power of Linear Regression, SVR, and ANNs on a real-world dataset.
* **Hyperparameter Tuning:** Assess the effectiveness of GridSearchCV in optimizing SVR performance.
* **Gain Insights:** Understand the influence of various features on California housing prices. 

## Dataset

The project utilizes a modified version of the California Housing Prices dataset, originally sourced from StatLib.

* **Link:** [https://raw.githubusercontent.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/master/data/California%20Housing.txt](https://raw.githubusercontent.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/master/data/California%20Housing.txt)
* **Citation:** Pace, R. Kelley, and Ronald Barry. "Sparse Spatial Autoregressions." Statistics & Probability Letters 33.3 (1997): 291-297.

**Features:**

* **longitude:** Longitude value of the house location.
* **latitude:** Latitude value of the house location.
* **housingMedianAge:** Median age of houses in the block.
* **totalRooms:** Total number of rooms in the block.
* **totalBedrooms:** Total number of bedrooms in the block.
* **population:** Population of the block.
* **households:** Number of households in the block.
* **medianIncome:** Median income of households in the block (in tens of thousands of US Dollars).
* **medianHouseValue:** Median house value in the block (in US Dollars).

**Dataset Description:** [Refer to the provided dataset link for a detailed description.]

## Instructions to Run Locally

This project uses a Jupyter Notebook for analysis. 

1. **Prerequisites:** Make sure you have Python 3.10.12 installed on your system. 
2. **Clone the repository:** Open your terminal or command prompt and execute:
   ```bash
   git clone https://github.com/ratulbanik/Evaluating-Linear-Regression-SVR-and-ANN-Models-for-California-Housing-Price-Prediction.git
   ```
3. **Navigate to the project directory:** `cd Evaluating-Linear-Regression-SVR-and-ANN-Models-for-California-Housing-Price-Prediction`
4. **Create a virtual environment (recommended):** `python3 -m venv .venv`
5. **Activate the environment:**
    * Windows: `.venv\Scripts\activate`
    * macOS/Linux: `source .venv/bin/activate`
6. **Install dependencies:** `pip install -r requirements.txt`
7. **Start Jupyter Notebook:** `jupyter notebook`
8. **Open the notebook:** Click on `california_housing_analysis.ipynb` within the Jupyter interface to open and run the notebook.

## Running in Google Colab

You can run this project directly in your browser using Google Colab. Here's how:

1. **Open the notebook in Colab:** Click on the "Open in Colab" badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ratulbanik/Evaluating-Linear-Regression-SVR-and-ANN-Models-for-California-Housing-Price-Prediction/blob/main/notebooks/california_housing_analysis.ipynb)
2. **Connect to a runtime:** Colab will prompt you to connect to a runtime environment. Select a GPU or TPU runtime if your analysis requires it.
3. **Run the cells:** The notebook is now ready! Execute the cells in order to run the code.

**Note:** You might need to install some dependencies within the Colab notebook if they are not already available in the Colab environment. 

## Dependencies

* Python 3.10.12
* Libraries:
    * pandas
    * numpy
    * matplotlib 
    * seaborn
    * scikit-learn
    * keras 
    * contextily

## Methodology

1. **Data Loading and Preprocessing:**
   * Imported necessary libraries: pandas, numpy, matplotlib, seaborn, and contextily (for map creation) library.
   * Loaded the dataset and renamed columns for clarity.
   * Examined data shape, head, tail, and information using `data.info()`.
   * Checked for unique values per column (`data.nunique()`).
   * Identified and handled missing values, calculating the percentage of missing data.
   * Checked for duplicate records (`data.duplicated()`).
   * Verified for zero or negative values, finding negative values only in 'longitude', which was acceptable.
   * Performed descriptive statistical analysis (`data.describe()`).
2. **Exploratory Data Analysis (EDA):**
   * Conducted univariate analysis using histograms for individual feature distributions.
   * Created scatterplots with color coding:
     * Longitude vs. Latitude, colored by median income to visualize income distribution.
     * Longitude vs. Latitude, colored by median house value to observe house value dispersion. 
   * Performed univariate analysis using box plots to detect potential outliers.
   * Generated scatter plots with 'medianHouseValue' as the dependent variable against other features.
   * Constructed a correlation matrix to reveal relationships between variables.
3. **Data Cleaning and Preparation:**
   * **Outlier Handling:** Employed the Interquartile Range (IQR) method to remove outliers, opting for row removal due to the dataset size.
   * **Data Standardization:** Standardized features to have zero mean and unit variance.
   * **Train-Test Split:** Partitioned the data into training (80%) and testing (20%) sets using `random_state=42` for reproducibility.
4. **Model Building and Evaluation:**
   * **Linear Regression:** Implemented a linear regression model.
   * **Support Vector Regression (SVR):** Built an SVR model.
   * **Hyperparameter Tuned SVR:** Used GridSearchCV to tune SVR hyperparameters for performance optimization. 
   * **Artificial Neural Networks (ANNs):** Constructed three custom sequential ANN models using Keras:
     * **Model 1:** Mean Absolute Error (MAE) loss, ReLU activation, Adam optimizer.
     * **Model 2:** Mean Squared Error (MSE) loss, ReLU activation, Adam optimizer.
     * **Model 3:** Huber loss, ReLU activation, Adam optimizer.
     * All models were trained for 100 epochs with a batch size of 32 and a validation split of 0.2. 
5. **Model Comparison:** 
   * Evaluated model performance using metrics: R-squared (R2), Mean Squared Error (MSE), Mean Absolute Error (MAE). 
   * Visualized model performance through residual plots (predicted vs. actual values) for all six models.
   * Compared R2 scores using a bar chart.

## Conclusion

The analysis revealed that the hyperparameter-tuned SVR and the ANN model using the MSE loss function exhibited the best performance, achieving an impressive R2 score of 0.77. The other two ANN models also performed commendably with R2 scores of 0.76.  Linear regression yielded the lowest performance with an R2 of 0.62.  The project highlighted the efficacy of more complex models like SVR and ANNs in capturing non-linear relationships within the data, ultimately leading to better predictive accuracy for California housing prices.

## Author & Contact 

* **GitHub:** [ratulbanik](https://github.com/ratulbanik)
* **LinkedIn:** [https://www.linkedin.com/in/ratul-banik/](https://www.linkedin.com/in/ratul-banik/)  