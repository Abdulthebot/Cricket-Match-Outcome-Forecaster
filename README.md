# 🏏 IPL Match Forecaster: A Data Science Deep Dive

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=for-the-badge&logo=xgboost&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

This repository contains a comprehensive data science analysis and predictive modeling project on 12 seasons of Indian Premier League (IPL) match data. The entire project is self-contained in a single Jupyter Notebook, demonstrating an end-to-end workflow from data loading and exploration to building and evaluating a machine learning model.

---

### **Key Insight: Top Performing Teams**

![Matches Won Chart](wins_chart.png)

---

## The Narrative: Decoding the Game of Cricket

In a sport as complex as cricket, outcomes are driven by a multitude of factors. This project moves beyond simple intuition by applying a data-driven approach to forecast match results. The goal was to build a model that could quantify the win probability for each team, providing a statistical lens through which to view the game.

## Key Findings from the Analysis

* **Top Performers:** Mumbai Indians stand out as the most successful team, followed closely by the Chennai Super Kings.
* **The Toss Advantage:** The team that wins the toss goes on to win the match approximately 52% of the time, indicating a slight but significant advantage.
* **Model Performance:** The final XGBoost model achieved an accuracy of **~64%** on the test set, outperforming a baseline Logistic Regression model.

## Model Feature Importance

The model identified the `toss_winner` as a highly influential factor, reinforcing the findings from the exploratory data analysis.

![Feature Importance Plot](feature_importance.png)

---

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/IPL-Match-Forecaster.git](https://github.com/YOUR_USERNAME/IPL-Match-Forecaster.git)
    cd IPL-Match-Forecaster
    ```
2.  **Run the Notebook:** Open and run the `.ipynb` file in Jupyter Notebook or Google Colab. The dataset is loaded directly from an online source within the notebook, so no manual data download is required.

## Architect's Notes

* **Self-Contained Notebook:** The project is designed to be fully reproducible in a single file. By loading data from a remote URL, it eliminates the need for manual downloads or managing data files in the repository.
* **Baseline vs. Advanced Model:** A baseline Logistic Regression model was first established to create a performance benchmark. The more complex XGBoost model was then implemented, demonstrating a rigorous, comparative approach to model selection.

### My Process & Learnings

The biggest challenge in this project was handling the inconsistencies in team names across 12 seasons of data. Standardizing them was a crucial first step and a powerful lesson in how 80% of data science is often data preparation. Seeing the feature importance plot confirm that the toss winner has a statistically significant impact was a fascinating insight that validated the model's ability to capture the nuances of the game.

### **Ethical Disclaimer**
This project is for academic and demonstrative purposes only. Sports outcomes are inherently unpredictable. The model's predictions are probabilistic and should **not** be used for any form of betting or financial decision-making.
