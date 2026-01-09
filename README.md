
# Project Title

AI and ML for Cybersecurity – Midterm Exam

Finding the Correlation

Student: Guram Lomidze
Course: AI and ML for Cybersecurity


The goal of this task is to find Pearson’s correlation coefficient between two variables 
𝑥
x and 
𝑦
y using data extracted from an online graph. The data points were collected manually by hovering over the blue dots on the graph and recording their coordinates.

|   x |   y |
| --: | --: |
|- 9.8 |- 8.5 |
|- 5.8 |- 6.7 |
| -5.0 |- 4.8 |
| -3.0 | -3.1 |
| -1.0 |- 0.9 |
| 1.0 | 0.8 |
| 3.0 | 2.9 |
| 5.0 | 4.7 |
| 7.0 | 6.5 |
| 9.0 | 8.3 |


The correlation was calculated programmatically using Python to ensure accuracy and reproducibility. code is provided in file correlation.Py

Result

The calculated Pearson correlation coefficient is:  0.9964524195013089


![Logo](https://github.com/GugaLomi/AI/blob/main/111111.png?raw=true)


1. Dataset Upload (1 point)

The dataset was uploaded to the repository. >>> https://github.com/GugaLomi/AI/blob/9ea5f2569278138cea59fda41a7f2af1e953abd3/g_lomidze25_63947.csv

2. Logistic Regression Model (6 points total)
2.1 Data Loading & Processing (2 points)
The dataset is loaded using pandas. Four numeric features are selected as predictors, while is_spam is used as the binary target variable.
provided in data_loading.py

2.2 Train/Test Split (70/30) (2 points)

70% of the data is used for training and 30% for validation. Stratification ensures balanced class representation.
code is provided in testsplit.py

2.3 Logistic Regression Model Code (1 point)



code is provided in logistic_regression.py

2.4 Model Coefficients (1 point)
| Feature           | Coefficient |
| ----------------- | ----------- |
| `words`           | 0.008       |
| `links`           | **0.842**   |
| `capital_words`   | 0.402       |
| `spam_word_count` | **0.756**   |


3. Model Validation (3 points)
3.1 Confusion Matrix & Accuracy (1 point)

code is provided in confmatrix.py 

3.2 Results (2 points)

I have got following results :

| Actual \ Predicted | Legitimate | Spam |
| ------------------ | ---------- | ---- |
| Legitimate         | 361        | 9    |
| Spam               | 19         | 361  |

Accuracy: 96.27%

4. Email Text Classification Function (3 points)

The function extracts the same features used in training and applies the trained model to classify new emails.

code is provided in third.py

5. Spam Email Example (1 point)

URGENT!!! Win FREE money now!!!
Click http://spam-offer.com to claim your prize.
LIMITED TIME OFFER!!!


This email contains:

Capitalized words

Spam keywords (free, win, urgent)

Multiple links
These characteristics strongly push the model toward the spam class


6. Legitimate Email Example (1 point)

Hi Guga,

Please find attached the meeting agenda for tomorrow.
Let me know if you have any questions.

Best regards,
Ana

No spam words, no links, minimal capitalization, and professional tone-Legitimate

7. Visualizations (4 points)
