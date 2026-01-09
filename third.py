import pandas as pd
import re
import argparse
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Configuration
# -----------------------------
SPAM_WORDS = ['free', 'win', 'urgent', 'money', 'offer']

# -----------------------------
# Feature Extraction from Email
# -----------------------------
def extract_features(email_text):
    words = email_text.split()
    return [[
        len(words),
        len(re.findall(r'http[s]?://', email_text)),
        sum(1 for w in words if w.isupper()),
        sum(1 for w in words if w.lower() in SPAM_WORDS)
    ]]

# -----------------------------
# Visualizations (Task 7)
# -----------------------------
def show_visualizations(df, cm, model, feature_names):
    # Visualization A: Class Distribution
    plt.figure()
    df['is_spam'].value_counts().plot(kind='bar')
    plt.title('Class Distribution: Spam vs Legitimate')
    plt.xlabel('Class (0 = Legitimate, 1 = Spam)')
    plt.ylabel('Number of Emails')
    plt.show()

    # Visualization B: Confusion Matrix Heatmap
    plt.figure()
    plt.imshow(cm)
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.colorbar()

    plt.xticks([0, 1], ['Legitimate', 'Spam'])
    plt.yticks([0, 1], ['Legitimate', 'Spam'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha='center', va='center')

    plt.show()

    # Visualization C: Feature Importance
    plt.figure()
    plt.bar(feature_names, model.coef_[0])
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.xlabel('Feature')
    plt.ylabel('Coefficient Value')
    plt.show()

# -----------------------------
# Main Program
# -----------------------------
def main(csv_path, visualize):
    print("[+] Loading dataset...")
    df = pd.read_csv(csv_path)

    X = df[['words', 'links', 'capital_words', 'spam_word_count']]
    y = df['is_spam']

    print("[+] Splitting data (70% train / 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("[+] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("[+] Evaluating model...")
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")

    print("\n=== Model Coefficients ===")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")

    if visualize:
        show_visualizations(df, cm, model, X.columns)

    # -----------------------------
    # Interactive Email Classification
    # -----------------------------
    print("\n=== Email Classification Mode ===")
    print("Type an email text to classify it.")
    print("Type 'exit' to quit.\n")

    while True:
        email_text = input("Email> ")
        if email_text.lower() == "exit":
            print("Exiting program.")
            break

        features = extract_features(email_text)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            print(f"[SPAM] Probability: {probability:.2f}\n")
        else:
            print(f"[LEGITIMATE] Probability: {1 - probability:.2f}\n")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email Spam Classifier")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV file (e.g. g_lomidze25_63947.csv)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show dataset and model visualizations"
    )

    args = parser.parse_args()
    main(args.data, args.visualize)
