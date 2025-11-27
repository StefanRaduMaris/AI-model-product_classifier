

import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/product_classifier.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    text = input("Enter product title:")
    if text.lower() == "exit":
        print("Exiting...")
        break

    # Create a DataFrame from input
    user_input = pd.DataFrame([{
        "Product Title": text,
    }])

    # Predict sentiment
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 40)
