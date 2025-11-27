

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer

#import our csv
url='https://raw.githubusercontent.com/StefanRaduMaris/AI-model-product_classifier/refs/heads/main/data/products.csv'
df=pd.read_csv(url)

#processing data
df = df.rename(columns={
"product ID":"Product ID",
"Product Title":"Product Title",
"Merchant ID":"Merchant ID",
" Category Label":"Category Label",
"_Product Code":"Product Code",
"Number_of_Views":"Number of Views",
"Mechant Rating":"Merchant Rating",
"Listing Date":"Listing Date",
})
df.to_csv("products.csv", index=False)

df = df.replace("Freezers", "Fridge Freezers").replace('fridge', 'Fridge Freezers').replace('Fridges',"Fridge Freezers")
df=df.replace('Mobile Phone', 'Mobile Phones').replace('CPU', 'CPUs')

df=df.dropna()

# Features and label
X = df[['Product Title']] 
y = df["Category Label"]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "Product Title"),
    ]
)

# Define pipeline with the best model (e.g. RandomForestClassifier)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the model to a file
joblib.dump(pipeline, "model/product_classifier.pkl")

print("Model trained and saved as 'model/product_classifier.pkl'")

