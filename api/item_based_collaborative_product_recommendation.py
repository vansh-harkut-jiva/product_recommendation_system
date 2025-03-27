from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Initialize FastAPI app
app = FastAPI()

# Load dataset
df = pd.read_csv("item_based_dataset.csv")  # Change to actual dataset path

# Data Preprocessing
df.drop_duplicates(inplace=True)
df.dropna(subset=["Product_ID", "Title", "Category"], inplace=True)

# Convert necessary columns to appropriate data types
df["Product_ID"] = pd.to_numeric(df["Product_ID"], errors="coerce").dropna().astype(int)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(df["Price"].median())
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(df["Rating"].median())
df["Purchase_Frequency"] = pd.to_numeric(df["Purchase_Frequency"], errors="coerce").fillna(0)

# Feature Engineering (Text Features) - Category Given 2X Weightage
df["Combined_Text"] = (df["Category"] + " ")*2 + df["Title"] + " " + df["Description"].fillna("")

# TF-IDF Vectorization for Text Similarity
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(df["Combined_Text"])
text_similarity = cosine_similarity(tfidf_matrix)

# Normalize Numerical Features
scaler = MinMaxScaler()
df[["Price", "Rating", "Purchase_Frequency"]] = scaler.fit_transform(df[["Price", "Rating", "Purchase_Frequency"]])

# Compute Numerical Feature Similarity
num_features = df[["Price", "Rating", "Purchase_Frequency"]]
num_similarity = cosine_similarity(num_features)

# Final Hybrid Similarity Matrix (Weighted Combination)
alpha = 0.6  # Weight for text-based similarity
beta = 0.4   # Weight for numerical feature similarity
final_similarity = (alpha * text_similarity) + (beta * num_similarity)

# Convert to DataFrame for easy lookup
product_similarity_df = pd.DataFrame(final_similarity, index=df["Product_ID"], columns=df["Product_ID"])

# Function to retrieve product details
def get_product_details(product_ids):
    filtered_products = df[df["Product_ID"].isin(product_ids)]
    return filtered_products[["Product_ID", "Title", "Price", "Category", "Dimensions", "Color", "Size", "Weight"]
    ].drop_duplicates().to_dict(orient="records")

# Item-based recommendations with Product Title as input
@app.get("/recommendations/")
async def get_recommendations(product_title: str, top_n: int = 5):
    product_match = df[df["Title"].str.lower() == product_title.lower()]
    
    if product_match.empty:
        raise HTTPException(status_code=404, detail=f"Product '{product_title}' not found.")
    
    product_id = product_match.iloc[0]["Product_ID"]
    
    if product_id not in product_similarity_df.index:
        raise HTTPException(status_code=404, detail="Product similarity data not available.")
    
    # Get similar products based on hybrid item-item similarity
    similar_products = product_similarity_df[product_id].sort_values(ascending=False).index[1:top_n+1]
    product_details = get_product_details(similar_products)
    
    return {"product_title": product_title, "recommended_products": product_details}
