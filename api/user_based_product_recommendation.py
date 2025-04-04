from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI()

# Load user-based collaborative filtering dataset
df = pd.read_csv("sample_ubcf_dataset.csv")
product_titles_df = pd.read_csv("product_titles.csv")  # Separate dataset for product details

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(subset=["user_id", "purchased_products", "browsed_products", "preferred_price_range"], inplace=True)

# Convert User_ID to string and strip spaces
df["user_id"] = df["user_id"].astype(str).str.strip()

# Function to clean and extract numeric product IDs
def extract_product_ids(value):
    return [int(i) for i in str(value).replace("[", "").replace("]", "").split(',') if i.strip().isdigit()]

# Extract numeric values from product lists
df["purchased_products"] = df["purchased_products"].apply(extract_product_ids)
df["browsed_products"] = df["browsed_products"].apply(extract_product_ids)

# Function to extract min and max values from Preferred_Price_Range
def extract_price_range(price_range):
    try:
        min_price, max_price = eval(price_range)  # Convert string tuple to actual tuple
        return pd.Series([min_price, max_price])
    except:
        return pd.Series([0, 0])

# Apply function to extract values
df[["preferred_price_min", "preferred_price_max"]] = df["preferred_price_range"].apply(extract_price_range)

# Encode categorical variables
df = pd.get_dummies(df, columns=["gender", "location"], drop_first=True)

# Create User Feature Matrix (Expanded Features)
feature_columns = ["age", "time_spent_on_site", "purchase_frequency", 
                   "average_rating_given", "review_sentiment_score", "preferred_price_min", "preferred_price_max"]

# Add newly encoded categorical columns
feature_columns.extend([col for col in df.columns if col.startswith("gender_") or col.startswith("location_")])

user_feature_matrix = df.set_index("user_id")[feature_columns].fillna(0)

# Compute User Similarity
user_similarity = cosine_similarity(user_feature_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_feature_matrix.index, columns=user_feature_matrix.index)

# Function to get product details with explanations
def get_product_details_with_explanations(user_id, product_ids, similar_users):
    product_details = []
    user_data = df[df["user_id"] == user_id].iloc[0]
    
    for product_id in product_ids:
        product_info = product_titles_df[product_titles_df["product_id"] == product_id].to_dict(orient="records")
        
        if product_info:
            product = product_info[0]
            explanation = ""

            # Check if similar users purchased the product
            similar_users_purchases = df[df["user_id"].isin(similar_users)]["purchased_products"].sum()
            if product_id in similar_users_purchases:
                explanation = "Similar users with matching preferences purchased this item."

            # Check if the product matches the user's preferred price range
            if user_data["preferred_price_min"] <= product["price"] <= user_data["preferred_price_max"]:
                explanation = "This product falls within your preferred price range."

            # Check if the user has browsed similar products
            if product_id in user_data["browsed_products"]:
                explanation = "You previously showed interest in similar products."

            # Default explanation if none of the above apply
            if not explanation:
                explanation = "Recommended based on your browsing and purchase history."

            product["explanation"] = explanation
            product_details.append(product)

    return product_details

# User-based recommendations
@app.get("/user-based-recommendations/{user_id}")
async def get_user_based_recommendations(user_id: str, top_n: int = 5):
    user_id = user_id.strip()  # Ensure input is clean
    
    if user_id not in df["user_id"].values:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in dataset.")
    
    if user_id not in user_similarity_df.index:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in similarity matrix.")
    
    # Find similar users
    similar_users = user_similarity_df[user_id].drop(user_id, errors="ignore").sort_values(ascending=False).index[:top_n]
    
    # Get products from similar users
    user_products = set(df[df["user_id"] == user_id]["purchased_products"].sum())
    recommended_products = set(df[df["user_id"].isin(similar_users)]["purchased_products"].sum()) - user_products
    
    # Get product details with explanations
    recommended_list = list(recommended_products)[:top_n]
    product_details = get_product_details_with_explanations(user_id, recommended_list, similar_users)
    
    return {"user_id": user_id, "recommended_products": product_details}
