from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialize FastAPI app
app = FastAPI()

# Load dataset
df = pd.read_csv("intent_based_dataset.csv")

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(subset=["User_ID", "Search_Queries", "Clicked_Products", "Browsed_Categories"], inplace=True)

# Convert User_ID to string
df["User_ID"] = df["User_ID"].astype(str).str.strip()

# Function to extract valid product IDs
def extract_product_ids(value):
    return [
        i.strip() for i in str(value).split(',')
        if i.strip() and re.match(r'^[a-zA-Z0-9 -]+$', i.strip()) and len(i.strip()) > 1
    ]

df["Clicked_Products"] = df["Clicked_Products"].apply(extract_product_ids)
df["Past_Purchases"] = df["Past_Purchases"].apply(extract_product_ids)
df["Add_to_Cart"] = df["Add_to_Cart"].apply(extract_product_ids)

# Encode categorical variables
df = pd.get_dummies(df, columns=["User_Type", "Customer_Lifecycle"], drop_first=True)

# Create User Feature Matrix
feature_columns = [
    "Time_Spent_on_Clicked_Products (sec)", "Wishlist", "Abandoned_Cart", 
    "Depth_of_Search", "Session_Frequency", "Session_Recency", "Engagement_Level"
]

df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Adjust Weighting
df["Add_to_Cart_Count"] = df["Add_to_Cart"].apply(len) * 3  # Carted items are critical
df["Engagement_Level"] = df["Engagement_Level"] * 2  # More engaged users get better similarity

# Create user feature matrix
user_feature_matrix = df.set_index("User_ID")[feature_columns]

# Compute User Similarity
user_similarity = cosine_similarity(user_feature_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_feature_matrix.index, columns=user_feature_matrix.index)

# Fallback recommendation function
def get_fallback_recommendations(top_n):
    top_clicked_products = df["Clicked_Products"].explode().value_counts().index[:top_n].tolist()
    return top_clicked_products

# Function to recommend products
def recommend_products(user_id, top_n=5, max_user_items=2):
    if user_id not in user_similarity_df.index:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    user_data = df[df["User_ID"] == user_id]
    if user_data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for user {user_id}.")

    clicked_products = set(sum(user_data["Clicked_Products"].tolist(), []))
    carted_products = set(sum(user_data["Add_to_Cart"].tolist(), []))
    past_purchases = set(sum(user_data["Past_Purchases"].tolist(), []))
    
    browsed_categories = user_data["Browsed_Categories"].iloc[0]
    browsed_categories = set(browsed_categories.split(", ")) if pd.notna(browsed_categories) else set()

    similar_users = user_similarity_df[user_id].drop(user_id, errors="ignore").sort_values(ascending=False).index[:20]
    
    similar_users_data = df[df["User_ID"].isin(similar_users)]
    similar_users_products = []
    
    similar_user_product_counts = {}
    for user in similar_users:
        user_row = df[df["User_ID"] == user]
        if not user_row.empty:
            for p in sum(user_row["Clicked_Products"].tolist(), []):
                if p not in past_purchases:
                    similar_user_product_counts[p] = similar_user_product_counts.get(p, 0) + 1
                    if p not in similar_users_products:
                        similar_users_products.append(p)
    
    user_interaction_products = [p for p in clicked_products.union(carted_products) if p not in past_purchases]
    
    user_interaction_products = sorted(
        user_interaction_products,
        key=lambda p: (p in carted_products, p in clicked_products),
        reverse=True
    )
    
    similar_users_products = sorted(
        similar_users_products,
        key=lambda p: similar_user_product_counts.get(p, 0),
        reverse=True
    )
    
    similar_users_products = [p for p in similar_users_products if p not in user_interaction_products]
    
    if browsed_categories:
        filtered_similar_products = []
        for product in similar_users_products:
            product_entries = df[df["Clicked_Products"].apply(lambda x: product in x)]
            product_categories = []
            for cat_string in product_entries["Browsed_Categories"]:
                if pd.notna(cat_string):
                    product_categories.extend(cat_string.split(", "))
            if any(category in browsed_categories for category in set(product_categories)):
                filtered_similar_products.append(product)
        
        if len(filtered_similar_products) > 0:
            similar_users_products = filtered_similar_products
    
    recommended_user_products = user_interaction_products[:max_user_items]
    remaining_slots = top_n - len(recommended_user_products)
    recommended_similar_products = similar_users_products[:remaining_slots]
    prioritized_products = recommended_user_products + recommended_similar_products
    
    if len(prioritized_products) < top_n:
        fallback_products = get_fallback_recommendations(top_n)
        for product in fallback_products:
            if product not in prioritized_products and product not in past_purchases:
                prioritized_products.append(product)
                if len(prioritized_products) >= top_n:
                    break
    
    prioritized_products = prioritized_products[:top_n]
    
    explanations = []
    for product in prioritized_products:
        reason = "This product is recommended because: "
        if product in carted_products:
            reason += "You added this product to your cart but did not purchase it."
        elif product in clicked_products:
            reason += "You have previously interacted with this product."
        else:
            reason += "Users with similar browsing behavior and engagement levels also interacted with it."
        
        explanations.append({"product": product, "reason": reason})

    return {"user_id": user_id, "recommended_products": explanations}

@app.get("/intent-based-recommendations/{user_id}")
async def get_intent_based_recommendations(user_id: str, top_n: int = 5, max_user_items: int = 2):
    user_id = user_id.strip()
    return recommend_products(user_id, top_n, max_user_items)