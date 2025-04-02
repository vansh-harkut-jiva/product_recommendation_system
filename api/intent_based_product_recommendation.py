from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI()

# Load intent-based dataset
df = pd.read_csv("intent_based_dataset.csv")

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(subset=["User_ID", "Search_Queries", "Clicked_Products", "Browsing_Categories"], inplace=True)

# Convert User_ID to string and strip spaces
df["User_ID"] = df["User_ID"].astype(str).str.strip()

# Function to extract product IDs
def extract_product_ids(value):
    return [i.strip() for i in str(value).split(',') if i.strip()]

df["Clicked_Products"] = df["Clicked_Products"].apply(extract_product_ids)
df["Past_Purchases"] = df["Past_Purchases"].apply(extract_product_ids)

# Encode categorical variables
df = pd.get_dummies(df, columns=["User_Type", "Customer_Lifecycle"], drop_first=True)

# Create User Feature Matrix
feature_columns = ["Time_Spent_on_Clicked_Products (sec)", "Add_to_Cart", "Wishlist", "Abandoned_Cart", 
                   "Depth_of_Search", "Session_Frequency", "Session_Recency", "Engagement_Level", 
                   "Mouse_Hesitation_Time (sec)"]
feature_columns.extend([col for col in df.columns if col.startswith("User_Type_") or col.startswith("Customer_Lifecycle_")])

# Ensure numeric conversion for feature columns
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

user_feature_matrix = df.set_index("User_ID")[feature_columns]

# Compute User Similarity
user_similarity = cosine_similarity(user_feature_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_feature_matrix.index, columns=user_feature_matrix.index)

# Function to recommend products with explanations
def recommend_products(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    
    # Find similar users
    similar_users = user_similarity_df[user_id].drop(user_id, errors="ignore").sort_values(ascending=False).index[:top_n]
    
    # Get products from similar users
    user_products = set(df[df["User_ID"] == user_id]["Clicked_Products"].sum())
    recommended_products = set(df[df["User_ID"].isin(similar_users)]["Clicked_Products"].sum()) - user_products
    
    explanations = []
    for product in list(recommended_products)[:top_n]:
        reason = "This product is recommended because: "
        
        # Find which similar users clicked on this product
        contributing_users = df[df["Clicked_Products"].apply(lambda x: product in x)]["User_ID"].tolist()
        
        # Extract feature similarities
        feature_impacts = []
        for feature in feature_columns:
            user_feature_value = df.loc[df["User_ID"] == user_id, feature].values[0]
            avg_similar_users_value = df.loc[df["User_ID"].isin(contributing_users), feature].mean()
            
            if avg_similar_users_value > user_feature_value:
                feature_impacts.append(feature.replace("_", " "))
        
        reason += ", ".join(feature_impacts) if feature_impacts else "similar users have interacted with it."
        explanations.append({"product": product, "reason": reason})
    
    return {"user_id": user_id, "recommended_products": explanations}

@app.get("/intent-based-recommendations/{user_id}")
async def get_intent_based_recommendations(user_id: str, top_n: int = 5):
    user_id = user_id.strip()
    return recommend_products(user_id, top_n)
