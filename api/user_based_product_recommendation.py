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
df.dropna(subset=["User_ID", "Purchased_Products", "Browsed_Products", "Preferred_Price_Range"], inplace=True)

# Convert User_ID to string and strip spaces
df["User_ID"] = df["User_ID"].astype(str).str.strip()

# Function to clean and extract numeric product IDs
def extract_product_ids(value):
    return [int(i) for i in str(value).replace("[", "").replace("]", "").split(',') if i.strip().isdigit()]

# Extract numeric values from product lists
df["Purchased_Products"] = df["Purchased_Products"].apply(extract_product_ids)
df["Browsed_Products"] = df["Browsed_Products"].apply(extract_product_ids)

# Function to extract min and max values from Preferred_Price_Range
def extract_price_range(price_range):
    try:
        min_price, max_price = eval(price_range)  # Convert string tuple to actual tuple
        return pd.Series([min_price, max_price])
    except:
        return pd.Series([0, 0])

# Apply function to extract values
df[["Preferred_Price_Min", "Preferred_Price_Max"]] = df["Preferred_Price_Range"].apply(extract_price_range)

# Encode categorical variables
df = pd.get_dummies(df, columns=["Gender", "Location"], drop_first=True)

# Create User Feature Matrix (Expanded Features)
feature_columns = ["Age", "Time_Spent_On_Site", "Purchase_Frequency", 
                   "Average_Rating_Given", "Review_Sentiment_Score", "Preferred_Price_Min", "Preferred_Price_Max"]

# Add newly encoded categorical columns
feature_columns.extend([col for col in df.columns if col.startswith("Gender_") or col.startswith("Location_")])

user_feature_matrix = df.set_index("User_ID")[feature_columns].fillna(0)

# Compute User Similarity
user_similarity = cosine_similarity(user_feature_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_feature_matrix.index, columns=user_feature_matrix.index)

# Function to get product details with explanations
def get_product_details_with_explanations(user_id, product_ids, similar_users):
    product_details = []
    user_data = df[df["User_ID"] == user_id].iloc[0]
    
    for product_id in product_ids:
        product_info = product_titles_df[product_titles_df["Product_ID"] == product_id].to_dict(orient="records")
        
        if product_info:
            product = product_info[0]
            explanation = ""

            # Check if similar users purchased the product
            similar_users_purchases = df[df["User_ID"].isin(similar_users)]["Purchased_Products"].sum()
            if product_id in similar_users_purchases:
                explanation = "Similar users with matching preferences purchased this item."

            # Check if the product matches the user's preferred price range
            if user_data["Preferred_Price_Min"] <= product["Price"] <= user_data["Preferred_Price_Max"]:
                explanation = "This product falls within your preferred price range."

            # Check if the user has browsed similar products
            if product_id in user_data["Browsed_Products"]:
                explanation = "You previously showed interest in similar products."

            # Default explanation if none of the above apply
            if not explanation:
                explanation = "Recommended based on your browsing and purchase history."

            product["Explanation"] = explanation
            product_details.append(product)

    return product_details

# User-based recommendations
@app.get("/user-based-recommendations/{user_id}")
async def get_user_based_recommendations(user_id: str, top_n: int = 5):
    user_id = user_id.strip()  # Ensure input is clean
    
    if user_id not in df["User_ID"].values:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in dataset.")
    
    if user_id not in user_similarity_df.index:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in similarity matrix.")
    
    # Find similar users
    similar_users = user_similarity_df[user_id].drop(user_id, errors="ignore").sort_values(ascending=False).index[:top_n]
    
    # Get products from similar users
    user_products = set(df[df["User_ID"] == user_id]["Purchased_Products"].sum())
    recommended_products = set(df[df["User_ID"].isin(similar_users)]["Purchased_Products"].sum()) - user_products
    
    # Get product details with explanations
    recommended_list = list(recommended_products)[:top_n]
    product_details = get_product_details_with_explanations(user_id, recommended_list, similar_users)
    
    return {"user_id": user_id, "recommended_products": product_details}
