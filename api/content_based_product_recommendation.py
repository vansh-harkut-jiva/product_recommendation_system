from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import time

# Initialize FastAPI
app = FastAPI()

# Global variables
df = None
cosine_sim = None
feature_matrix = None
tfidf_title = None
tfidf_desc = None
category_encoder = None
brand_encoder = None
task_results = {}

class ProductRequest(BaseModel):
    product_title: str

# Feature Engineering Function
def preprocess_features(df):
    global tfidf_title, tfidf_desc, category_encoder, brand_encoder
    df = df.fillna("")

    # TF-IDF for Title and Description
    tfidf_title = TfidfVectorizer(stop_words="english")
    tfidf_matrix_title = tfidf_title.fit_transform(df["title"])

    tfidf_desc = TfidfVectorizer(stop_words="english")
    tfidf_matrix_desc = tfidf_desc.fit_transform(df["description"])

    # Category Encoding (One-Hot Encoding)
    category_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    category_encoded = category_encoder.fit_transform(df[["category"]])

    # Brand Encoding (Label Encoding + Weighting)
    brand_encoder = LabelEncoder()
    df["brand_encoded"] = brand_encoder.fit_transform(df["brand"]) * 0.5  # Lower weight for brand

    # Price Scaling
    price_scaler = MinMaxScaler()
    df["price_scaled"] = price_scaler.fit_transform(df[["price"]])

    # Feature Matrix
    global feature_matrix
    feature_matrix = np.hstack([
        tfidf_matrix_title.toarray(),
        tfidf_matrix_desc.toarray(),
        category_encoded,
        df[["brand_encoded", "price_scaled"]].values
    ])
    return df

@app.on_event("startup")
def startup_event():
    global df, cosine_sim
    try:
        df = pd.read_csv("content_based_dataset.csv")
        df = preprocess_features(df)
        cosine_sim = cosine_similarity(feature_matrix)
        print("Startup: Data and cosine similarity matrix initialized successfully.")
    except Exception as e:
        print(f"Error during startup: {e}")

# Get recommendations with category & brand filtering
def get_recommendations(product_title, df, cosine_sim, top_n=5):
    product_title = product_title.lower()
    df["title_lower"] = df["title"].str.lower()

    if product_title not in df["title_lower"].values:
        raise HTTPException(status_code=404, detail=f"Product '{product_title}' not found.")

    idx = df[df["title_lower"] == product_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filter by same category
    product_category = df.iloc[idx]["category"]
    filtered_scores = [s for s in sim_scores if df.iloc[s[0]]["category"] == product_category]

    # Select top N recommendations
    product_indices = [i[0] for i in filtered_scores[:top_n]]
    recommended_df = df.iloc[product_indices][["title", "category", "brand", "price", "rating"]]

    return {
        "Product Details": df.iloc[idx][["title", "category", "brand", "price", "rating"]].to_dict(),
        "Recommendations": recommended_df.to_dict(orient="records")
    }

@app.post("/recommendations/")
def get_product_recommendations(request: ProductRequest, background_tasks: BackgroundTasks):
    product_title = request.product_title
    task_id = str(uuid.uuid4())  # Generate a unique task ID
    
    # Immediately mark the task as "processing"
    task_results[task_id] = {"status": "processing"}

    # Run the recommendation process in the background
    background_tasks.add_task(run_recommendation_task, task_id, product_title)

    return {
        "message": "Request is being processed in the background.",
        "task_id": task_id
    }

# Run recommendation tasks in the background
def run_recommendation_task(task_id, product_title):
    try:
        # Simulating processing time
        time.sleep(2)  # Sleep for a few seconds to simulate processing
        recommendations = get_recommendations(product_title, df, cosine_sim)
        task_results[task_id] = {"status": "completed", "data": recommendations}
    except HTTPException as e:
        task_results[task_id] = {"status": "failed", "error": str(e)}
    except Exception as e:
        task_results[task_id] = {"status": "failed", "error": str(e)}

@app.get("/task/{task_id}/")
def get_task_results(task_id: str):
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found or still processing.")
    
    return task_results[task_id]
