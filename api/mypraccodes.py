from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid

# Initialize FastAPI
app = FastAPI()

# Load dataset and cosine similarity globally (to avoid reloading every time)
df = None
cosine_sim = None
tfidf = None
task_results = {}  # Dictionary to store background task results

# Define the input model for the product title
class ProductRequest(BaseModel):
    product_title: str  # We still use 'product_title' as input in the request

# Load the dataset and compute cosine similarity at the startup event
@app.on_event("startup")
def startup_event():
    global df, cosine_sim, tfidf
    # Load the dataset (assuming you have your dataset loaded here)
    df = pd.read_csv('sample_ecommerce_products.csv')  # Replace with your actual data file

    # Compute cosine similarity based on product titles (or descriptions)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['title'])  # Changed from 'product_title' to 'title'
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to generate complementary product recommendations dynamically
def get_complementary_products(product_title, df):
    keywords = product_title.lower().split()
    complementary_products = []

    for _, row in df.iterrows():
        if any(keyword in row['title'].lower() for keyword in keywords) or \
           any(keyword in row['description'].lower() for keyword in keywords):  # Use 'title' column
            complementary_products.append({
                "title": row['title'],
                "category": row['category'],
                "price": row['price'],
                "rating": row['rating']
            })

    return complementary_products

# Function to get main recommendations
def get_recommendations(product_title, df, cosine_sim, top_n=5):
    product_title = product_title.lower()

    if product_title not in df['title'].str.lower().values:  # Changed from 'product_title' to 'title'
        raise HTTPException(status_code=404, detail=f"Product '{product_title}' not found in dataset.")

    idx = df[df['title'].str.lower() == product_title].index[0]  # Changed from 'product_title' to 'title'

    # Get similarity scores for all products
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    product_indices = [i[0] for i in sim_scores]
    recommended_df = df.iloc[product_indices][['title', 'category', 'price', 'rating']]  # Changed from 'product_title' to 'title'

    input_category = df.iloc[idx]['category']
    same_category_products = recommended_df[recommended_df['category'] == input_category]
    final_recommendations = same_category_products.head(top_n)

    if final_recommendations.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations available in the same category for '{product_title}'.")

    complementary_products = get_complementary_products(product_title, df)

    # Combine the recommendations into one list of dictionaries
    all_recommendations = final_recommendations.to_dict(orient='records') + complementary_products

    # Remove duplicate products based on 'title'
    seen_titles = set()
    unique_recommendations = []
    for product in all_recommendations:
        if product['title'] not in seen_titles:
            unique_recommendations.append(product)
            seen_titles.add(product['title'])

    return {
        "Product Details": df.iloc[idx][['title', 'category', 'price', 'rating']].to_dict(),
        "Product Recommendations": unique_recommendations
    }

# Define the endpoint for product recommendation
@app.post("/recommendations/")
def get_product_recommendations(request: ProductRequest, background_tasks: BackgroundTasks):
    try:
        product_title = request.product_title
        task_id = str(uuid.uuid4())  # Unique ID for this task
        # Offload recommendation logic to background to avoid blocking the request
        background_tasks.add_task(run_recommendation_task, task_id, product_title)
        return {"message": "Your request is being processed in the background.", "task_id": task_id}
    except HTTPException as e:
        raise e

# Background task function
def run_recommendation_task(task_id, product_title):
    try:
        recommendations = get_recommendations(product_title, df, cosine_sim)
        task_results[task_id] = recommendations  # Store results with the task_id
    except Exception as e:
        task_results[task_id] = {"error": str(e)}

# Endpoint to get results for a specific task_id
@app.get("/task/{task_id}/")
def get_task_results(task_id: str):
    result = task_results.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found or still processing.")
    return result
