import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("Amazon.csv")  # 🔴 change file name

# -----------------------------
# 2. CLEAN DATA
# -----------------------------
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])

# keep only needed columns
df = df[['user_id', 'product_id', 'product_name', 'rating']]

# -----------------------------
# 3. CREATE USER-PRODUCT MATRIX
# -----------------------------
matrix = df.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating'
).fillna(0)

# -----------------------------
# 4. SIMILARITY
# -----------------------------
similarity = cosine_similarity(matrix)
similarity_df = pd.DataFrame(
    similarity,
    index=matrix.index,
    columns=matrix.index
)

# -----------------------------
# 5. PRODUCT NAME MAPPING
# -----------------------------
product_map = df[['product_id', 'product_name']].drop_duplicates()
product_dict = dict(zip(product_map['product_id'], product_map['product_name']))

# -----------------------------
# 6. RECOMMEND FUNCTION
# -----------------------------
def recommend(user_id, top_n=5):

    if user_id not in matrix.index:
        return ["User not found"]

    # most similar user
    similar_user = similarity_df[user_id].sort_values(ascending=False).index[1]

    user_products = matrix.loc[user_id]
    sim_products = matrix.loc[similar_user]

    rec = []

    for product in sim_products.index:
        if sim_products[product] > 0 and user_products[product] == 0:
            rec.append(product)

    rec = rec[:top_n]

    return [product_dict.get(pid, pid) for pid in rec]

# -----------------------------
# 7. USER INPUT
# -----------------------------
user_input = input("Enter User ID: ").strip()

recommendations = recommend(user_input)

print("\n🎯 Recommended Products:\n")
for i, product in enumerate(recommendations, 1):
    print(f"{i}. {product}")

# -----------------------------
# 8. VISUALIZATION (CLEAN)
# -----------------------------
top_products = (
    df.groupby('product_name')['rating']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# shorten names
top_products.index = top_products.index.str[:30]

plt.figure(figsize=(8,5))
plt.barh(top_products.index, top_products.values)

plt.title("Top Rated Products")
plt.xlabel("Rating")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.show()