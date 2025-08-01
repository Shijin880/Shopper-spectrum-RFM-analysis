
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    # Load the already processed and segmented data
    df = pd.read_csv("/content/rfm_customer_segmentation (2).csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) # Ensure InvoiceDate is datetime
    return df

data = load_data()

# Prepare RFM data (already calculated in the notebook)
# This function is now simplified to just return the relevant columns and scaler/model if needed for prediction
def prepare_rfm(df):
    # Assume RFM and Cluster/Segment are already in the loaded data
    rfm_data = df[['CustomerID', 'Recency', 'Frequency', 'MonetaryValue', 'Cluster', 'Segment']]
    # Fit scaler and a dummy KMeans model for prediction purposes in the app
    scaler = StandardScaler()
    # We need to fit the scaler on the data used for clustering in the notebook
    scaler.fit(df[['Recency', 'Frequency', 'MonetaryValue']])
    n_clusters = rfm_data['Cluster'].nunique()
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # Fit the dummy model to get predict functionality, even though we have segments
    # In a real app, you'd save/load the actual fitted model and scaler
    model.fit(scaler.transform(rfm_data[['Recency', 'Frequency', 'MonetaryValue']]))

    return rfm_data, scaler, model

rfm_data, scaler, kmeans_model = prepare_rfm(data)

# Build product similarity matrix
def get_similarity_matrix(df):
    # Use the 'AllProductsPurchased' column which is a list of products per customer
    # We need to expand this list to create a customer-product matrix
    # This requires processing the list column
    # A more robust way is to use the original df before aggregation if available
    # For this simplified app, let's assume we can process the list
    # This part might need adjustment based on how 'AllProductsPurchased' is structured
    # Let's recreate a simplified pivot for demonstration
    # This is a placeholder and might not work directly with a list of lists
    # A better approach is to pass the original df or a pre-calculated similarity matrix
    # Given the current data structure, let's skip product similarity for now or use a simplified approach
    # Let's create a dummy similarity matrix or remove this feature if not feasible
    st.warning("Product recommendation feature is simplified. A full implementation requires detailed product-level data.")
    # Dummy similarity matrix for demonstration
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    sim_data = np.random.rand(len(products), len(products))
    np.fill_diagonal(sim_data, 1.0)
    sim_df = pd.DataFrame(sim_data, index=products, columns=products)
    return sim_df

# Only build similarity matrix if 'AllProductsPurchased' column exists and is suitable
similarity_matrix = None
if 'AllProductsPurchased' in data.columns:
     st.info("Attempting to build product similarity matrix...")
     try:
         # Flatten the list of lists in 'AllProductsPurchased' to get all unique products
         all_products_list = [product for sublist in data['AllProductsPurchased'].dropna() for product in eval(sublist)] # eval is risky, but needed for string representation of list
         unique_products = list(set(all_products_list))

         # Create a customer-product matrix (simplified - presence/absence)
         customer_product_matrix_data = []
         for index, row in data.iterrows():
             customer_id = row['CustomerID']
             purchased_products = eval(row['AllProductsPurchased']) if pd.notna(row['AllProductsPurchased']) else []
             row_data = {'CustomerID': customer_id}
             for product in unique_products:
                 row_data[product] = 1 if product in purchased_products else 0
             customer_product_matrix_data.append(row_data)

         customer_product_matrix = pd.DataFrame(customer_product_matrix_data).set_index('CustomerID').fillna(0)

         # Calculate cosine similarity
         if not customer_product_matrix.empty:
             similarity = cosine_similarity(customer_product_matrix.T)
             similarity_matrix = pd.DataFrame(similarity, index=customer_product_matrix.columns, columns=customer_product_matrix.columns)
             st.success("Product similarity matrix built.")
         else:
             st.warning("Customer-product matrix is empty, cannot build similarity matrix.")

     except Exception as e:
         st.error(f"Error building product similarity matrix: {e}")
         similarity_matrix = None
else:
    st.warning(" 'AllProductsPurchased' column not found in the loaded data. Skipping product similarity.")


# Streamlit UI
st.set_page_config(page_title="🛍️ Shopper Spectrum", layout="wide")
st.title("🛍️ Shopper Spectrum Web App")

tab1, tab2 = st.tabs(["🔁 Product Recommender", "🎯 Customer Segmentation"])

# Tab 1: Product Recommender
with tab1:
    st.subheader("🔎 Find Similar Products")
    if similarity_matrix is not None:
        product_list = similarity_matrix.index.tolist()
        product_input = st.selectbox("Select a Product:", product_list)
        if st.button("Get Recommendations"):
            if product_input in similarity_matrix.index:
                # Ensure recommendations are from the similarity matrix
                recommendations = similarity_matrix[product_input].sort_values(ascending=False).drop(product_input).head(5) # Exclude the product itself
                st.markdown("### 📝 Top 5 Similar Products")
                if not recommendations.empty:
                    for i, (product, score) in enumerate(recommendations.items(), 1):
                        st.markdown(f"**{i}.** {product} (Similarity: `{score:.2f}`)")
                else:
                     st.info("No similar products found.")

            else:
                st.error("❌ Selected product not found in similarity matrix.")
    else:
        st.warning("Product recommendation is not available due to data format issues.")


# Tab 2: Customer Segmentation
with tab2:
    st.subheader("🧮 Predict Customer Segment")
    st.write("Enter RFM values to predict customer segment.")
    recency = st.number_input("Recency (days)", min_value=0, value=recency_df['Recency'].mean() if 'recency_df' in locals() else 10)
    frequency = st.number_input("Frequency (no. of purchases)", min_value=0, value=frequency_df['Frequency'].mean() if 'frequency_df' in locals() else 5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=customer_monetary.mean() if 'customer_monetary' in locals() else 500.0)

    if st.button("Predict Segment"):
        try:
            user_input = np.array([[recency, frequency, monetary]])
            input_scaled = scaler.transform(user_input)
            cluster_label = kmeans_model.predict(input_scaled)[0]
            # Use the segment mapping derived in the notebook
            # This mapping should ideally be saved and loaded
            # For now, we'll use a simplified mapping or try to infer from loaded data
            # Assuming 'Segment' column exists in loaded data with correct labels
            predicted_segment = rfm_data[rfm_data['Cluster'] == cluster_label]['Segment'].mode()[0] if not rfm_data[rfm_data['Cluster'] == cluster_label]['Segment'].empty else "Unknown"

            st.success(f"📊 Predicted Segment: **{predicted_segment}**")
        except Exception as e:
            st.error(f"Error predicting segment: {e}")

    st.subheader("Overview of Customer Segments")
    if not rfm_data.empty:
        segment_counts = rfm_data['Segment'].value_counts()
        st.write("Number of customers in each segment:")
        st.bar_chart(segment_counts)

        st.write("Average RFM values per segment:")
        segment_rfm_avg = rfm_data.groupby('Segment')[['Recency', 'Frequency', 'MonetaryValue']].mean()
        st.dataframe(segment_rfm_avg)
    else:
        st.info("No segmented customer data available.")

