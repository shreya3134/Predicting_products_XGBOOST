from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

app = Flask(__name__)

# -----------------------------
# LOAD ORIGINAL DATA
# -----------------------------
df = pd.read_csv("restaurant-1-orders.csv", sep=',', quotechar='"', on_bad_lines='skip')

# Convert Order Date safely
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.dropna(subset=['Order Date'])

# Ensure numeric columns are correctly typed
df['Order Number'] = pd.to_numeric(df['Order Number'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Product Price'] = pd.to_numeric(df['Product Price'], errors='coerce')
df['Total products'] = pd.to_numeric(df['Total products'], errors='coerce')

df = df.dropna(subset=['Order Number', 'Quantity', 'Product Price', 'Total products'])

# -----------------------------
# PREPROCESS DATA
# -----------------------------
df['Order_Date_Only'] = df['Order Date'].dt.date

daily_item_sales = df.groupby(['Item Name', 'Order_Date_Only']).agg(
    Quantity=('Quantity', 'sum'),
    Total_Price=('Total products', 'sum')
).reset_index()

quantity_pivot = daily_item_sales.pivot(index='Order_Date_Only', columns='Item Name', values='Quantity').fillna(0)
price_pivot = daily_item_sales.pivot(index='Order_Date_Only', columns='Item Name', values='Total_Price').fillna(0)

look_back = 7
item_names = quantity_pivot.columns.tolist()
n_items = len(item_names)

# -----------------------------
# LOAD TRAINED MODELS
# -----------------------------
with open("multi_quantity.pkl", "rb") as f:
    multi_quantity = pickle.load(f)

with open("multi_price.pkl", "rb") as f:
    multi_price = pickle.load(f)

# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    target_date = None

    if request.method == "POST":
        date_str = request.form["date"]
        target_date = pd.to_datetime(date_str).date()

        last_date = quantity_pivot.index[-1]
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.date()

        if target_date <= last_date:
            prediction = "❌ Error: Target date must be after the last date in your dataset!"
        else:
            # Prepare input data
            X_input_q = quantity_pivot.values[-look_back:].copy()
            X_input_p = price_pivot.values[-look_back:].copy()

            Xq_flat = X_input_q.reshape(1, look_back * n_items)
            Xp_flat = X_input_p.reshape(1, look_back * n_items)

            # Predict
            yq_pred = multi_quantity.predict(Xq_flat)[0]
            yp_pred = multi_price.predict(Xp_flat)[0]

            # Create prediction DataFrame
            target_pred = pd.DataFrame({
                "Item Name": item_names,
                "Predicted_Quantity": yq_pred,
                "Predicted_Total_Price": yp_pred
            }).sort_values(by="Predicted_Quantity", ascending=False)

            # ✅ Choose number of top rows based on date (even/odd)
            if target_date.day % 2 == 0:
                top_n = 20
            else:
                top_n = 30

            # Save and display accordingly
            app.config["LATEST_PREDICTION"] = target_pred
            app.config["LATEST_DATE"] = target_date
            prediction = target_pred.head(top_n).to_dict(orient="records")

    return render_template("index.html", prediction=prediction, target_date=target_date)


# -----------------------------
# SAVE PREDICTIONS TO CSV
# -----------------------------
@app.route("/save_predictions", methods=["POST"])
def save_predictions():
    if "LATEST_PREDICTION" not in app.config:
        return "No predictions to save!"

    preds = app.config["LATEST_PREDICTION"]
    target_date = app.config.get("LATEST_DATE", None)

    # Load existing file safely
    existing_df = pd.read_csv("restaurant-1-orders.csv", sep=',', quotechar='"', on_bad_lines='skip')
    existing_df['Order Number'] = pd.to_numeric(existing_df['Order Number'], errors='coerce')

    # Get next starting order number
    start_order_num = int(existing_df['Order Number'].max()) + 1

    # Build final prediction dataframe matching dataset format
    preds_final = pd.DataFrame({
        "Order Number": range(start_order_num, start_order_num + len(preds)),
        "Order Date": [f"{target_date} 18:11"] * len(preds),
        "Item Name": preds["Item Name"],
        "Quantity": np.maximum(np.round(preds["Predicted_Quantity"], 0), 1).astype(int),
        "Product Price": np.round(preds["Predicted_Total_Price"] / np.maximum(preds["Predicted_Quantity"], 1), 2),
        "Total products": np.round(preds["Predicted_Total_Price"], 2)
    })

    # Append to existing CSV
    preds_final.to_csv("restaurant-1-orders.csv", mode="a", header=False, index=False)

    return "✅ Predictions saved successfully in dataset format!"


# -----------------------------
# DOWNLOAD UPDATED CSV
# -----------------------------
@app.route("/download")
def download_csv():
    if os.path.exists("restaurant-1-orders.csv"):
        return send_file("restaurant-1-orders.csv", as_attachment=True)
    else:
        return "❌ CSV file not found."


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
