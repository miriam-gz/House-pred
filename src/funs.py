import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

def info_df(df, types=False):
    print(f"DataFrame Information:")
    print(f"- # Rows: {df.shape[0]}")
    print(f"- # Columns: {df.shape[1]}")
    
    feat = {"Categorical features": len(df.select_dtypes(include=["object", "category"]).columns), 
            "Numerical features": len(df.select_dtypes(include=["number"]).columns), 
            "Datetime features": len(df.select_dtypes(include=["datetime"]).columns),
            "Boolean features": len(df.select_dtypes(include=["bool"]).columns)}
    
    if types:
        for name, num in feat.items():
            if num > 0:
                print(f"- {name}: {num}")


def find_duplicates(df):
    duplicate_rows = df[df.duplicated()]

    if not duplicate_rows.empty:
        return duplicate_rows.index.tolist()
    else:
        print("No duplicate rows found.")
    

def preprocessing(df):
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df.drop(columns=["id"], inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    df["year_sold"] = df["date"].dt.year
    df["month_sold"] = df["date"].dt.month
    df.drop(columns=["date"], inplace=True)
    df["house_age"] = df["year_sold"] - df["yr_built"]
    df["time_since_renovation"] = np.where(df["yr_renovated"] != 0, df["year_sold"] - df["yr_renovated"], 0)
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    df["total_sqft"] = df["sqft_above"] + df["sqft_basement"]

    return df


def train_evaluate_models(models, X_train, X_test, y_train, y_test, save_path):
    results = {}
    n_models = len(models)
    n_cols = 2
    n_rows = np.ceil(n_models / n_cols).astype(int)  
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes = axes.flatten()
    
    for idx, (model_name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        
        model_filename = os.path.join(save_path, "models", model_name.replace(" ", "_").lower() + "_model.pkl")
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        y_pred = model.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {"RMSE": rmse, "R2": r2}
        
        plot_real_vs_predicted(axes[idx], model_name, y_test, y_pred)
    
    for ax in axes[n_models:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "output", "real_vs_predicted_plots.png"))
    plt.show()

    return results


def plot_real_vs_predicted(ax, model_name, y_test, y_pred):
    ax.scatter(y_test, y_pred, color="blue", alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    ax.set_title(f"{model_name} - Real vs Predicted")
    ax.set_xlabel("Real values")
    ax.set_ylabel("Predicted values")