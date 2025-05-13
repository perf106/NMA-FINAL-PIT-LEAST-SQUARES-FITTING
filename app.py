import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Least Squares Fitting", layout="centered")
st.title("📉 Least Squares Fitting App")

st.markdown("Select the method, enter values, and press **Fit Curve** to perform least squares fitting.")

# --- Method Selection ---
method = st.selectbox("Choose the fitting method:", ["Linear", "Polynomial"])

# --- Degree Slider (only for polynomial) ---
degree = 1
if method == "Polynomial":
    degree = st.slider("Select the degree of polynomial", min_value=1, max_value=5, value=2)

# --- Input Fields ---
x_input = st.text_area("Enter x values (comma-separated)", "1, 2, 3, 4, 5")
y_input = st.text_area("Enter y values (comma-separated)", "2, 4, 5, 4, 5")

# --- Fit Button ---
fit_button = st.button("📈 Fit Curve")

if fit_button:
    try:
        # --- Parse Inputs ---
        x = np.array([float(i.strip()) for i in x_input.split(',')])
        y = np.array([float(i.strip()) for i in y_input.split(',')])

        if len(x) != len(y):
            st.error("❗ x and y must have the same number of values.")
        else:
            # --- Perform Fitting ---
            if method == "Linear":
                # Linear regression formula
                n = len(x)
                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xy = np.sum(x * y)
                sum_x_squared = np.sum(x ** 2)

                m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
                b = (sum_y - m * sum_x) / n
                y_pred = m * x + b

                st.subheader("📈 Linear Regression Equation")
                st.latex(f"\\hat{{y}} = {round(m, 4)}x + {round(b, 4)}")

                # Curve for graph
                def predict(x_vals): return m * x_vals + b

            else:
                # Polynomial fitting
                coeffs = np.polyfit(x, y, degree)
                poly = np.poly1d(coeffs)
                y_pred = poly(x)

                st.subheader(f"📈 Polynomial Degree {degree} Equation")
                st.latex(poly)

                # Curve for graph
                def predict(x_vals): return poly(x_vals)

            # --- Results Table ---
            df = pd.DataFrame({
                "x": x,
                "Observed y": y,
                "Predicted ŷ": y_pred,
                "Error (y - ŷ)": y - y_pred
            })
            st.subheader("📊 Results Table")
            st.dataframe(df.style.format(precision=4), height=250)

            # --- Plot ---
            x_range = np.linspace(min(x), max(x), 300)
            y_range = predict(x_range)

            fig, ax = plt.subplots()
            ax.scatter(x, y, color='deepskyblue', label='Observed y')
            ax.plot(x_range, y_range, color='orange', label='Fitted Curve')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"{method} Least Squares Fitting")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
