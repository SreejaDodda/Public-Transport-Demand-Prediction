import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
import matplotlib.pyplot as plt
from pathlib import Path
import folium
from streamlit_folium import st_folium

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Public Transport AI - Hyderabad",
    page_icon="🚌",
    layout="wide"
)

st.title("🚌 Smart Public Transport System (Hyderabad)")
st.subheader(
    "AI-Based Demand Prediction, Route Optimisation, and GIS Mapping"
)

# ==========================================
# 2. DATASET LOADING & GENERATION
# ==========================================

@st.cache_data
def load_hyderabad_stops():
    bus_stops_path = (
        Path(__file__).resolve().parent
        / 
          "dataset"
        / "hyderabad_bus_stops.csv"
    )
    if bus_stops_path.exists():
        df_stops = pd.read_csv(bus_stops_path)
        df_stops.columns = df_stops.columns.str.strip().str.lower()
        return df_stops
    return pd.DataFrame()

@st.cache_data
def load_gtfs_data():

    folder = Path(__file__).resolve().parent / "dataset"

    routes = pd.read_csv(folder / "routes.txt", dtype=str)
    trips = pd.read_csv(folder / "trips.txt", dtype=str)
    stops = pd.read_csv(folder / "stops.txt", dtype=str)
    stop_times = pd.read_csv(folder / "stop_times.txt", dtype=str)

    for data in [routes, trips, stops, stop_times]:
        data.columns = data.columns.str.strip().str.lower()

    return routes, trips, stops, stop_times


@st.cache_data
def generate_data():
    np.random.seed(42)
    rows = []
    routes = ["R1", "R2", "R3", "R4"]
    stops = ["S1", "S2", "S3"]

    for day in range(60):
        for route in routes:
            for stop in stops:
                for hour in range(6, 22, 2):
                    traffic = np.random.choice(["Low", "Medium", "High"])
                    temperature = np.random.randint(22, 39)
                    rainfall = np.random.choice([0, 0, 0, 2, 5, 8])
                    peak = 35 if hour in [8, 10, 16, 18, 20] else 5
                    
                    route_demand = {"R1": 25, "R2": 15, "R3": 10, "R4": 20}[route]
                    stop_demand = {"S1": 15, "S2": 5, "S3": 0}[stop]
                    traffic_demand = {"Low": 0, "Medium": 8, "High": 15}[traffic]

                    passengers = (
                        20 + peak + route_demand
                        + stop_demand + traffic_demand
                        - rainfall * 0.5
                        + np.random.normal(0, 5)
                    )

                    rows.append({
                        "route_id": route,
                        "stop_id": stop,
                        "hour": hour,
                        "temperature": temperature,
                        "rainfall": rainfall,
                        "traffic_level": traffic,
                        "passenger_count": max(5, round(passengers))
                    })
    return pd.DataFrame(rows)


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

def clean_data(data):
    df = data.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()

    required = ["route_id", "stop_id", "passenger_count"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    if "hour" not in df.columns:
        if "time" not in df.columns:
            raise ValueError("Dataset must contain either hour or time.")
        df["hour"] = pd.to_datetime(
            df["time"].astype(str), format="mixed", errors="coerce"
        ).dt.hour

    for col, default in [
        ("temperature", 30),
        ("rainfall", 0),
        ("traffic_level", "Medium")
    ]:
        if col not in df.columns:
            df[col] = default

    df["passenger_count"] = pd.to_numeric(
        df["passenger_count"], errors="coerce"
    )

    for col in ["hour", "temperature", "rainfall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["route_id", "stop_id", "hour", "passenger_count"]
    )
    df = df.drop_duplicates()

    df["route_id"] = df["route_id"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)

    df["traffic_level"] = (
        df["traffic_level"].fillna("Medium").astype(str).str.strip().str.title()
    )

    df["temperature"] = df["temperature"].fillna(30)
    df["rainfall"] = df["rainfall"].fillna(0)

    df = df[
        df["hour"].between(0, 23) &
        (df["passenger_count"] >= 0)
    ]

    if len(df) < 20:
        raise ValueError("At least 20 valid records are required.")

    # Identify peak hours
    df["is_peak_hour"] = (
        df["hour"].isin([8, 10, 16, 18, 20])
    ).astype(int)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce", dayfirst=True
        )

        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

    else:
        df["day_of_week"] = 0
        df["is_weekend"] = 0
    return df.reset_index(drop=True)


def create_pdf_report(route_results, algorithm, hour, traffic, capacity, fleet):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()
    elements = []

    # Project Title
    elements.append(Paragraph(
        "SMART PUBLIC TRANSPORT SYSTEM - HYDERABAD",
        styles["Title"]
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        "AI-Based Demand Prediction and Route Optimisation Report",
        styles["Heading2"]
    ))
    elements.append(Spacer(1, 15))

    # Input Details
    elements.append(Paragraph(
        f"<b>Selected ML Algorithm:</b> {algorithm}",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        f"<b>Selected Time:</b> {hour}:00",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        f"<b>Traffic Level:</b> {traffic}",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        f"<b>Bus Capacity:</b> {capacity}",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        f"<b>Available Fleet:</b> {fleet}",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 20))

    # Calculate Summary
    total_passengers = int(
        route_results["Predicted Passengers"].sum()
    )

    total_needed = int(
        route_results["Buses Needed"].sum()
    )

    total_allocated = int(
        route_results["Buses Allocated"].sum()
    )

    total_extra = int(
        route_results["Additional Buses Needed"].sum()
    )

    # Summary Section
    elements.append(Paragraph(
        "BUS ALLOCATION SUMMARY",
        styles["Heading2"]
    ))
    elements.append(Spacer(1, 10))

    summary_data = [
        ["Description", "Result"],
        ["Total Predicted Passengers", str(total_passengers)],
        ["Total Buses Needed", str(total_needed)],
        ["Buses Allocated", str(total_allocated)],
        ["Additional Buses Required", str(total_extra)]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[260, 150]
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Route-wise Details
    elements.append(Paragraph(
        "ROUTE-WISE BUS ALLOCATION",
        styles["Heading2"]
    ))
    elements.append(Spacer(1, 10))

    table_data = [[
        "Rank", "Route", "Passengers",
        "Needed", "Allocated", "Extra", "Status"
    ]]

    for _, row in route_results.iterrows():
        table_data.append([
            str(row["Priority Rank"]),
            str(row["Route"]),
            str(row["Predicted Passengers"]),
            str(row["Buses Needed"]),
            str(row["Buses Allocated"]),
            str(row["Additional Buses Needed"]),
            str(row["Allocation Status"])
        ])

    report_table = Table(
        table_data,
        colWidths=[35, 45, 65, 50, 55, 40, 100],
        repeatRows=1
    )

    report_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    elements.append(report_table)
    elements.append(Spacer(1, 20))

    # Final Recommendation
    elements.append(Paragraph(
        "FINAL ALLOCATION STATUS",
        styles["Heading2"]
    ))

    if total_extra > 0:
        status = (
            f"{total_extra} additional buses are required "
            "to meet the predicted demand."
        )
    else:
        status = (
            "Available buses are sufficient for the "
            "predicted demand."
        )

    elements.append(Paragraph(status, styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(
        "Generated by AI-Based Public Transport Demand "
        "Prediction and Route Optimisation System.",
        styles["Italic"]
    ))

    document.build(elements)

    buffer.seek(0)
    return buffer.getvalue()

# ==========================================
# 4. SIDEBAR & DATA LOADING
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Dataset Analysis",
        "Model Comparison",
        "Demand Prediction",
        "Route Optimisation",
        "Hyderabad Bus Map",
        "Individual Route Map",
        "Real Transport Data"
    ]
)

st.sidebar.divider()

data_source = st.sidebar.radio(
    "Choose Training Dataset",
    ["Sample Dataset", "Upload CSV"]
)

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Transport CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to continue, or select Sample Dataset.")
        st.stop()
    try:
        raw_data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    raw_data = generate_data()

try:
    df = clean_data(raw_data)
except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()

features = [
    "route_id",
    "stop_id",
    "hour",
    "temperature",
    "rainfall",
    "traffic_level",
    "is_peak_hour",
    "day_of_week",
    "is_weekend"
]

target = "passenger_count"

categorical = [
    "route_id",
    "stop_id",
    "traffic_level"
]

numerical = [
    "hour",
    "temperature",
    "rainfall",
    "is_peak_hour",
    "day_of_week",
    "is_weekend"
]

# ==========================================
# 5. TRAIN ML ALGORITHMS
# ==========================================

if "date" in df.columns:
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce", dayfirst=True
    )

    df = df.dropna(subset=["date"])
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)

    X = df[features]
    y = df[target]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

else:
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
algorithms = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
}

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", "passthrough", numerical)
])

trained_models = {}
results = []

for name, algorithm in algorithms.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", algorithm)])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)
    comparison_df = pd.DataFrame({
    "Actual Passengers": y_test.values,
    "Predicted Passengers": predicted
})

    trained_models[name] = pipeline
    results.append({
        "Algorithm": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 3)
    })

results_df = pd.DataFrame(results)
best_algorithm = results_df.sort_values("RMSE").iloc[0]["Algorithm"]

selected_algorithm = st.sidebar.selectbox(
    "Select ML Algorithm",
    list(algorithms.keys()),
    index=list(algorithms.keys()).index(best_algorithm)
)

model = trained_models[selected_algorithm]
st.subheader("Actual vs Predicted Passenger Demand")

st.line_chart(comparison_df)


# ==========================================
# 6. DASHBOARD
# ==========================================

if page == "Dashboard":
    st.header("📊 Project Dashboard")
    st.info("Welcome to the Smart Public Transport System for Hyderabad. Powered by Machine Learning and GIS data.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", len(df))
    c2.metric("Total Routes", df["route_id"].nunique())
    c3.metric("Average Passengers", round(df["passenger_count"].mean(), 1))
    c4.metric("ML Algorithms", len(algorithms))

    st.divider()

    st.subheader("🚍 Transport Overview")

    m1, m2 = st.columns(2)

    m1.metric(
        "Total Passenger Count",
        int(df["passenger_count"].sum())
    )

    m2.metric(
        "Average Demand per Route",
        round(df.groupby("route_id")["passenger_count"].mean().mean(), 1)
    )
    st.divider()
    st.subheader("Average Passenger Demand by Route")
    route_average = df.groupby("route_id")["passenger_count"].mean()
    st.bar_chart(route_average)

    st.subheader("Passenger Demand by Time")
    hourly = df.groupby("hour")["passenger_count"].mean().sort_index()
    st.line_chart(hourly)
    st.subheader("Route-Wise Passenger Demand Analysis")

    route_demand = (
    df.groupby("route_id")["passenger_count"]
    .mean()
    .reset_index()
    )

    route_demand.columns = ["Bus Route", "Average Passengers"]

    st.bar_chart(route_demand.set_index("Bus Route"))

    st.dataframe(route_demand, use_container_width=True)
    st.divider()
    st.subheader("🚍 Bus Allocation Overview")

    if "route_results" in st.session_state:

        saved_results = st.session_state["route_results"]

        total_needed = int(saved_results["Buses Needed"].sum())
        total_allocated = int(saved_results["Buses Allocated"].sum())
        total_extra = int(saved_results["Additional Buses Needed"].sum())

        b1, b2, b3 = st.columns(3)

        b1.metric("Total Buses Needed", total_needed)
        b2.metric("Buses Allocated", total_allocated)
        b3.metric("Additional Buses Required", total_extra)

        if total_extra > 0:
            st.warning(f"{total_extra} additional buses are required.")
        else:
            st.success("Available buses are sufficient.")

    else:
        st.info("Please run Route Optimisation first to view bus allocation results.")

    st.subheader("Model Performance")
    st.dataframe(results_df, use_container_width=True)

    # Feature Importance
    st.divider()
    st.header("📌 Model Feature Analysis")
    selected_estimator = model.named_steps["model"]
    if hasattr(selected_estimator, "feature_importances_"):
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importance = selected_estimator.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(12)
        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.info("Feature importance chart is available for Decision Tree and Random Forest models.")


# ==========================================
# 7. DATASET ANALYSIS
# ==========================================

elif page == "Dataset Analysis":
    st.header("📁 Dataset Analysis")
    st.write("Dataset shape:", df.shape)
    st.write("Dataset preview:")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Download Clean Dataset")
    st.download_button(
        "Download Clean CSV",
        df.to_csv(index=False),
        "clean_transport_data.csv",
        "text/csv"
    )


# ==========================================
# 8. MODEL COMPARISON
# ==========================================

elif page == "Model Comparison":
    st.header("🤖 Machine Learning Comparison")
    st.dataframe(results_df.sort_values("RMSE"), use_container_width=True)

    st.subheader("RMSE Comparison")
    st.bar_chart(results_df.set_index("Algorithm")["RMSE"])
    st.success(f"Best model by lowest RMSE: {best_algorithm}")


# ==========================================
# 9. PASSENGER DEMAND PREDICTION
# ==========================================

elif page == "Demand Prediction":
    st.header("🔍 Passenger Demand Prediction")
    col1, col2 = st.columns(2)
    routes = sorted(df["route_id"].unique())

    with col1:
        route = st.selectbox("Select Route", routes)
        stops = sorted(df[df["route_id"] == route]["stop_id"].unique())
        stop = st.selectbox("Select Bus Stop", stops)
        hour = st.slider("Select Time (Hour)", 0, 23, 8)

    with col2:
        temperature = st.slider("Temperature (°C)", 0, 50, 30)
        rainfall = st.slider("Rainfall (mm)", 0, 50, 0)
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])

    bus_capacity = st.number_input("Passenger Capacity per Bus", min_value=1, max_value=200, value=50)

 
if st.button("Predict Passenger Demand"):

        # 1. Base input dictionary
        input_dict = {
            "route_id": route,
            "stop_id": stop,
            "hour": hour,
            "temperature": temperature,
            "rainfall": rainfall,
            "traffic_level": traffic
        }

        # 2. Automatically supply any extra columns the trained model expects (like day_of_week, is_weekened, etc.)
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in input_dict:
                    # Provide sensible default values for engineered features
                    if col in ["day_of_week", "is_weekend", "is_peak_hour"]:
                        input_dict[col] = 0
                    else:
                        input_dict[col] = 0

        input_data = pd.DataFrame([input_dict])

        # 3. Predict using the trained model pipeline
        prediction = max(
            0,
            round(model.predict(input_data)[0])
        )

        required_buses = max(
            1,
            math.ceil(prediction / bus_capacity)
        )

        st.divider()

        st.subheader("Prediction Result")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Expected Passengers",
            prediction
        )

        c2.metric(
            "Required Buses",
            required_buses
        )

        c3.metric(
            "Selected Algorithm",
            selected_algorithm
        )

        if prediction >= 80:
            st.warning("High Passenger Demand")
        elif prediction >= 40:
            st.info("Medium Passenger Demand")
        else:
            st.success("Low Passenger Demand")

        st.write(
            f"Recommendation: Plan approximately "
            f"{required_buses} buses for this demand."
        )

        st.caption(
            "Bus allocation is an estimate based on "
            "the entered capacity assumption."
        )
   


# ==========================================
# 10. ROUTE OPTIMISATION
# ==========================================

elif page == "Route Optimisation":
    st.header("🗺️ Smart Route Optimisation")
    capacity = st.number_input("Bus Capacity", 1, 200, 50)
    fleet = st.number_input("Total Available Buses", 1, 500, 10)
    hour = st.slider("Select Time", 0, 23, 8)
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])

    if st.button("Optimise Bus Allocation"):
        route_records = []
        for r in sorted(df["route_id"].unique()):
            route_stops = df[df["route_id"] == r]["stop_id"]
            reference = route_stops.mode().iloc[0]
            route_input = pd.DataFrame([{
                "route_id": r,
                "stop_id": reference,
                "hour": hour,
                "temperature": 30,
                "rainfall": 0,
                "traffic_level": traffic,
                "is_peak_hour": int(hour in [8, 10, 16, 18, 20]),
                "day_of_week": 0,
                "is_weekend": 0
            }])
            demand = max(0, round(model.predict(route_input)[0]))
            buses_needed = max(1, math.ceil(demand / capacity))
            route_records.append({
                "Route": r, "Reference Stop": reference,
                "Predicted Passengers": demand, "Buses Needed": buses_needed
            })

        route_results = pd.DataFrame(route_records).sort_values("Predicted Passengers", ascending=False).reset_index(drop=True)
        route_results.insert(0, "Priority Rank", range(1, len(route_results) + 1))

        remaining = int(fleet)
        allocations = []
        for _, row in route_results.iterrows():
            assigned = min(int(row["Buses Needed"]), remaining)
            remaining -= assigned
            allocations.append(assigned)

        route_results["Buses Allocated"] = allocations
        route_results["Additional Buses Needed"] = (
            route_results["Buses Needed"] -
            route_results["Buses Allocated"]
        ).clip(lower=0)
        route_results["Allocation Status"] = np.where(
            route_results["Buses Allocated"] >= route_results["Buses Needed"],
            "Fully allocated", "Additional buses needed"
        )
        st.session_state["route_results"] = route_results.copy()

        st.dataframe(route_results, use_container_width=True)
        st.subheader("Bus Allocation Summary")

        total_needed = int(route_results["Buses Needed"].sum())
        total_allocated = int(route_results["Buses Allocated"].sum())
        total_extra = int(route_results["Additional Buses Needed"].sum())

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Buses Needed", total_needed)
        c2.metric("Buses Allocated", total_allocated)
        c3.metric("Additional Buses Required", total_extra)

        if total_extra > 0:
            st.warning(
                f"{total_extra} additional buses are required."
            )
        else:
            st.success("Available buses are sufficient for predicted demand.")

        pdf_data = create_pdf_report(route_results, selected_algorithm, hour, traffic, capacity, fleet)
        st.download_button("Download PDF Report", data=pdf_data, file_name="route_optimisation_report.pdf", mime="application/pdf")


# ==========================================
# 11. HYDERABAD BUS MAP (ALL STOPS)
# ==========================================

elif page == "Hyderabad Bus Map":
    st.header("🗺️ Hyderabad Real Bus Stop Map")
    bus_stops = load_hyderabad_stops()

    if not bus_stops.empty and {"stop_name", "stop_lat", "stop_lon"}.issubset(bus_stops.columns):
        bus_stops["stop_lat"] = pd.to_numeric(bus_stops["stop_lat"], errors="coerce")
        bus_stops["stop_lon"] = pd.to_numeric(bus_stops["stop_lon"], errors="coerce")
        bus_stops = bus_stops.dropna(subset=["stop_lat", "stop_lon"])

        st.success(f"Loaded {len(bus_stops)} Hyderabad bus stops from dataset.")
        st.map(bus_stops, latitude="stop_lat", longitude="stop_lon")

        st.subheader("Bus Stop Details")
        st.dataframe(bus_stops.head(100), use_container_width=True)
    else:
        st.error("Required columns ('stop_name', 'stop_lat', 'stop_lon') not found in 'hyderabad_bus_stops.csv'.")


# ==========================================
# 12. INDIVIDUAL ROUTE MAP (CONNECTED PATHS)
# ==========================================

elif page == "Individual Route Map":

    st.header("🛣️ Individual Bus Route Mapping (Hyderabad)")
    st.write("Visualize connecting bus stops and route paths across Hyderabad.")

    bus_stops = load_hyderabad_stops()

    if not bus_stops.empty:
        if "zone_id" in bus_stops.columns:
            
            zone_counts = bus_stops["zone_id"].value_counts()
            valid_zones = zone_counts[zone_counts >= 2].index.tolist()
            
            if not valid_zones:
                valid_zones = bus_stops["zone_id"].dropna().unique().tolist()

            selected_zone = st.selectbox(
                "Select Hyderabad Route / Zone (with multiple stops)",
                sorted(valid_zones)
            )

            zone_df = bus_stops[bus_stops["zone_id"] == selected_zone].copy()

            if not zone_df.empty:
                st.success(f"Displaying {len(zone_df)} stops in route group: {selected_zone}")

                center_lat = zone_df["stop_lat"].mean()
                center_lon = zone_df["stop_lon"].mean()

                route_map = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=14,
                    tiles="OpenStreetMap"
                )

                coordinates = list(zip(zone_df["stop_lat"], zone_df["stop_lon"]))

                if len(coordinates) > 1:
                    folium.PolyLine(
                        coordinates,
                        color="blue",
                        weight=5,
                        opacity=0.8,
                        tooltip=f"Route Path: {selected_zone}"
                    ).add_to(route_map)

                for _, row in zone_df.iterrows():
                    folium.Marker(
                        location=[row["stop_lat"], row["stop_lon"]],
                        popup=f"<b>Stop:</b> {row['stop_name']}<br><b>Zone:</b> {row['zone_id']}",
                        icon=folium.Icon(color="red", icon="bus", prefix="fa")
                    ).add_to(route_map)

                # ==========================================
                # HIDE OPENSTREETMAP WATERMARK / ATTRIBUTION
                # ==========================================
                st.markdown(
                    """
                    <style>
                    .leaflet-control-attribution {
                        display: none !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Render the map
                st_folium(route_map, width=800, height=500)

                st.subheader("Stop Details for Selected Route")
                st.dataframe(
                    zone_df[["stop_id", "stop_name", "zone_id", "stop_lat", "stop_lon"]],
                    use_container_width=True
                )
            else:
                st.warning("No stops found in this zone.")
        else:
            st.error("Column 'zone_id' is missing from hyderabad_bus_stops.csv.")
    else:
        st.error("Bus-stop CSV file not found inside the 'dataset' folder.")

# ==========================================
# . REAL TRANSPORT DATA INTEGRATION
# ==========================================

elif page == "Real Transport Data":

    st.header("🚌 Real Hyderabad Transport Data")
    st.write("Explore actual bus routes, trips, stops and schedules using GTFS data.")

    try:
        routes_gtfs, trips_gtfs, stops_gtfs, times_gtfs = load_gtfs_data()

        # Dataset overview
        st.subheader("Transport Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Routes", len(routes_gtfs))
        c2.metric("Total Trips", len(trips_gtfs))
        c3.metric("Total Stops", len(stops_gtfs))
        c4.metric("Stop Time Records", len(times_gtfs))

        # Route selection
        st.divider()
        st.subheader("Select Hyderabad Bus Route")

        route_ids = routes_gtfs["route_id"].dropna().unique().tolist()

        route_labels = {}

        for _, row in routes_gtfs.iterrows():
            rid = row["route_id"]

            short_name = str(row.get("route_short_name", "")).strip()
            long_name = str(row.get("route_long_name", "")).strip()

            parts = [
                x for x in [short_name, long_name]
                if x and x.lower() != "nan"
            ]

            route_labels[rid] = (
                " - ".join(parts) + f" ({rid})"
                if parts else rid
            )

        selected_route = st.selectbox(
            "Choose Bus Route",
            route_ids,
            format_func=lambda x: route_labels.get(x, x)
        )

        # Find trips
        selected_trips = trips_gtfs[
            trips_gtfs["route_id"] == selected_route
        ].copy()

        if not selected_trips.empty:

            st.success(
                f"Found {len(selected_trips)} trips for this route."
            )

            selected_trip = st.selectbox(
                "Select Trip",
                selected_trips["trip_id"].tolist()
            )

            # Get stop timings for selected trip
            trip_times = times_gtfs[
                times_gtfs["trip_id"] == selected_trip
            ].copy()

            trip_times["stop_sequence"] = pd.to_numeric(
                trip_times["stop_sequence"],
                errors="coerce"
            )

            trip_times = trip_times.sort_values("stop_sequence")

            # Join stop names and coordinates
            route_details = trip_times.merge(
                stops_gtfs,
                on="stop_id",
                how="left"
            )

            st.divider()
            st.subheader("Bus Stop Sequence")

            display_columns = [
                col for col in [
                    "stop_sequence",
                    "stop_name",
                    "stop_id",
                    "arrival_time",
                    "departure_time"
                ]
                if col in route_details.columns
            ]

            st.dataframe(
                route_details[display_columns],
                use_container_width=True
            )

            # Map
            if {"stop_lat", "stop_lon"}.issubset(route_details.columns):

                route_details["stop_lat"] = pd.to_numeric(
                    route_details["stop_lat"],
                    errors="coerce"
                )

                route_details["stop_lon"] = pd.to_numeric(
                    route_details["stop_lon"],
                    errors="coerce"
                )

                map_data = route_details.dropna(
                    subset=["stop_lat", "stop_lon"]
                )

                st.divider()
                st.subheader("Selected Bus Trip Map")

                if not map_data.empty:

                    st.map(
                        map_data,
                        latitude="stop_lat",
                        longitude="stop_lon"
                    )

                else:
                    st.warning("No valid coordinates found for this trip.")

            else:
                st.warning("Stop latitude or longitude columns are missing.")

        else:
            st.warning("No trips found for the selected route.")

    except Exception as e:
        st.error(f"Could not load GTFS data: {e}")

# ==========================================
# 13. FOOTER
# ==========================================

st.divider()
st.caption("AI-Based Public Transport Demand Prediction & Route Optimisation | Hyderabad Transit Project")