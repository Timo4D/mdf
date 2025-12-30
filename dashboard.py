"""
Dash dashboard to explore held-out test patients and run predictions
with the saved RandomForest model. Also supports uploading new CSV data
or entering feature values manually for prediction.

Key behaviors:
- Uses original sway_static_features.csv and aggregates features on-the-fly.
- Recreates the same train/test split from modeling_advanced.py (20% test, stratified, random_state=42).
- Tab 1: Lists held-out test patients; select one to predict with the saved RandomForest_model.joblib.
- Tab 2: Upload a CSV or fill in feature fields; uploaded rows can populate the fields; predict on demand.
"""

from pathlib import Path
from typing import Dict, List

import base64
from io import StringIO

import dash
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    dash_table,
    dcc,
    html,
    no_update,
)
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Data + split replicated from modeling_advanced.py
# Uses original data and aggregates features on-the-fly
# ------------------------------------------------------------
DATA_PATH = Path("sway_static_features.csv")
MODEL_PATH = Path("RandomForest_model.joblib")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find data file at {DATA_PATH.resolve()}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find model file at {MODEL_PATH.resolve()}")


# Metric prefixes and directions for aggregation (from modeling_advanced.py)
METRICS = ["AREA", "MDIST", "MFREQ", "MVELO", "RDIST", "TOTEX"]
DIRECTIONS = ["AP", "ML"]
META_COLS = ["part_id", "group", "age_num", "sex", "height", "weight", "BMI", "recorded_in_the_lab", "faller"]


def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw sway features to minimized format matching the model.
    
    Mirrors the aggregation logic from modeling_advanced.py:
    - Calculates mean for each metric across all exercises
    - Also calculates AP and ML specific means
    """
    feature_cols = [c for c in df.columns if c not in META_COLS]
    
    # Start with meta columns that exist in the dataframe
    result_cols = [c for c in META_COLS if c in df.columns]
    result = df[result_cols].copy()
    
    # For each metric, calculate the mean over all exercises
    for metric in METRICS:
        metric_cols = [c for c in feature_cols if c.startswith(metric)]
        if metric_cols:
            result[f"{metric}_mean"] = df[metric_cols].mean(axis=1, skipna=True)
    
    # AP- and ML-specific means
    for metric in METRICS:
        for d in DIRECTIONS:
            cols = [c for c in feature_cols if c.startswith(f"{metric}_{d}")]
            if cols:
                result[f"{metric}_{d}_mean"] = df[cols].mean(axis=1, skipna=True)
    
    return result


def prepare_data() -> Dict[str, object]:
    """Load data, aggregate features, and recreate the train/test split.
    
    Matches the preprocessing from modeling_advanced.py:
    - Aggregates raw sway features into mean values
    - Meta columns excluded from features: part_id, group, recorded_in_the_lab, faller
    - Remaining columns (age_num, sex, height, weight, BMI + aggregated sway metrics) are features
    """
    # Load and aggregate raw data
    df_raw = pd.read_csv(DATA_PATH)
    df = aggregate_features(df_raw)
    
    # Match the meta columns definition from modeling_advanced.py
    model_meta_cols = ["part_id", "group", "recorded_in_the_lab", "faller"]
    
    # Feature columns are all columns not in model_meta_cols
    feature_cols = [c for c in df.columns if c not in model_meta_cols]
    
    # Separate numeric and categorical features
    numeric_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    
    X = df[feature_cols]
    y = df["faller"].astype(int)

    # Stratified split with fixed seed to match modeling_advanced.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Keep the original (raw) rows for display by using indices from X_test
    test_rows_raw = df_raw.loc[X_test.index]
    # Also keep aggregated rows for prediction
    test_rows_aggregated = df.loc[X_test.index]

    return {
        "df_raw": df_raw,
        "df_aggregated": df,
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "test_rows_raw": test_rows_raw,
        "test_rows_aggregated": test_rows_aggregated,
        "feature_cols": feature_cols,
        "num_features": numeric_cols,
        "cat_features": categorical_cols,
    }


data_bundle = prepare_data()
df_raw = data_bundle["df_raw"]
X = data_bundle["X"]
X_test = data_bundle["X_test"]
test_rows_raw = data_bundle["test_rows_raw"]
test_rows = data_bundle["test_rows_aggregated"]
feature_cols = data_bundle["feature_cols"]
num_features = data_bundle["num_features"]
cat_features = data_bundle["cat_features"]

# Demographic columns that are features (not excluded from model input)
DEMOGRAPHIC_FEATURE_COLS = ["age_num", "sex", "height", "weight", "BMI"]

# Original raw feature columns for upload interface
# Includes sway metrics + demographic features (but not part_id, group, recorded_in_the_lab, faller)
raw_feature_cols = [c for c in df_raw.columns if c not in ["part_id", "group", "recorded_in_the_lab", "faller"]]

# Aggregated feature columns for simplified input mode
aggregated_feature_cols = DEMOGRAPHIC_FEATURE_COLS + [
    # General mean metrics
    "AREA_mean", "MDIST_mean", "MFREQ_mean", "MVELO_mean", "RDIST_mean", "TOTEX_mean",
    # AP-specific means
    "AREA_AP_mean", "MDIST_AP_mean", "MFREQ_AP_mean", "MVELO_AP_mean", "RDIST_AP_mean", "TOTEX_AP_mean",
    # ML-specific means
    "AREA_ML_mean", "MDIST_ML_mean", "MFREQ_ML_mean", "MVELO_ML_mean", "RDIST_ML_mean", "TOTEX_ML_mean",
]

# Explicitly preserve order from the split
patient_ids: List[int] = test_rows.index.tolist()
patient_labels: Dict[int, str] = {
    idx: f"Patient {i + 1}" for i, idx in enumerate(patient_ids)
}

# Data for the test table - round to 3 decimal places
table_df = test_rows.drop(columns=["group", "faller"], errors="ignore").copy()
# Round numeric columns to 3 decimal places
for col in table_df.select_dtypes(include=['float64', 'float32']).columns:
    table_df[col] = table_df[col].round(3)
table_df.insert(0, "Patient", [patient_labels[idx] for idx in patient_ids])
table_df.insert(1, "row_index", patient_ids)

table_columns = [{"name": "Patient", "id": "Patient", "hideable": False}] + [
    {
        "name": col, 
        "id": col, 
        "type": "numeric", 
        "format": {"specifier": ".3f"},
        "hideable": True
    } 
    if col not in ["Patient", "row_index"] and table_df[col].dtype in ['float64', 'float32'] 
    else {"name": col, "id": col, "hideable": col != "row_index"}
    for col in table_df.columns if col != "Patient"
]
table_data = table_df.to_dict("records")

# Store raw data for expandable view
raw_test_data = test_rows_raw.copy()
for col in raw_test_data.select_dtypes(include=['float64', 'float32']).columns:
    raw_test_data[col] = raw_test_data[col].round(3)
raw_test_data.insert(0, "Patient", [patient_labels[idx] for idx in patient_ids])
raw_test_data.insert(1, "row_index", patient_ids)
raw_table_data = raw_test_data.to_dict("records")

# Load model once; the saved joblib is expected to include preprocessing
model = load(MODEL_PATH)


# Color-coded category colors for feature groups
CATEGORY_COLORS = {
    "demographic": {"bg": "#e8f5e9", "border": "#4caf50", "label": "Demographics"},
    "general": {"bg": "#e3f2fd", "border": "#2196f3", "label": "General Sway"},
    "ap": {"bg": "#fff3e0", "border": "#ff9800", "label": "AP (Anterior-Posterior)"},
    "ml": {"bg": "#fce4ec", "border": "#e91e63", "label": "ML (Medial-Lateral)"},
}


def get_feature_category(col: str) -> str:
    """Determine the category of a feature column for color coding."""
    if col in DEMOGRAPHIC_FEATURE_COLS:
        return "demographic"
    elif "_AP_" in col or col.endswith("_AP") or "_AP_mean" in col:
        return "ap"
    elif "_ML_" in col or col.endswith("_ML") or "_ML_mean" in col:
        return "ml"
    else:
        return "general"


# Simple theme tokens for a clean, product-like look
THEME = {
    "bg": "linear-gradient(120deg, #f6f7fb 0%, #eef2f7 50%, #e8edf5 100%)",
    "card": "#ffffff",
    "border": "#e3e7ee",
    "text": "#0f172a",
    "muted": "#4b5563",
    "accent": "#2563eb",
    "accent_light": "#dbeafe",
}


def create_input_field(col: str, input_type: str = "aggregated") -> html.Div:
    """Create a color-coded input field for a feature column."""
    category = get_feature_category(col)
    colors = CATEGORY_COLORS[category]
    
    return html.Div(
        [
            html.Label(col, style={"fontSize": "11px", "fontWeight": "600", "wordBreak": "break-word"}),
            dcc.Input(
                id={"type": f"{input_type}-input", "col": col},
                type="number",
                placeholder="Enter value",
                style={"width": "100%", "padding": "6px", "borderRadius": "4px", "boxSizing": "border-box"},
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "0.3rem",
            "padding": "10px",
            "backgroundColor": colors["bg"],
            "borderLeft": f"4px solid {colors['border']}",
            "borderRadius": "6px",
            "minWidth": "180px",
            "minHeight": "70px",
        },
    )


def create_category_section(category: str, columns: List[str], input_type: str) -> html.Div:
    """Create a section with header for a category of input fields."""
    colors = CATEGORY_COLORS[category]
    
    return html.Div(
        [
            # Category header
            html.Div(
                [
                    html.Div(
                        style={
                            "width": "12px",
                            "height": "12px",
                            "backgroundColor": colors["border"],
                            "borderRadius": "3px",
                            "marginRight": "8px",
                        }
                    ),
                    html.Span(colors["label"], style={"fontWeight": "600", "fontSize": "14px"}),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "10px",
                    "paddingBottom": "6px",
                    "borderBottom": f"2px solid {colors['border']}",
                },
            ),
            # Input fields grid
            html.Div(
                [create_input_field(col, input_type) for col in columns],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))",
                    "gap": "0.5rem",
                },
            ),
        ],
        style={"marginBottom": "20px"},
    )


def create_grouped_inputs(columns: List[str], input_type: str) -> html.Div:
    """Create input fields grouped by category with headers."""
    # Group columns by category
    grouped = {"demographic": [], "general": [], "ap": [], "ml": []}
    for col in columns:
        category = get_feature_category(col)
        grouped[category].append(col)
    
    sections = []
    for category in ["demographic", "general", "ap", "ml"]:
        if grouped[category]:
            sections.append(create_category_section(category, grouped[category], input_type))
    
    return html.Div(sections)


def create_category_legend() -> html.Div:
    """Create a legend showing color codes for feature categories."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        style={
                            "width": "16px",
                            "height": "16px",
                            "backgroundColor": colors["bg"],
                            "borderLeft": f"3px solid {colors['border']}",
                            "borderRadius": "2px",
                        }
                    ),
                    html.Span(colors["label"], style={"fontSize": "12px", "marginLeft": "6px"}),
                ],
                style={"display": "flex", "alignItems": "center", "marginRight": "16px"},
            )
            for category, colors in CATEGORY_COLORS.items()
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": "12px", "gap": "8px"},
    )


# Generate color-coded column styles for the patient table
def get_table_column_styles(columns: List[str]) -> List[Dict]:
    """Generate conditional styling for table columns based on category."""
    conditions = []
    for col in columns:
        if col in ["Patient", "row_index"]:
            continue
        category = get_feature_category(col)
        colors = CATEGORY_COLORS[category]
        conditions.append({
            "if": {"column_id": col},
            "backgroundColor": colors["bg"],
        })
    return conditions

table_column_styles = get_table_column_styles(list(table_df.columns))


# ------------------------------------------------------------
# Dash app
# ------------------------------------------------------------
app: Dash = dash.Dash(__name__)

# Custom CSS for DataTable toggle button styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Style the column toggle button */
            .dash-spreadsheet-container .column-header-dropdown {
                background-color: #2563eb !important;
                color: white !important;
                border-radius: 4px !important;
                padding: 4px 8px !important;
                font-size: 11px !important;
                font-weight: 500 !important;
                cursor: pointer !important;
            }
            .dash-spreadsheet-container .column-header-dropdown:hover {
                background-color: #1d4ed8 !important;
            }
            /* Style the dropdown menu */
            .dash-spreadsheet-container .column-header-dropdown-menu {
                border-radius: 6px !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
                border: 1px solid #e3e7ee !important;
            }
            .dash-spreadsheet-container .column-header-dropdown-menu label {
                padding: 6px 12px !important;
                font-size: 12px !important;
            }
            .dash-spreadsheet-container .column-header-dropdown-menu label:hover {
                background-color: #f0f2f5 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "Faller Prediction",
                            style={
                                "fontSize": "14px",
                                "letterSpacing": "0.12em",
                                "textTransform": "uppercase",
                                "color": THEME["muted"],
                            },
                        ),
                        html.Div(
                            "Faller Prediction Dashboard",
                            style={
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "color": THEME["text"],
                                "marginTop": "4px",
                            },
                        ),
                        html.Div(
                            "Evaluate held-out patients, upload CSVs, or enter values to predict with the saved RandomForest model.",
                            style={
                                "color": THEME["muted"],
                                "marginTop": "6px",
                                "maxWidth": "900px",
                                "lineHeight": "1.5",
                            },
                        ),
                    ]
                ),
            ],
            style={
                "marginBottom": "18px",
                "padding": "18px 20px",
                "borderRadius": "14px",
                "background": THEME["card"],
                "border": f"1px solid {THEME['border']}",
                "boxShadow": "0 12px 30px rgba(15, 23, 42, 0.08)",
            },
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-test",
            children=[
                dcc.Tab(
                    label="Test patients",
                    value="tab-test",
                    children=[
                        html.P(
                            "Select a held-out test patient to run a prediction "
                            "with the saved RandomForest model."
                        ),
                        html.Div(
                            [
                                html.Label("Select one patient from the table (aggregated features):"),
                                # Color legend for table
                                create_category_legend(),
                                dash_table.DataTable(
                                    id="patient-table",
                                    columns=table_columns,
                                    data=table_data,
                                    row_selectable="single",
                                    selected_rows=[],
                                    page_size=10,
                                    sort_action="native",
                                    filter_action="native",
                                    column_selectable=False,
                                    style_table={
                                        "overflowX": "auto",
                                        "border": f"1px solid {THEME['border']}",
                                    },
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "6px 8px",
                                        "fontSize": "12px",
                                        "whiteSpace": "nowrap",
                                        "minWidth": "80px",
                                    },
                                    style_header={
                                        "fontWeight": "700",
                                        "backgroundColor": "#f0f2f5",
                                    },
                                    style_data_conditional=table_column_styles,
                                    hidden_columns=["row_index"],
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "Predict",
                                            id="predict-btn",
                                            n_clicks=0,
                                            disabled=True,
                                            style={
                                                "padding": "0.5rem 1.5rem",
                                                "marginTop": "0.75rem",
                                                "marginRight": "0.5rem",
                                                "cursor": "pointer",
                                                "backgroundColor": THEME["accent"],
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "6px",
                                                "fontWeight": "600",
                                            },
                                        ),
                                        html.Button(
                                            "Show Raw Data ▼",
                                            id="toggle-raw-data-btn",
                                            n_clicks=0,
                                            disabled=True,
                                            style={
                                                "padding": "0.5rem 1rem",
                                                "marginTop": "0.75rem",
                                                "cursor": "pointer",
                                                "backgroundColor": "#6b7280",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "6px",
                                                "fontWeight": "500",
                                            },
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center"},
                                ),
                                html.Div(
                                    id="prediction-output",
                                    style={"marginTop": "0.75rem", "fontWeight": "600"},
                                ),
                                # Expandable raw data section
                                html.Div(
                                    id="raw-data-section",
                                    children=[
                                        html.Hr(),
                                        html.H5("Raw Per-Exercise Data", style={"marginTop": "1rem"}),
                                        html.Div(id="raw-data-content"),
                                    ],
                                    style={"display": "none"},
                                ),
                                dcc.Store(id="raw-table-store", data=raw_table_data),
                            ],
                            style={
                                "width": "100%",
                                "padding": "0.75rem",
                                "background": THEME["card"],
                                "border": f"1px solid {THEME['border']}",
                                "borderRadius": "12px",
                                "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                            },
                        ),
                    ],
                    style={"padding": "12px"},
                ),
                dcc.Tab(
                    label="Upload data",
                    value="tab-upload",
                    children=[
                        html.Div(
                            [
                                html.P(
                                    "Upload a CSV or manually enter feature values, then predict "
                                    "with the saved RandomForest model. Uploading a file will populate "
                                    "the fields from the selected row."
                                ),
                                dcc.Upload(
                                    id="upload-area",
                                    children=html.Div(
                                        ["Drag and drop or ", html.B("select a CSV file")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "80px",
                                        "lineHeight": "80px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "6px",
                                        "textAlign": "center",
                                        "backgroundColor": THEME["accent_light"],
                                        "marginBottom": "0.75rem",
                                        "color": THEME["accent"],
                                        "fontWeight": "600",
                                    },
                                ),
                                html.Div(id="upload-status", style={"marginBottom": "0.5rem"}),
                                dash_table.DataTable(
                                    id="upload-table",
                                    columns=[{"name": c, "id": c} for c in ["row_index"] + raw_feature_cols],
                                    data=[],
                                    row_selectable="single",
                                    selected_rows=[],
                                    page_size=5,
                                    style_table={
                                        "overflowX": "auto",
                                        "border": f"1px solid {THEME['border']}",
                                    },
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "6px 8px",
                                        "fontSize": "13px",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                    },
                                    style_header={
                                        "fontWeight": "700",
                                        "backgroundColor": "#f0f2f5",
                                    },
                                    column_selectable=False,
                                ),
                                html.Hr(),
                                html.H4("Manual Input"),
                                html.P(
                                    "Choose input mode: Aggregated (simplified mean values) or Raw (per-exercise data).",
                                    style={"fontSize": "13px", "color": THEME["muted"]},
                                ),
                                # Input mode toggle
                                dcc.RadioItems(
                                    id="input-mode-toggle",
                                    options=[
                                        {"label": " Aggregated Features (simplified)", "value": "aggregated"},
                                        {"label": " Raw Data (per-exercise)", "value": "raw"},
                                    ],
                                    value="aggregated",
                                    inline=True,
                                    style={"marginBottom": "12px", "fontWeight": "500"},
                                    inputStyle={"marginRight": "6px"},
                                    labelStyle={"marginRight": "20px"},
                                ),
                                # Color legend
                                create_category_legend(),
                                # Aggregated inputs section
                                html.Div(
                                    id="aggregated-inputs-section",
                                    children=[
                                        create_grouped_inputs(aggregated_feature_cols, "aggregated"),
                                    ],
                                    style={"display": "block"},
                                ),
                                # Raw inputs section (hidden by default)
                                html.Div(
                                    id="raw-inputs-section",
                                    children=[
                                        create_grouped_inputs(raw_feature_cols, "raw"),
                                    ],
                                    style={"display": "none"},
                                ),
                                html.Button(
                                    "Predict",
                                    id="predict-manual-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "0.5rem 1.5rem",
                                        "marginTop": "1rem",
                                        "cursor": "pointer",
                                        "backgroundColor": THEME["accent"],
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "6px",
                                        "fontWeight": "600",
                                    },
                                ),
                                html.Div(
                                    id="manual-predict-output",
                                    style={"marginTop": "0.75rem", "fontWeight": "600"},
                                ),
                                dcc.Store(id="uploaded-rows"),
                                dcc.Store(id="current-input-mode", data="aggregated"),
                            ],
                            style={
                                "padding": "0.75rem",
                                "background": THEME["card"],
                                "border": f"1px solid {THEME['border']}",
                                "borderRadius": "12px",
                                "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                            },
                        )
                    ],
                    style={"padding": "12px"},
                ),
            ],
            style={
                "background": THEME["card"],
                "borderRadius": "12px",
                "border": f"1px solid {THEME['border']}",
                "boxShadow": "0 10px 28px rgba(15, 23, 42, 0.07)",
            },
        ),
    ],
    style={
        "fontFamily": "Segoe UI, -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif",
        "padding": "1.5rem",
        "background": THEME["bg"],
        "minHeight": "100vh",
        "color": THEME["text"],
    },
)


@app.callback(
    [
        Output("predict-btn", "disabled"),
        Output("toggle-raw-data-btn", "disabled"),
    ],
    Input("patient-table", "selected_rows"),
)
def toggle_buttons(selected_idx):
    """Enable buttons when a patient is selected."""
    disabled = not selected_idx
    return disabled, disabled


@app.callback(
    [
        Output("raw-data-section", "style"),
        Output("toggle-raw-data-btn", "children"),
        Output("raw-data-content", "children"),
    ],
    Input("toggle-raw-data-btn", "n_clicks"),
    State("patient-table", "selected_rows"),
    State("raw-table-store", "data"),
    State("raw-data-section", "style"),
    prevent_initial_call=True,
)
def toggle_raw_data(n_clicks, selected_rows, raw_data, current_style):
    """Toggle display of raw per-exercise data for selected patient."""
    if not selected_rows:
        return {"display": "none"}, "Show Raw Data ▼", None
    
    # Toggle visibility
    is_visible = current_style.get("display") == "block"
    
    if is_visible:
        # Hide section
        return {"display": "none"}, "Show Raw Data ▼", None
    else:
        # Show section with selected patient's raw data
        patient_idx = selected_rows[0]
        patient_raw = raw_data[patient_idx]
        
        # Group raw data by category for display
        grouped_data = {"demographic": [], "general": [], "ap": [], "ml": []}
        for col, val in patient_raw.items():
            if col in ["Patient", "row_index"]:
                continue
            category = get_feature_category(col)
            if val is not None and val != "":
                grouped_data[category].append((col, val))
        
        # Create display sections
        sections = []
        for category in ["demographic", "general", "ap", "ml"]:
            if grouped_data[category]:
                colors = CATEGORY_COLORS[category]
                items = grouped_data[category]
                
                sections.append(
                    html.Div(
                        [
                            html.Div(
                                colors["label"],
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "13px",
                                    "marginBottom": "6px",
                                    "color": colors["border"],
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(f"{col}: ", style={"fontWeight": "500", "fontSize": "11px"}),
                                            html.Span(f"{val:.3f}" if isinstance(val, float) else str(val), 
                                                     style={"fontSize": "11px"}),
                                        ],
                                        style={
                                            "padding": "4px 8px",
                                            "backgroundColor": colors["bg"],
                                            "borderRadius": "4px",
                                            "display": "inline-block",
                                            "margin": "2px",
                                        },
                                    )
                                    for col, val in items
                                ],
                                style={"display": "flex", "flexWrap": "wrap", "gap": "4px"},
                            ),
                        ],
                        style={"marginBottom": "12px"},
                    )
                )
        
        content = html.Div(
            [html.Div(f"Patient: {patient_raw['Patient']}", style={"fontWeight": "600", "marginBottom": "10px"})] + sections
        )
        
        return {"display": "block"}, "Hide Raw Data ▲", content


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("patient-table", "selected_rows"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, selected_idx):
    """Run prediction for the selected patient using the saved model."""
    if not selected_idx:
        return "Please select a patient first."

    table_row_position = selected_idx[0]
    patient_index = patient_ids[table_row_position]

    # Single-row DataFrame with the same columns as training
    row_features = X.loc[[patient_index]]
    pred = model.predict(row_features)[0]

    prob_text = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_features)[0][1]
        prob_text = f" | Probability of faller: {proba:.3f}"

    label = "Faller" if pred == 1 else "Non-faller"
    return f"{patient_labels[patient_index]}: {label}{prob_text}"


def decode_contents(contents: str) -> pd.DataFrame:
    """Decode a dash upload contents string into a DataFrame."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(StringIO(decoded.decode("utf-8")))


@app.callback(
    [
        Output("upload-status", "children"),
        Output("upload-table", "data"),
        Output("uploaded-rows", "data"),
        Output("upload-table", "selected_rows"),
    ],
    Input("upload-area", "contents"),
    State("upload-area", "filename"),
)
def handle_upload(contents, filename):
    """Parse uploaded CSV with original raw features, align and populate table + store."""
    if contents is None:
        return "No file uploaded yet.", [], [], []

    try:
        df_upload = decode_contents(contents)
    except Exception as e:  # pragma: no cover - user input parsing
        return f"Could not read file: {e}", [], [], []

    # Keep only known raw feature columns; add missing as None
    missing_cols = [c for c in raw_feature_cols if c not in df_upload.columns]
    present_cols = [c for c in df_upload.columns if c in raw_feature_cols]

    aligned = df_upload.reindex(columns=raw_feature_cols)
    aligned = aligned.infer_objects()
    aligned = aligned.replace({pd.NA: None})

    # Attach a row index to track selections
    aligned = aligned.reset_index().rename(columns={"index": "row_index"})

    data_records = aligned.to_dict("records")
    msg_parts = [f"Loaded '{filename}' with {len(aligned)} rows."]
    if missing_cols:
        msg_parts.append(f"Missing columns filled with blank: {', '.join(missing_cols[:5])}..." if len(missing_cols) > 5 else f"Missing columns filled with blank: {', '.join(missing_cols)}.")

    # Auto-select first row so fields populate
    selected_rows = [0] if data_records else []

    return " ".join(msg_parts), data_records, data_records, selected_rows


@app.callback(
    [
        Output("aggregated-inputs-section", "style"),
        Output("raw-inputs-section", "style"),
        Output("current-input-mode", "data"),
    ],
    Input("input-mode-toggle", "value"),
)
def toggle_input_sections(mode):
    """Toggle visibility of aggregated/raw input sections based on selected mode."""
    if mode == "aggregated":
        return {"display": "block"}, {"display": "none"}, "aggregated"
    else:
        return {"display": "none"}, {"display": "block"}, "raw"


@app.callback(
    Output({"type": "raw-input", "col": ALL}, "value"),
    Input("upload-table", "selected_rows"),
    State("uploaded-rows", "data"),
)
def populate_raw_inputs(selected_rows, data_records):
    """Populate raw input fields from selected uploaded row."""
    if not data_records or not selected_rows:
        return [None for _ in raw_feature_cols]

    row = data_records[selected_rows[0]]
    return [row.get(col) for col in raw_feature_cols]


@app.callback(
    Output("manual-predict-output", "children"),
    Input("predict-manual-btn", "n_clicks"),
    State("current-input-mode", "data"),
    State({"type": "aggregated-input", "col": ALL}, "value"),
    State({"type": "raw-input", "col": ALL}, "value"),
    prevent_initial_call=True,
)
def predict_manual(n_clicks, input_mode, aggregated_values, raw_values):
    """Predict from manual inputs. Handles both aggregated and raw input modes."""
    try:
        if input_mode == "aggregated":
            # Use aggregated values directly
            row_dict = {}
            for col, val in zip(aggregated_feature_cols, aggregated_values):
                row_dict[col] = None if val in (None, "") else float(val)
            
            # Create DataFrame with aggregated features (already in model format)
            model_features = pd.DataFrame([row_dict], columns=aggregated_feature_cols)
            # Reorder to match feature_cols expected by model
            model_features = model_features.reindex(columns=feature_cols, fill_value=None)
        else:
            # Use raw values and aggregate
            row_dict = {}
            for col, val in zip(raw_feature_cols, raw_values):
                row_dict[col] = None if val in (None, "") else float(val)
            
            # Create DataFrame with raw features
            raw_df = pd.DataFrame([row_dict], columns=raw_feature_cols)
            
            # Aggregate to model format
            aggregated_df = aggregate_features(raw_df)
            
            # Extract only the feature columns the model expects
            model_features = aggregated_df[feature_cols]
        
        pred = model.predict(model_features)[0]

        prob_text = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(model_features)[0][1]
            prob_text = f" | Probability of faller: {proba:.3f}"

        label = "Faller" if pred == 1 else "Non-faller"
        return f"Prediction: {label}{prob_text}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
