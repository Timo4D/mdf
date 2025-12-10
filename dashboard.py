"""
Dash dashboard to explore held-out test patients and run predictions
with the saved RandomForest model. Also supports uploading new CSV data
or entering feature values manually for prediction.

Key behaviors:
- Recreates the same train/test split from modeling.ipynb (20% test, stratified, random_state=42).
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
# Data + split replicated from modeling.ipynb
# ------------------------------------------------------------
DATA_PATH = Path("sway_static_features.csv")
MODEL_PATH = Path("RandomForest_model.joblib")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find data file at {DATA_PATH.resolve()}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find model file at {MODEL_PATH.resolve()}")


def prepare_data() -> Dict[str, object]:
    """Load data, build features, and recreate the train/test split."""
    df = pd.read_csv(DATA_PATH)
    working = df.copy()

    # Clean age into numeric if needed (mirrors modeling.ipynb logic)
    if "age" in working.columns and "age_num" not in working.columns:
        working["age_num"] = pd.to_numeric(
            working["age"].replace({">89": "90"}), errors="coerce"
        )

    # Columns not used as features
    non_feature_cols = {"part_id", "faller", "age", "group"}
    non_feature_cols.add("clinically-at-risk")

    numeric_cols = working.select_dtypes(include=["int64", "float64"]).columns
    num_features = [c for c in numeric_cols if c not in non_feature_cols]

    categorical_cols = working.select_dtypes(include=["object"]).columns
    cat_features = [c for c in categorical_cols if c not in non_feature_cols]

    X = working[num_features + cat_features]
    y = working["faller"].astype(int)

    # Stratified split with fixed seed to match notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Keep the original (unprocessed) rows for display by using indices from X_test
    test_rows = df.loc[X_test.index]

    return {
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "test_rows": test_rows,
        "num_features": num_features,
        "cat_features": cat_features,
    }


data_bundle = prepare_data()
X = data_bundle["X"]
X_test = data_bundle["X_test"]
test_rows = data_bundle["test_rows"]
num_features = data_bundle["num_features"]
cat_features = data_bundle["cat_features"]

# Explicitly preserve order from the split
patient_ids: List[int] = test_rows.index.tolist()
patient_labels: Dict[int, str] = {
    idx: f"Patient {i + 1}" for i, idx in enumerate(patient_ids)
}

# Data for the test table
table_df = test_rows.drop(columns=["group", "faller"], errors="ignore").copy()
table_df.insert(0, "Patient", [patient_labels[idx] for idx in patient_ids])
table_df.insert(1, "row_index", patient_ids)

table_columns = [{"name": "Patient", "id": "Patient"}] + [
    {"name": col, "id": col} for col in table_df.columns if col != "Patient"
]
table_data = table_df.to_dict("records")

# Load model once; the saved joblib is expected to include preprocessing
model = load(MODEL_PATH)


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


# ------------------------------------------------------------
# Dash app
# ------------------------------------------------------------
app: Dash = dash.Dash(__name__)

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
                                html.Label("Select one patient from the table:"),
                                dash_table.DataTable(
                                    id="patient-table",
                                    columns=table_columns,
                                    data=table_data,
                                    row_selectable="single",
                                    selected_rows=[],
                                    page_size=len(patient_ids),
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
                                        "fontSize": "13px",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                    },
                                    style_header={
                                        "fontWeight": "700",
                                        "backgroundColor": "#f0f2f5",
                                    },
                                    hidden_columns=["row_index"],
                                ),
                                html.Button(
                                    "Predict",
                                    id="predict-btn",
                                    n_clicks=0,
                                    disabled=True,
                                    style={
                                        "padding": "0.4rem 1rem",
                                        "marginTop": "0.5rem",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Div(
                                    id="prediction-output",
                                    style={"marginTop": "0.75rem", "fontWeight": "600"},
                                ),
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
                                    columns=[{"name": c, "id": c} for c in ["row_index"] + list(X.columns)],
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
                                html.H4("Manual / populated inputs"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(col),
                                                dcc.Input(
                                                    id={"type": "feature-input", "col": col},
                                                    type="number" if col in num_features else "text",
                                                    placeholder="Enter value",
                                                    style={"width": "100%", "padding": "6px"},
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "flexDirection": "column",
                                                "gap": "0.2rem",
                                            },
                                        )
                                        for col in X.columns
                                    ],
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(auto-fill, minmax(220px, 1fr))",
                                        "gap": "0.75rem",
                                    },
                                ),
                                html.Button(
                                    "Predict",
                                    id="predict-manual-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "0.4rem 1rem",
                                        "marginTop": "0.75rem",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Div(
                                    id="manual-predict-output",
                                    style={"marginTop": "0.75rem", "fontWeight": "600"},
                                ),
                                dcc.Store(id="uploaded-rows"),
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
    Output("predict-btn", "disabled"),
    Input("patient-table", "selected_rows"),
)
def toggle_predict_button(selected_idx):
    """Enable predict button when a patient is selected."""
    return not selected_idx


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
    """Parse uploaded CSV, align to feature columns, and populate table + store."""
    if contents is None:
        return "No file uploaded yet.", [], [], []

    try:
        df_upload = decode_contents(contents)
    except Exception as e:  # pragma: no cover - user input parsing
        return f"Could not read file: {e}", [], [], []

    # Keep only known columns; add missing as None
    missing_cols = [c for c in X.columns if c not in df_upload.columns]
    present_cols = [c for c in df_upload.columns if c in X.columns]

    aligned = df_upload.reindex(columns=X.columns)
    aligned = aligned.infer_objects()
    aligned = aligned.replace({pd.NA: None})

    # Attach a row index to track selections
    aligned = aligned.reset_index().rename(columns={"index": "row_index"})

    data_records = aligned.to_dict("records")
    msg_parts = [f"Loaded '{filename}' with {len(aligned)} rows."]
    if missing_cols:
        msg_parts.append(f"Missing columns filled with blank: {', '.join(missing_cols)}.")

    # Auto-select first row so fields populate
    selected_rows = [0] if data_records else []

    return " ".join(msg_parts), data_records, data_records, selected_rows


@app.callback(
    Output({"type": "feature-input", "col": ALL}, "value"),
    Input("upload-table", "selected_rows"),
    State("uploaded-rows", "data"),
)
def populate_inputs(selected_rows, data_records):
    """Populate manual input fields from selected uploaded row."""
    if not data_records or not selected_rows:
        return [None for _ in X.columns]

    row = data_records[selected_rows[0]]
    return [row.get(col) for col in X.columns]


@app.callback(
    Output("manual-predict-output", "children"),
    Input("predict-manual-btn", "n_clicks"),
    State({"type": "feature-input", "col": ALL}, "value"),
    prevent_initial_call=True,
)
def predict_manual(n_clicks, values):
    """Predict from manual/filled inputs."""
    row_dict = {}
    for col, val in zip(X.columns, values):
        if col in num_features:
            row_dict[col] = None if val in (None, "") else float(val)
        else:
            row_dict[col] = None if val in (None, "") else val

    row_df = pd.DataFrame([row_dict], columns=X.columns)
    pred = model.predict(row_df)[0]

    prob_text = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_df)[0][1]
        prob_text = f" | Probability of faller: {proba:.3f}"

    label = "Faller" if pred == 1 else "Non-faller"
    return f"Prediction: {label}{prob_text}"


if __name__ == "__main__":
    app.run(debug=True)
