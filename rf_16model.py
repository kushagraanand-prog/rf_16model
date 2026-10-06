import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CL Slag Cu-Loss Counterfactual",
    layout="wide"
)


# ============================================================
# 3. TITLE
# ============================================================

st.title(
    "CL Slag Cu-Loss Counterfactual Analysis"
)

st.markdown(
    """
    ### XGBoost-based engineered-feature counterfactual analysis

    This application evaluates how the existing process condition
    can be changed to increase the predicted probability of the
    lower Cu-loss class.

    The counterfactual optimizer modifies only the four actual
    S-furnace operating variables:

    - C-Slag Feed Rate
    - Silica Feed Rate
    - S-Furnace Air
    - S-Furnace Oxygen

    Blend composition, concentrate feed rate, and the other
    process/model variables remain fixed.
    """
)


# ============================================================
# 4. MODEL / ARTIFACT PATHS
# ============================================================

# Works in Kaggle Notebook
APP_DIR = os.getcwd()

MODEL_PATH = os.path.join(
    APP_DIR,
    "xgb_final.pkl"
)

FEATURE_ORDER_PATH = os.path.join(
    APP_DIR,
    "feature_order.pkl"
)

X_TRAIN_PATH = os.path.join(
    APP_DIR,
    "X_train.npy"
)

Y_TRAIN_PATH = os.path.join(
    APP_DIR,
    "y_train.npy"
)

RAW_X_TRAIN_PATH = os.path.join(
    APP_DIR,
    "raw_X_train.pkl"
)


# ============================================================
# 5. EXACT MODEL FEATURE ORDER
# ============================================================

EXPECTED_FEATURE_ORDER = [
    "Cu_Input",
    "Fe_Input",
    "Sulfur_Feed",
    "Matte_Flow",
    "Slag_Flow",
    "SILICA FEED RATE ",
    "O2_to_Sulfur",
    "Fe/SiO2",
    "Fe3O4_Cls",
    "Matte Grade"
]


# ============================================================
# 6. COUNTERFACTUAL SETTINGS
# ============================================================

TARGET_CLASS = 1

DESIRED_PROB = 0.90

TARGET_PENALTY = 1_000_000.0

# Small penalty against unnecessary movement
CHANGE_WEIGHT = 1.0


# ============================================================
# 7. DE SEED SETTINGS
# ============================================================

# 20 seeds for DE configuration screening
SCREENING_SEEDS = list(
    range(42, 62)
)

# 50 independent seeds for final optimization
FINAL_SEEDS = list(
    range(100, 150)
)


# ============================================================
# 8. DE CONFIGURATIONS
# ============================================================

DE_CONFIGURATIONS = [

    {
        "name": "DE-1: Best1Bin Moderate",
        "strategy": "best1bin",
        "maxiter": 150,
        "popsize": 20,
        "mutation": (0.5, 1.0),
        "recombination": 0.7,
        "tol": 1e-3,
        "polish": True
    },

    {
        "name": "DE-2: Best1Bin Large",
        "strategy": "best1bin",
        "maxiter": 250,
        "popsize": 30,
        "mutation": (0.5, 1.0),
        "recombination": 0.8,
        "tol": 1e-4,
        "polish": True
    },

    {
        "name": "DE-3: Best1Bin High Exploration",
        "strategy": "best1bin",
        "maxiter": 300,
        "popsize": 30,
        "mutation": (0.5, 1.5),
        "recombination": 0.8,
        "tol": 1e-4,
        "polish": True
    },

    {
        "name": "DE-4: Rand1Bin",
        "strategy": "rand1bin",
        "maxiter": 250,
        "popsize": 30,
        "mutation": (0.5, 1.2),
        "recombination": 0.8,
        "tol": 1e-4,
        "polish": True
    },

    {
        "name": "DE-5: Rand2Bin",
        "strategy": "rand2bin",
        "maxiter": 300,
        "popsize": 30,
        "mutation": (0.5, 1.5),
        "recombination": 0.8,
        "tol": 1e-4,
        "polish": True
    },

    {
        "name": "DE-6: Best2Bin",
        "strategy": "best2bin",
        "maxiter": 250,
        "popsize": 30,
        "mutation": (0.5, 1.2),
        "recombination": 0.8,
        "tol": 1e-4,
        "polish": True
    },

    {
        "name": "DE-7: CurrentToBest",
        "strategy": "currenttobest1bin",
        "maxiter": 250,
        "popsize": 30,
        "mutation": (0.5, 1.0),
        "recombination": 0.9,
        "tol": 1e-4,
        "polish": True
    },

    {
        "name": "DE-8: Large Population",
        "strategy": "best1bin",
        "maxiter": 300,
        "popsize": 40,
        "mutation": (0.5, 1.2),
        "recombination": 0.8,
        "tol": 1e-4,
        "polish": True
    }
]


# ============================================================
# 9. REQUIRED RAW COLUMNS
# ============================================================

REQUIRED_RAW_COLUMNS = [
    "Cu",
    "Fe",
    "S",
    "Al2O3",
    "CaO",
    "MgO",
    "CONC. FEED RATE",
    "C-SLAG FEED RATE - S Furnace",
    "SILICA FEED RATE ",
    "S-FURNACE AIR",
    "S-FURNACE OXYGEN",
    "S_in_Matte",
    "Fe_in_Cl_Slag",
    "Fe/SiO2",
    "Fe3O4_Cls",
    "Matte Grade"
]


# ============================================================
# 10. CHECK ARTIFACT FILES
# ============================================================

def check_artifact_files():

    required_files = {

        "XGBoost model":
            MODEL_PATH,

        "Feature order":
            FEATURE_ORDER_PATH,

        "X_train":
            X_TRAIN_PATH,

        "y_train":
            Y_TRAIN_PATH,

        "Raw X_train":
            RAW_X_TRAIN_PATH
    }

    missing_files = []

    for name, filepath in required_files.items():

        if not os.path.isfile(filepath):

            missing_files.append(
                f"{name}: {filepath}"
            )

    if missing_files:

        error_message = (
            "The following required artifact files "
            "were not found:\n\n"
        )

        error_message += "\n".join(
            missing_files
        )

        error_message += (
            "\n\nAll five files must be in the "
            "same folder as app.py."
        )

        raise FileNotFoundError(
            error_message
        )


# ============================================================
# 11. LOAD XGBOOST MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.isfile(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"XGBoost model not found: "
            f"{MODEL_PATH}"
        )

    try:

        loaded_model = joblib.load(
            MODEL_PATH
        )

    except Exception as e:

        raise RuntimeError(
            "The XGBoost model file exists, "
            "but it could not be loaded.\n"
            f"File: {MODEL_PATH}\n"
            f"Original error: {e}"
        ) from e

    if not hasattr(
        loaded_model,
        "predict_proba"
    ):

        raise TypeError(
            "xgb_final.pkl was loaded, but "
            "the object does not have "
            "predict_proba(). Make sure the "
            "file was created using XGBClassifier."
        )

    if not hasattr(
        loaded_model,
        "predict"
    ):

        raise TypeError(
            "xgb_final.pkl was loaded, but "
            "the object does not have predict()."
        )

    return loaded_model


# ============================================================
# 12. LOAD FEATURE ORDER
# ============================================================

@st.cache_resource
def load_feature_order():

    if not os.path.isfile(
        FEATURE_ORDER_PATH
    ):

        raise FileNotFoundError(
            f"Feature-order file not found: "
            f"{FEATURE_ORDER_PATH}"
        )

    try:

        feature_order = joblib.load(
            FEATURE_ORDER_PATH
        )

    except Exception as e:

        raise RuntimeError(
            "feature_order.pkl exists, "
            "but it could not be loaded.\n"
            f"File: {FEATURE_ORDER_PATH}\n"
            f"Original error: {e}"
        ) from e

    if not isinstance(
        feature_order,
        (list, tuple)
    ):

        raise TypeError(
            "feature_order.pkl must contain "
            "a list or tuple of feature names."
        )

    return list(
        feature_order
    )


# ============================================================
# 13. LOAD TRAINING DATA
# ============================================================

@st.cache_data
def load_training_data():

    try:

        X_train = np.load(
            X_TRAIN_PATH,
            allow_pickle=False
        )

    except Exception as e:

        raise RuntimeError(
            "X_train.npy exists, but it "
            "could not be loaded.\n"
            f"File: {X_TRAIN_PATH}\n"
            f"Original error: {e}"
        ) from e

    try:

        y_train = np.load(
            Y_TRAIN_PATH,
            allow_pickle=False
        )

    except Exception as e:

        raise RuntimeError(
            "y_train.npy exists, but it "
            "could not be loaded.\n"
            f"File: {Y_TRAIN_PATH}\n"
            f"Original error: {e}"
        ) from e

    try:

        raw_X_train = joblib.load(
            RAW_X_TRAIN_PATH
        )

    except Exception as e:

        raise RuntimeError(
            "raw_X_train.pkl exists, but it "
            "could not be loaded.\n"
            f"File: {RAW_X_TRAIN_PATH}\n"
            f"Original error: {e}"
        ) from e

    if not isinstance(
        raw_X_train,
        pd.DataFrame
    ):

        raise TypeError(
            "raw_X_train.pkl does not contain "
            "a pandas DataFrame."
        )

    return (
        X_train,
        y_train,
        raw_X_train
    )


# ============================================================
# 14. LOAD EVERYTHING
# ============================================================

try:

    check_artifact_files()

    model = load_model()

    saved_feature_order = (
        load_feature_order()
    )

    (
        X_train,
        y_train,
        raw_X_train
    ) = load_training_data()

except Exception as e:

    st.error(
        "Unable to load the required "
        "model artifacts."
    )

    st.exception(e)

    st.subheader(
        "Application folder"
    )

    st.code(
        APP_DIR
    )

    st.subheader(
        "Artifact file check"
    )

    artifact_status = {

        "xgb_final.pkl":
            MODEL_PATH,

        "feature_order.pkl":
            FEATURE_ORDER_PATH,

        "X_train.npy":
            X_TRAIN_PATH,

        "y_train.npy":
            Y_TRAIN_PATH,

        "raw_X_train.pkl":
            RAW_X_TRAIN_PATH
    }

    for filename, filepath in (
        artifact_status.items()
    ):

        if os.path.isfile(filepath):

            st.success(
                f"✓ {filename}"
            )

        else:

            st.error(
                f"✗ {filename} — NOT FOUND"
            )

    st.stop()


# ============================================================
# 15. VERIFY FEATURE ORDER
# ============================================================

if (
    saved_feature_order
    != EXPECTED_FEATURE_ORDER
):

    st.error(
        "Feature-order mismatch detected."
    )

    st.subheader(
        "Expected feature order"
    )

    st.code(
        "\n".join(
            f"{i + 1}. {feature}"
            for i, feature in enumerate(
                EXPECTED_FEATURE_ORDER
            )
        )
    )

    st.subheader(
        "Loaded feature order"
    )

    st.code(
        "\n".join(
            f"{i + 1}. {feature}"
            for i, feature in enumerate(
                saved_feature_order
            )
        )
    )

    st.stop()


# ============================================================
# 16. SET FINAL FEATURE ORDER
# ============================================================

FEATURE_ORDER = (
    EXPECTED_FEATURE_ORDER.copy()
)


# ============================================================
# 17. VERIFY X_train
# ============================================================

if X_train.ndim != 2:

    st.error(
        "X_train.npy must be a "
        "2-dimensional array."
    )

    st.write(
        "Loaded shape:",
        X_train.shape
    )

    st.stop()


if (
    X_train.shape[1]
    != len(FEATURE_ORDER)
):

    st.error(
        "X_train feature count does not "
        "match the saved feature order."
    )

    st.write(
        "X_train columns:",
        X_train.shape[1]
    )

    st.write(
        "Expected columns:",
        len(FEATURE_ORDER)
    )

    st.stop()


# ============================================================
# 18. VERIFY y_train
# ============================================================

if y_train.ndim != 1:

    y_train = np.ravel(
        y_train
    )


if len(X_train) != len(y_train):

    st.error(
        "X_train and y_train have "
        "different numbers of rows."
    )

    st.write(
        "X_train rows:",
        len(X_train)
    )

    st.write(
        "y_train rows:",
        len(y_train)
    )

    st.stop()


# ============================================================
# 19. VERIFY RAW TRAINING DATA
# ============================================================

missing_raw_columns = [
    col
    for col in REQUIRED_RAW_COLUMNS
    if col not in raw_X_train.columns
]


if missing_raw_columns:

    st.error(
        "raw_X_train.pkl is missing "
        "required columns."
    )

    st.write(
        "Missing columns:"
    )

    st.code(
        "\n".join(
            missing_raw_columns
        )
    )

    st.stop()


# ============================================================
# 20. VERIFY ROW ALIGNMENT
# ============================================================

if len(X_train) != len(raw_X_train):

    st.error(
        "X_train and raw_X_train have "
        "different numbers of rows."
    )

    st.write(
        "X_train rows:",
        len(X_train)
    )

    st.write(
        "raw_X_train rows:",
        len(raw_X_train)
    )

    st.stop()


# ============================================================
# 21. SIDEBAR
# ============================================================

st.sidebar.success(
    "✓ Model artifacts loaded successfully"
)

st.sidebar.caption(
    f"Artifact folder: {APP_DIR}"
)

with st.sidebar:

    st.header(
        "Model"
    )

    st.success(
        "XGBoost Classifier"
    )

    st.write(
        "Target class:"
    )

    st.write(
        "Class 1 = Lower Cu-loss class"
    )

    st.write(
        "Class 0 = Higher Cu-loss class"
    )

    st.write(
        f"Desired probability: "
        f"{DESIRED_PROB:.0%}"
    )

    st.divider()

    st.header(
        "Counterfactual philosophy"
    )

    st.write(
        """
        Blend composition and concentrate feed
        are kept constant.

        Only the four actual S-furnace control
        variables are optimized.

        Fe/SiO₂, Fe₃O₄ and Matte Grade remain
        fixed during counterfactual generation.
        """
    )


# ============================================================
# 22. ENGINEERED FEATURE CALCULATION
# ============================================================

def calculate_engineered_features(
    raw_values,
    c_slag_feed,
    silica_feed,
    furnace_air,
    furnace_oxygen,
    fe_sio2,
    fe3o4_cls,
    matte_grade
):

    """
    Calculate the existing 10 model features.

    IMPORTANT:
    This function is NOT part of DE tuning.

    DE only supplies different values for:
        c_slag_feed
        silica_feed
        furnace_air
        furnace_oxygen

    All other inputs remain fixed.
    """

    # --------------------------------------------------------
    # Fixed blend / concentrate values
    # --------------------------------------------------------

    Cu = float(
        raw_values["Cu"]
    )

    Fe = float(
        raw_values["Fe"]
    )

    S = float(
        raw_values["S"]
    )

    conc_feed = float(
        raw_values["CONC. FEED RATE"]
    )

    S_in_Matte = float(
        raw_values["S_in_Matte"]
    )

    # --------------------------------------------------------
    # 1. Cu Input
    # --------------------------------------------------------

    Cu_Input = (
        (Cu / 100.0)
        * conc_feed
        - 0.125 * c_slag_feed
    )

    # --------------------------------------------------------
    # 2. Fe Input
    # --------------------------------------------------------

    Fe_Input = (
        (Fe / 100.0)
        * conc_feed
        - 0.45 * c_slag_feed
    )

    # --------------------------------------------------------
    # 3. Sulfur Feed
    # --------------------------------------------------------

    Sulfur_Feed = (
        (S / 100.0)
        * conc_feed
    )

    # --------------------------------------------------------
    # 4. Matte Flow
    # --------------------------------------------------------

    matte_grade_decimal = (
        matte_grade / 100.0
    )

    if matte_grade_decimal <= 0:

        return None

    Matte_Flow = (
        Cu_Input
        / matte_grade_decimal
    )

    # --------------------------------------------------------
    # 5. Fe in Matte
    # --------------------------------------------------------

    Fe_in_Matte = (
        100.0
        - matte_grade
        - S_in_Matte
        - 0.5
    )

    # --------------------------------------------------------
    # 6. Fe to Matte
    # --------------------------------------------------------

    Fe_to_Matte = (
        (Fe_in_Matte / 100.0)
        * Matte_Flow
    )

    # --------------------------------------------------------
    # 7. Fe to Slag
    # --------------------------------------------------------

    Fe_to_Slag = (
        Fe_Input
        - Fe_to_Matte
    )

    # --------------------------------------------------------
    # 8. Slag Flow
    # --------------------------------------------------------

    fe_slag_decimal = (
        float(
            raw_values["Fe_in_Cl_Slag"]
        )
        / 100.0
    )

    if fe_slag_decimal <= 0:

        return None

    Slag_Flow = (
        Fe_to_Slag
        / fe_slag_decimal
    )

    # --------------------------------------------------------
    # 9. O2 to Sulfur
    # --------------------------------------------------------

    if Sulfur_Feed <= 0:

        return None

    O2_to_Sulfur = (
        (
            furnace_air * 0.21
            - furnace_oxygen * 0.95
        )
        / Sulfur_Feed
    )

    # --------------------------------------------------------
    # 10 MODEL FEATURE VECTOR
    # --------------------------------------------------------

    feature_dict = {

        "Cu_Input":
            Cu_Input,

        "Fe_Input":
            Fe_Input,

        "Sulfur_Feed":
            Sulfur_Feed,

        "Matte_Flow":
            Matte_Flow,

        "Slag_Flow":
            Slag_Flow,

        "SILICA FEED RATE ":
            silica_feed,

        "O2_to_Sulfur":
            O2_to_Sulfur,

        "Fe/SiO2":
            fe_sio2,

        "Fe3O4_Cls":
            fe3o4_cls,

        "Matte Grade":
            matte_grade
    }

    return feature_dict


# ============================================================
# 23. CREATE MODEL INPUT
# ============================================================

def feature_dict_to_array(
    feature_dict
):

    values = [
        feature_dict[col]
        for col in FEATURE_ORDER
    ]

    return np.asarray(
        values,
        dtype=float
    ).reshape(
        1,
        -1
    )


# ============================================================
# 24. PREDICTION FUNCTIONS
# ============================================================

def predict_probability(
    feature_dict
):

    X = feature_dict_to_array(
        feature_dict
    )

    probability = (
        model.predict_proba(X)[0]
    )

    return float(
        probability[TARGET_CLASS]
    )


def predict_class(
    feature_dict
):

    X = feature_dict_to_array(
        feature_dict
    )

    return int(
        model.predict(X)[0]
    )


# ============================================================
# 25. CURRENT PROCESS CONDITION
# ============================================================

st.header(
    "1. Current Process Condition"
)

st.info(
    """
    Enter the current process condition.

    These values define the baseline condition from
    which the counterfactual is calculated.
    """
)


# ============================================================
# 26. DEFAULT CURRENT VALUES
# ============================================================

default_values = {}

for col in REQUIRED_RAW_COLUMNS:

    try:

        default_values[col] = float(
            raw_X_train[col].median()
        )

    except Exception:

        default_values[col] = 0.0


# ============================================================
# 27. BLEND COMPOSITION
# ============================================================

st.subheader(
    "Blend Composition — Fixed"
)

col1, col2, col3 = st.columns(3)


with col1:

    current_Cu = st.number_input(
        "Cu (%)",
        value=float(
            default_values["Cu"]
        ),
        format="%.4f"
    )

    current_Fe = st.number_input(
        "Fe (%)",
        value=float(
            default_values["Fe"]
        ),
        format="%.4f"
    )


with col2:

    current_S = st.number_input(
        "S (%)",
        value=float(
            default_values["S"]
        ),
        format="%.4f"
    )

    current_Al2O3 = st.number_input(
        "Al2O3 (%)",
        value=float(
            default_values["Al2O3"]
        ),
        format="%.4f"
    )


with col3:

    current_CaO = st.number_input(
        "CaO (%)",
        value=float(
            default_values["CaO"]
        ),
        format="%.4f"
    )

    current_MgO = st.number_input(
        "MgO (%)",
        value=float(
            default_values["MgO"]
        ),
        format="%.4f"
    )


# ============================================================
# 28. FIXED CONCENTRATE FEED
# ============================================================

st.subheader(
    "Concentrate Feed — Fixed"
)

current_conc_feed = st.number_input(
    "CONC. FEED RATE",
    value=float(
        default_values[
            "CONC. FEED RATE"
        ]
    ),
    format="%.4f"
)


# ============================================================
# 29. OTHER FIXED PROCESS VARIABLES
# ============================================================

st.subheader(
    "Other Process Inputs"
)

col1, col2 = st.columns(2)


with col1:

    current_S_in_Matte = st.number_input(
        "S_in_Matte",
        value=float(
            default_values[
                "S_in_Matte"
            ]
        ),
        format="%.4f"
    )

    current_Fe_in_Cl_Slag = st.number_input(
        "Fe_in_Cl_Slag (%)",
        value=float(
            default_values[
                "Fe_in_Cl_Slag"
            ]
        ),
        format="%.4f"
    )


with col2:

    st.write(
        "The following variables remain FIXED "
        "during counterfactual optimization:"
    )

    st.write(
        "- Fe/SiO₂"
    )

    st.write(
        "- Fe₃O₄ in CL slag"
    )

    st.write(
        "- Matte Grade"
    )


# ============================================================
# 30. CURRENT FURNACE OPERATING CONDITION
# ============================================================

st.subheader(
    "Current Furnace Operating Condition"
)

col1, col2 = st.columns(2)


with col1:

    current_cslag = st.number_input(
        "C-Slag Feed Rate",
        value=float(
            default_values[
                "C-SLAG FEED RATE - S Furnace"
            ]
        ),
        format="%.4f"
    )

    current_silica = st.number_input(
        "Silica Feed Rate",
        value=float(
            default_values[
                "SILICA FEED RATE "
            ]
        ),
        format="%.4f"
    )


with col2:

    current_air = st.number_input(
        "S-Furnace Air",
        value=float(
            default_values[
                "S-FURNACE AIR"
            ]
        ),
        format="%.4f"
    )

    current_oxygen = st.number_input(
        "S-Furnace Oxygen",
        value=float(
            default_values[
                "S-FURNACE OXYGEN"
            ]
        ),
        format="%.4f"
    )


# ============================================================
# 31. FIXED ENGINEERED VARIABLES
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    current_fe_sio2 = st.number_input(
        "Current Fe/SiO₂",
        value=float(
            default_values["Fe/SiO2"]
        ),
        format="%.4f"
    )


with col2:

    current_fe3o4 = st.number_input(
        "Current Fe₃O₄ (%)",
        value=float(
            default_values["Fe3O4_Cls"]
        ),
        format="%.4f"
    )


with col3:

    current_matte_grade = st.number_input(
        "Current Matte Grade (%)",
        value=float(
            default_values["Matte Grade"]
        ),
        format="%.4f"
    )


# ============================================================
# 32. CURRENT RAW VALUE DICTIONARY
# ============================================================

current_raw_values = {

    "Cu":
        current_Cu,

    "Fe":
        current_Fe,

    "S":
        current_S,

    "Al2O3":
        current_Al2O3,

    "CaO":
        current_CaO,

    "MgO":
        current_MgO,

    "CONC. FEED RATE":
        current_conc_feed,

    "S_in_Matte":
        current_S_in_Matte,

    "Fe_in_Cl_Slag":
        current_Fe_in_Cl_Slag
}


# ============================================================
# 33. CURRENT ENGINEERED FEATURES
# ============================================================

current_features = (
    calculate_engineered_features(

        current_raw_values,

        current_cslag,

        current_silica,

        current_air,

        current_oxygen,

        current_fe_sio2,

        current_fe3o4,

        current_matte_grade
    )
)


if current_features is None:

    st.error(
        "Unable to calculate current "
        "engineered features. Please check "
        "the entered process values."
    )

    st.stop()


# ============================================================
# 34. CURRENT PREDICTION
# ============================================================

current_probability = (
    predict_probability(
        current_features
    )
)

current_class = (
    predict_class(
        current_features
    )
)


# ============================================================
# 35. CURRENT MODEL RESULT
# ============================================================

st.header(
    "2. Current XGBoost Prediction"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Predicted Class",
        current_class
    )


with col2:

    st.metric(
        "P(Lower Cu-Loss Class)",
        f"{current_probability:.2%}"
    )


with col3:

    if current_class == TARGET_CLASS:

        st.success(
            "Current condition classified "
            "as LOWER Cu-loss"
        )

    else:

        st.warning(
            "Current condition classified "
            "as HIGHER Cu-loss"
        )


# ============================================================
# 36. CURRENT ENGINEERED FEATURES
# ============================================================

st.subheader(
    "Current Engineered Features"
)

current_feature_table = pd.DataFrame({

    "Feature":
        FEATURE_ORDER,

    "Current Value": [
        current_features[x]
        for x in FEATURE_ORDER
    ]
})

st.dataframe(
    current_feature_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 37. DE SEARCH CONTROL VARIABLES
# ============================================================

CONTROL_VARIABLES = [

    "C-SLAG FEED RATE - S Furnace",

    "SILICA FEED RATE ",

    "S-FURNACE AIR",

    "S-FURNACE OXYGEN"
]


# ============================================================
# 38. HISTORICAL BOUNDS
# ============================================================

def get_bounds(
    column
):

    series = pd.to_numeric(
        raw_X_train[column],
        errors="coerce"
    ).dropna()

    if len(series) == 0:

        raise ValueError(
            f"No valid values found for {column}"
        )

    return (
        float(series.min()),
        float(series.max())
    )


bounds = {}

for variable in CONTROL_VARIABLES:

    bounds[variable] = (
        get_bounds(variable)
    )


# ============================================================
# 39. DISPLAY DE SEARCH BOUNDS
# ============================================================

st.header(
    "3. Differential Evolution Counterfactual Optimization"
)

st.subheader(
    "DE Search Variables"
)

bounds_table = pd.DataFrame({

    "Variable":
        CONTROL_VARIABLES,

    "Current Value": [

        current_cslag,

        current_silica,

        current_air,

        current_oxygen
    ],

    "Historical Minimum": [

        bounds[x][0]
        for x in CONTROL_VARIABLES
    ],

    "Historical Maximum": [

        bounds[x][1]
        for x in CONTROL_VARIABLES
    ]
})

st.dataframe(
    bounds_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 40. NORMALIZED OPERATING CHANGE
# ============================================================

def calculate_change_score(
    candidate,
    current_values
):

    total_change = 0.0

    for i, variable in enumerate(
        CONTROL_VARIABLES
    ):

        low, high = (
            bounds[variable]
        )

        scale = max(
            high - low,
            1e-12
        )

        normalized_change = (
            (
                candidate[i]
                -
                current_values[i]
            )
            /
            scale
        )

        total_change += (
            normalized_change ** 2
        )

    return float(
        total_change
    )


# ============================================================
# 41. CURRENT CONTROL VALUES
# ============================================================

CURRENT_CONTROL_VALUES = np.array(

    [
        current_cslag,
        current_silica,
        current_air,
        current_oxygen
    ],

    dtype=float
)


# ============================================================
# 42. DE OBJECTIVE FUNCTION
# ============================================================

def make_de_objective():

    def objective(x):

        cslag = float(
            x[0]
        )

        silica = float(
            x[1]
        )

        air = float(
            x[2]
        )

        oxygen = float(
            x[3]
        )

        # ----------------------------------------------------
        # Existing engineering calculation
        # ----------------------------------------------------

        features = (
            calculate_engineered_features(

                current_raw_values,

                cslag,

                silica,

                air,

                oxygen,

                current_fe_sio2,

                current_fe3o4,

                current_matte_grade
            )
        )

        if features is None:

            return TARGET_PENALTY

        # ----------------------------------------------------
        # Existing XGBoost prediction
        # ----------------------------------------------------

        probability = (
            predict_probability(
                features
            )
        )

        # ----------------------------------------------------
        # Operating movement penalty
        # ----------------------------------------------------

        change_score = (
            calculate_change_score(

                x,

                CURRENT_CONTROL_VALUES
            )
        )

        # ----------------------------------------------------
        # Probability target penalty
        # ----------------------------------------------------

        probability_gap = max(

            0.0,

            TARGET_PROBABILITY
            -
            probability
        )

        probability_penalty = (

            TARGET_PENALTY
            *
            probability_gap ** 2
        )

        # ----------------------------------------------------
        # Final DE objective
        # ----------------------------------------------------

        objective_value = (

            probability_penalty

            +

            CHANGE_WEIGHT
            *
            change_score
        )

        return float(
            objective_value
        )

    return objective


# ============================================================
# 43. RUN ONE DE CONFIGURATION / ONE SEED
# ============================================================

def run_single_de(
    config,
    seed
):

    objective = (
        make_de_objective()
    )

    de_bounds = [

        bounds[
            CONTROL_VARIABLES[0]
        ],

        bounds[
            CONTROL_VARIABLES[1]
        ],

        bounds[
            CONTROL_VARIABLES[2]
        ],

        bounds[
            CONTROL_VARIABLES[3]
        ]
    ]

    try:

        result = differential_evolution(

            objective,

            de_bounds,

            strategy=config[
                "strategy"
            ],

            maxiter=config[
                "maxiter"
            ],

            popsize=config[
                "popsize"
            ],

            mutation=config[
                "mutation"
            ],

            recombination=config[
                "recombination"
            ],

            tol=config[
                "tol"
            ],

            polish=config[
                "polish"
            ],

            seed=seed,

            updating="immediate",

            workers=1
        )

    except Exception as e:

        return {

            "success":
                False,

            "seed":
                seed,

            "candidate":
                np.array(
                    CURRENT_CONTROL_VALUES,
                    dtype=float
                ),

            "probability":
                0.0,

            "change_score":
                np.inf,

            "objective":
                np.inf,

            "de_success":
                False,

            "iterations":
                0,

            "function_evaluations":
                0,

            "error":
                str(e)
        }


    candidate = np.asarray(
        result.x,
        dtype=float
    )


    # ========================================================
    # Evaluate final DE candidate
    # ========================================================

    features = (
        calculate_engineered_features(

            current_raw_values,

            candidate[0],

            candidate[1],

            candidate[2],

            candidate[3],

            current_fe_sio2,

            current_fe3o4,

            current_matte_grade
        )
    )


    if features is None:

        return {

            "success":
                False,

            "seed":
                seed,

            "candidate":
                candidate,

            "probability":
                0.0,

            "change_score":
                np.inf,

            "objective":
                np.inf,

            "de_success":
                result.success,

            "iterations":
                result.nit,

            "function_evaluations":
                result.nfev,

            "error":
                "Invalid engineered features"
        }


    probability = (
        predict_probability(
            features
        )
    )


    change_score = (
        calculate_change_score(

            candidate,

            CURRENT_CONTROL_VALUES
        )
    )


    feasible = (
        probability
        >=
        TARGET_PROBABILITY
    )


    return {

        "success":
            feasible,

        "seed":
            seed,

        "candidate":
            candidate,

        "probability":
            probability,

        "change_score":
            change_score,

        "objective":
            result.fun,

        "de_success":
            result.success,

        "iterations":
            result.nit,

        "function_evaluations":
            result.nfev,

        "error":
            ""
    }


# ============================================================
# 44. RUN DE SCREENING
# ============================================================

def run_de_screening():

    all_results = []

    total_runs = (

        len(DE_CONFIGURATIONS)

        *

        len(SCREENING_SEEDS)
    )

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()

    completed = 0


    for config in DE_CONFIGURATIONS:

        for seed in SCREENING_SEEDS:

            status_text.write(

                f"Testing {config['name']} "
                f"| Seed {seed} "
                f"| Run {completed + 1}/{total_runs}"
            )


            result = run_single_de(

                config,

                seed
            )


            result[
                "configuration"
            ] = config["name"]

            result[
                "strategy"
            ] = config["strategy"]

            result[
                "popsize"
            ] = config["popsize"]

            result[
                "maxiter"
            ] = config["maxiter"]


            all_results.append(
                result
            )


            completed += 1


            progress_bar.progress(

                completed
                /
                total_runs
            )


    status_text.write(
        "DE configuration screening completed."
    )

    progress_bar.empty()

    return all_results


# ============================================================
# 45. SUMMARIZE DE SCREENING
# ============================================================

def summarize_de_results(
    all_results
):

    summary = []


    for config in DE_CONFIGURATIONS:

        config_results = [

            r

            for r in all_results

            if r["configuration"]
            ==
            config["name"]
        ]


        probabilities = np.array([

            r["probability"]

            for r in config_results
        ])


        feasible_results = [

            r

            for r in config_results

            if r["success"]
        ]


        success_count = (
            len(
                feasible_results
            )
        )


        success_rate = (

            success_count

            /

            len(
                config_results
            )
        )


        if feasible_results:

            feasible_probabilities = np.array([

                r["probability"]

                for r in feasible_results
            ])


            feasible_changes = np.array([

                r["change_score"]

                for r in feasible_results
            ])


            best_probability = float(
                np.max(
                    feasible_probabilities
                )
            )


            mean_feasible_probability = float(

                np.mean(
                    feasible_probabilities
                )
            )


            mean_feasible_change = float(

                np.mean(
                    feasible_changes
                )
            )


            best_change = float(

                np.min(
                    feasible_changes
                )
            )

        else:

            best_probability = float(

                np.max(
                    probabilities
                )
            )

            mean_feasible_probability = 0.0

            mean_feasible_change = np.inf

            best_change = np.inf


        summary.append({

            "Configuration":
                config["name"],

            "Strategy":
                config["strategy"],

            "Population":
                config["popsize"],

            "Max Iterations":
                config["maxiter"],

            "Best P(Class 1)":
                best_probability,

            "Mean P(Class 1)":
                float(
                    np.mean(
                        probabilities
                    )
                ),

            "Std P(Class 1)":
                float(
                    np.std(
                        probabilities
                    )
                ),

            "Minimum P(Class 1)":
                float(
                    np.min(
                        probabilities
                    )
                ),

            "Success Rate":
                success_rate,

            "Successful Runs":
                f"{success_count}/"
                f"{len(config_results)}",

            "Mean Change Score":
                mean_feasible_change,

            "Best Change Score":
                best_change
        })


    return pd.DataFrame(
        summary
    )


# ============================================================
# 46. RANK DE CONFIGURATIONS
# ============================================================

def rank_de_configurations(
    summary_df
):

    ranked = (
        summary_df.copy()
    )


    # --------------------------------------------------------
    # Ranking philosophy
    #
    # 1. Highest success rate
    # 2. Highest mean probability
    # 3. Lowest probability variation
    # 4. Lowest operating movement
    # --------------------------------------------------------

    ranked["_success"] = (
        ranked[
            "Success Rate"
        ]
    )


    ranked["_mean_probability"] = (
        ranked[
            "Mean P(Class 1)"
        ]
    )


    ranked["_negative_std"] = (

        -
        ranked[
            "Std P(Class 1)"
        ]
    )


    ranked["_negative_change"] = (

        -
        ranked[
            "Mean Change Score"
        ]
    )


    ranked = ranked.sort_values(

        by=[

            "_success",

            "_mean_probability",

            "_negative_std",

            "_negative_change"
        ],

        ascending=False
    )


    ranked = ranked.drop(

        columns=[

            "_success",

            "_mean_probability",

            "_negative_std",

            "_negative_change"
        ]
    )


    ranked.insert(

        0,

        "Rank",

        range(
            1,
            len(ranked) + 1
        )
    )


    return ranked.reset_index(
        drop=True
    )


# ============================================================
# 47. FINAL MULTI-SEED OPTIMIZATION
# ============================================================

def run_final_multi_seed(
    selected_config
):

    final_results = []

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()

    total = len(
        FINAL_SEEDS
    )


    for i, seed in enumerate(
        FINAL_SEEDS
    ):

        status_text.write(

            f"Final optimization "
            f"| Seed {seed} "
            f"| Run {i + 1}/{total}"
        )


        result = run_single_de(

            selected_config,

            seed
        )


        final_results.append(
            result
        )


        progress_bar.progress(

            (i + 1)
            /
            total
        )


    status_text.write(
        "Final multi-seed optimization completed."
    )

    progress_bar.empty()

    return final_results


# ============================================================
# 48. SELECT FINAL COUNTERFACTUAL
# ============================================================

def select_final_counterfactual(
    final_results
):

    feasible = [

        r

        for r in final_results

        if r["success"]
    ]


    if not feasible:

        return None


    # --------------------------------------------------------
    # Find maximum probability among successful solutions
    # --------------------------------------------------------

    max_probability = max(

        r["probability"]

        for r in feasible
    )


    # --------------------------------------------------------
    # Keep solutions within 0.5 percentage points
    # of the best probability.
    #
    # Among these near-best solutions, select the one
    # requiring the smallest operating movement.
    # --------------------------------------------------------

    probability_threshold = (

        max_probability
        -
        0.005
    )


    near_best = [

        r

        for r in feasible

        if r["probability"]
        >=
        probability_threshold
    ]


    selected = min(

        near_best,

        key=lambda r:
            r["change_score"]
    )


    return selected


# ============================================================
# 49. DE TUNING INTERFACE
# ============================================================

st.subheader(
    "Differential Evolution Hyperparameter Tuning"
)

st.info(

    f"""
    The application will test
    {len(DE_CONFIGURATIONS)}
    DE configurations.

    Each configuration will be tested using
    {len(SCREENING_SEEDS)}
    independent random seeds.

    This gives:

    {len(DE_CONFIGURATIONS)}
    ×
    {len(SCREENING_SEEDS)}
    =
    {len(DE_CONFIGURATIONS) * len(SCREENING_SEEDS)}
    DE screening runs.

    Only these four furnace controls are optimized:

    • C-Slag Feed Rate
    • Silica Feed Rate
    • S-Furnace Air
    • S-Furnace Oxygen

    All predictive/model parameters remain unchanged.
    """
)


# ============================================================
# 50. SESSION STATE
# ============================================================

if (
    "de_screening_results"
    not in st.session_state
):

    st.session_state.de_screening_results = None


if (
    "de_summary"
    not in st.session_state
):

    st.session_state.de_summary = None


if (
    "selected_de_config"
    not in st.session_state
):

    st.session_state.selected_de_config = None


if (
    "final_de_results"
    not in st.session_state
):

    st.session_state.final_de_results = None


if (
    "final_counterfactual"
    not in st.session_state
):

    st.session_state.final_counterfactual = None


# ============================================================
# 51. SCREENING BUTTON
# ============================================================

run_screening = st.button(

    "Run DE Hyperparameter Screening",

    type="primary"
)


if run_screening:

    with st.spinner(

        "Testing DE configurations "
        "across multiple seeds..."
    ):

        screening_results = (
            run_de_screening()
        )


        summary = (
            summarize_de_results(
                screening_results
            )
        )


        ranked_summary = (
            rank_de_configurations(
                summary
            )
        )


        st.session_state.de_screening_results = (
            screening_results
        )


        st.session_state.de_summary = (
            ranked_summary
        )


        # ----------------------------------------------------
        # Automatically select Rank-1 configuration
        # ----------------------------------------------------

        best_configuration_name = (

            ranked_summary.iloc[0][
                "Configuration"
            ]
        )


        selected_config = next(

            config

            for config in DE_CONFIGURATIONS

            if config["name"]
            ==
            best_configuration_name
        )


        st.session_state.selected_de_config = (
            selected_config
        )


        # Clear any previous final result
        st.session_state.final_de_results = None

        st.session_state.final_counterfactual = None


# ============================================================
# 52. DISPLAY SCREENING RESULTS
# ============================================================

if (
    st.session_state.de_summary
    is not None
):

    st.subheader(
        "DE Configuration Screening Results"
    )


    display_summary = (
        st.session_state.de_summary.copy()
    )


    probability_columns = [

        "Best P(Class 1)",

        "Mean P(Class 1)",

        "Std P(Class 1)",

        "Minimum P(Class 1)"
    ]


    for column in probability_columns:

        display_summary[column] = (

            display_summary[column]

            .map(
                lambda x:
                f"{x:.2%}"
            )
        )


    display_summary[
        "Success Rate"
    ] = (

        display_summary[
            "Success Rate"
        ]

        .map(
            lambda x:
            f"{x:.1%}"
        )
    )


    st.dataframe(

        display_summary,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# 53. SELECTED DE CONFIGURATION
# ============================================================

if (
    st.session_state.selected_de_config
    is not None
):

    selected = (
        st.session_state.selected_de_config
    )


    st.subheader(
        "Selected Robust DE Configuration"
    )


    config_display = pd.DataFrame({

        "Parameter": [

            "Strategy",

            "Population Size",

            "Maximum Iterations",

            "Mutation",

            "Recombination",

            "Tolerance",

            "Polishing",

            "Screening Seeds"
        ],


        "Selected Value": [

            selected["strategy"],

            selected["popsize"],

            selected["maxiter"],

            str(
                selected["mutation"]
            ),

            selected[
                "recombination"
            ],

            selected["tol"],

            selected["polish"],

            len(
                SCREENING_SEEDS
            )
        ]
    })


    st.dataframe(

        config_display,

        use_container_width=True,

        hide_index=True
    )


    st.success(

        "The most robust DE configuration "
        "has been selected based on "
        "multi-seed performance."
    )


# ============================================================
# 54. FINAL 50-SEED OPTIMIZATION BUTTON
# ============================================================

if (
    st.session_state.selected_de_config
    is not None
):

    run_final = st.button(

        "Run Final 50-Seed Counterfactual Optimization",

        type="primary"
    )


    if run_final:

        with st.spinner(

            "Running the selected DE configuration "
            "across 50 independent seeds..."
        ):

            final_results = (
                run_final_multi_seed(

                    st.session_state
                    .selected_de_config
                )
            )


            final_counterfactual = (
                select_final_counterfactual(

                    final_results
                )
            )


            st.session_state.final_de_results = (
                final_results
            )


            st.session_state.final_counterfactual = (
                final_counterfactual
            )


# ============================================================
# 55. FINAL MULTI-SEED STABILITY RESULTS
# ============================================================

if (
    st.session_state.final_de_results
    is not None
):

    final_results = (
        st.session_state.final_de_results
    )


    final_probabilities = np.array([

        r["probability"]

        for r in final_results
    ])


    successful_results = [

        r

        for r in final_results

        if r["success"]
    ]


    st.subheader(
        "Final Multi-Seed Stability Analysis"
    )


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    with metric1:

        st.metric(

            "Best P(Class 1)",

            f"{np.max(final_probabilities):.2%}"
        )


    with metric2:

        st.metric(

            "Mean P(Class 1)",

            f"{np.mean(final_probabilities):.2%}"
        )


    with metric3:

        st.metric(

            "Std. Deviation",

            f"{np.std(final_probabilities):.2%}"
        )


    with metric4:

        st.metric(

            "Runs ≥ Target",

            f"{len(successful_results)}"
            f"/{len(final_results)}"
        )


    # --------------------------------------------------------
    # Final seed results
    # --------------------------------------------------------

    final_table = []


    for r in final_results:

        final_table.append({

            "Seed":
                r["seed"],

            "P(Class 1)":
                r["probability"],

            "Target Reached":
                "YES"
                if r["success"]
                else "NO",

            "Change Score":
                r["change_score"],

            "DE Converged":
                r["de_success"],

            "Iterations":
                r["iterations"],

            "Function Evaluations":
                r["function_evaluations"]
        })


    final_table = pd.DataFrame(
        final_table
    )


    final_table[
        "P(Class 1)"
    ] = (

        final_table[
            "P(Class 1)"
        ]

        .map(
            lambda x:
            f"{x:.2%}"
        )
    )


    st.dataframe(

        final_table,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# 56. FINAL COUNTERFACTUAL
# ============================================================

if (
    st.session_state.final_counterfactual
    is not None
):

    selected_cf = (
        st.session_state.final_counterfactual
    )


    candidate = (
        selected_cf["candidate"]
    )


    st.header(
        "4. Recommended Counterfactual"
    )


    st.success(

        f"""
        Final recommended counterfactual was
        selected from the robust DE configuration
        using the independent 50-seed optimization.

        Predicted probability of Lower Cu-loss class:

        {selected_cf["probability"]:.2%}
        """
    )


    # ========================================================
    # Recommended operating values
    # ========================================================

    recommendation_table = pd.DataFrame({

        "Operating Variable":
            CONTROL_VARIABLES,

        "Current Value": [

            current_cslag,

            current_silica,

            current_air,

            current_oxygen
        ],

        "Recommended Value": [

            candidate[0],

            candidate[1],

            candidate[2],

            candidate[3]
        ],

        "Change": [

            candidate[0]
            -
            current_cslag,

            candidate[1]
            -
            current_silica,

            candidate[2]
            -
            current_air,

            candidate[3]
            -
            current_oxygen
        ]
    })


    recommendation_table[
        "% Change"
    ] = (

        recommendation_table[
            "Change"
        ]

        /

        recommendation_table[
            "Current Value"
        ].replace(
            0,
            np.nan
        )

        *

        100
    )


    st.subheader(
        "Recommended Furnace Operating Conditions"
    )


    st.dataframe(

        recommendation_table,

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # FINAL ENGINEERED FEATURES
    # ========================================================

    final_features = (
        calculate_engineered_features(

            current_raw_values,

            candidate[0],

            candidate[1],

            candidate[2],

            candidate[3],

            current_fe_sio2,

            current_fe3o4,

            current_matte_grade
        )
    )


    final_probability = (
        predict_probability(
            final_features
        )
    )


    final_class = (
        predict_class(
            final_features
        )
    )


    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    st.subheader(
        "Final Counterfactual XGBoost Prediction"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(

            "Predicted Class",

            final_class
        )


    with c2:

        st.metric(

            "P(Lower Cu-Loss)",

            f"{final_probability:.2%}"
        )


    with c3:

        probability_improvement = (

            final_probability

            -

            current_probability
        )


        st.metric(

            "Probability Improvement",

            f"{probability_improvement:+.2%}"
        )


    # ========================================================
    # FINAL ENGINEERED FEATURES
    # ========================================================

    st.subheader(
        "Final Counterfactual Engineered Features"
    )


    final_feature_table = pd.DataFrame({

        "Feature":
            FEATURE_ORDER,

        "Current Value": [

            current_features[x]

            for x in FEATURE_ORDER
        ],

        "Counterfactual Value": [

            final_features[x]

            for x in FEATURE_ORDER
        ]
    })


    final_feature_table[
        "Change"
    ] = (

        final_feature_table[
            "Counterfactual Value"
        ]

        -

        final_feature_table[
            "Current Value"
        ]
    )


    st.dataframe(

        final_feature_table,

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # FIXED VARIABLES CONFIRMATION
    # ========================================================

    st.subheader(
        "Counterfactual Constraints"
    )


    constraint_table = pd.DataFrame({

        "Variable": [

            "Cu",

            "Fe",

            "S",

            "Al2O3",

            "CaO",

            "MgO",

            "CONC. FEED RATE",

            "S_in_Matte",

            "Fe_in_Cl_Slag",

            "Fe/SiO2",

            "Fe3O4_Cls",

            "Matte Grade"
        ],

        "Status": [

            "FIXED",

            "FIXED",

            "FIXED",

            "FIXED",

            "FIXED",

            "FIXED",

            "FIXED",

            "FIXED",

            "FIXED",

            "FIXED",

            "FIXED",

            "FIXED"
        ]
    })


    st.dataframe(

        constraint_table,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# 57. RESET BUTTON
# ============================================================

st.divider()

if st.button(
    "Reset DE Optimization"
):

    st.session_state.de_screening_results = None

    st.session_state.de_summary = None

    st.session_state.selected_de_config = None

    st.session_state.final_de_results = None

    st.session_state.final_counterfactual = None

    st.rerun()
