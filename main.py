import argparse
import pandas as pd
import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    r2_score, mean_absolute_error
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ---------------- CLI ----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()

# -------- Target detection --------
def detect_target_column(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    non_null_cols = [c for c in df.columns if df[c].isna().mean() == 0]
    return non_null_cols[-1]

# -------- Preprocessing --------
def preprocess(train_df, test_df, target_col):
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    combined = pd.concat([X, test_df], axis=0)

    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

        if combined[col].isna().all():
            combined[col] = combined[col].astype(str).fillna("missing")
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col])
        else:
            combined[col] = combined[col].fillna(combined[col].median())

    return combined.iloc[:len(X)], y, combined.iloc[len(X):]

# -------- Yield category --------
def yield_category(value, p33, p66):
    if value <= p33:
        return "LOW"
    elif value <= p66:
        return "MEDIUM"
    else:
        return "HIGH"

# -------- Classification mapping --------
def class_to_category(label):
    mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return mapping.get(label, "MEDIUM")

# -------- Farmer explanation (3 languages) --------
def explain(category, lang):
    messages = {
        "en": {
            "LOW": (
                "⚠️ Expected crop yield is LOW.\n"
                "Current soil or weather conditions are not suitable.\n\n"
                "👉 What you can do:\n"
                "• Improve irrigation\n"
                "• Add organic manure or fertilizer\n"
                "• Consider switching to a suitable crop"
            ),
            "MEDIUM": (
                "✅ Expected crop yield is MEDIUM.\n"
                "Crop growth is acceptable, but improvement is possible.\n\n"
                "👉 What you can do:\n"
                "• Maintain regular irrigation\n"
                "• Monitor pests and diseases\n"
                "• Use balanced fertilizer"
            ),
            "HIGH": (
                "🌟 Expected crop yield is HIGH.\n"
                "Soil and weather conditions are very favorable.\n\n"
                "👉 What you can do:\n"
                "• Continue current farming practices\n"
                "• Monitor crop health\n"
                "• Harvest at the right time"
            )
        },

        "hi": {
            "LOW": (
                "⚠️ फसल की पैदावार कम रहने की संभावना है।\n"
                "मिट्टी या मौसम अनुकूल नहीं है।\n\n"
                "👉 क्या करें:\n"
                "• सिंचाई सुधारें\n"
                "• जैविक खाद डालें\n"
                "• वैकल्पिक फसल पर विचार करें"
            ),
            "MEDIUM": (
                "✅ फसल की पैदावार मध्यम रहने की संभावना है।\n"
                "फसल ठीक है, लेकिन सुधार संभव है।\n\n"
                "👉 क्या करें:\n"
                "• नियमित सिंचाई रखें\n"
                "• कीट और रोगों पर ध्यान दें\n"
                "• संतुलित खाद का प्रयोग करें"
            ),
            "HIGH": (
                "🌟 फसल की पैदावार अच्छी रहने की संभावना है।\n"
                "मिट्टी और मौसम अनुकूल हैं।\n\n"
                "👉 क्या करें:\n"
                "• वर्तमान खेती जारी रखें\n"
                "• फसल की निगरानी करें\n"
                "• सही समय पर कटाई करें"
            )
        },

        "kn": {
            "LOW": (
                "⚠️ ನಿಮ್ಮ ಬೆಳೆ ಇಳುವರಿ ಕಡಿಮೆ ಇರಬಹುದು.\n"
                "ಮಣ್ಣು ಅಥವಾ ಹವಾಮಾನ ಅನುಕೂಲಕರವಲ್ಲ.\n\n"
                "👉 ನೀವು ಮಾಡಬೇಕಾದದ್ದು:\n"
                "• ನೀರಾವರಿ ಸುಧಾರಿಸಿ\n"
                "• ಜೈವಿಕ ಗೊಬ್ಬರ ಬಳಸಿ\n"
                "• ಬೇರೆ ಬೆಳೆ ಯೋಚಿಸಿ"
            ),
            "MEDIUM": (
                "✅ ನಿಮ್ಮ ಬೆಳೆ ಇಳುವರಿ ಮಧ್ಯಮವಾಗಿರಬಹುದು.\n"
                "ಬೆಳೆ ಚೆನ್ನಾಗಿದೆ, ಆದರೆ ಇನ್ನೂ ಸುಧಾರಿಸಬಹುದು.\n\n"
                "👉 ನೀವು ಮಾಡಬೇಕಾದದ್ದು:\n"
                "• ಸಮಯಕ್ಕೆ ಸರಿಯಾಗಿ ನೀರು ನೀಡಿ\n"
                "• ಕೀಟ ಮತ್ತು ರೋಗಗಳನ್ನು ಗಮನಿಸಿ\n"
                "• ಸಮತೋಲನ ಗೊಬ್ಬರ ಬಳಸಿ"
            ),
            "HIGH": (
                "🌟 ನಿಮ್ಮ ಬೆಳೆ ಇಳುವರಿ ಹೆಚ್ಚು ಇರಬಹುದು.\n"
                "ಮಣ್ಣು ಮತ್ತು ಹವಾಮಾನ ಅತ್ಯುತ್ತಮವಾಗಿದೆ.\n\n"
                "👉 ನೀವು ಮಾಡಬೇಕಾದದ್ದು:\n"
                "• ಈಗಿನ ಕೃಷಿ ವಿಧಾನ ಮುಂದುವರಿಸಿ\n"
                "• ಬೆಳೆ ಆರೋಗ್ಯ ಗಮನಿಸಿ\n"
                "• ಸರಿಯಾದ ಸಮಯದಲ್ಲಿ ಕೊಯ್ಲು ಮಾಡಿ"
            )
        }
    }
    return messages[lang][category]

# ---------------- MAIN ----------------
def main():
    args = parse_args()

    print("🌾 AgriTech AI — Universal Model Mode")

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
    test_df  = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    target_col = detect_target_column(train_df)
    print(f"Detected target column: {target_col}")

    train_df = train_df.dropna(subset=[target_col])

    X, y, X_test = preprocess(train_df, test_df, target_col)

    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
    data_size = len(train_df)

    # -------- REGRESSION --------
    if is_regression:
        print("🧠 Task detected: REGRESSION")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        model = (
            HistGradientBoostingRegressor(max_depth=10, learning_rate=0.08, max_iter=300, random_state=SEED)
            if data_size > 200_000 else
            RandomForestRegressor(n_estimators=150, max_depth=18, random_state=SEED, n_jobs=1)
        )

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)

        print(f"R2 Score: {r2_score(y_val, val_preds):.4f}")
        print(f"MAE: {mean_absolute_error(y_val, val_preds):.4f}")

        test_preds = np.round(model.predict(X_test), 2)

        p33, p66 = np.percentile(test_preds, [33, 66])
        avg_pred = np.mean(test_preds)
        category = yield_category(avg_pred, p33, p66)

    # -------- CLASSIFICATION --------
    else:
        print("🧠 Task detected: CLASSIFICATION")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED,
            stratify=y if y.nunique() > 1 else None
        )

        model = (
            HistGradientBoostingClassifier(max_depth=10, learning_rate=0.08, max_iter=300, random_state=SEED)
            if data_size > 200_000 else
            RandomForestClassifier(n_estimators=150, max_depth=18, random_state=SEED, n_jobs=1)
        )

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)

        print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
        print(f"F1 Score: {f1_score(y_val, val_preds, average='weighted'):.4f}")

        test_preds = model.predict(X_test)

        # ---------- Farmer advisory logic ----------
        most_common_class = pd.Series(test_preds).mode()[0]
        freq = pd.Series(test_preds).value_counts(normalize=True)[most_common_class]

        if freq < 0.33:
            category = "LOW"
        elif freq < 0.66:
            category = "MEDIUM"
        else:
            category = "HIGH"

    # -------- Language choice --------
    print("\nChoose language for farmer explanation:")
    print("1. English\n2. Hindi\n3. Kannada")
    choice = input("Enter choice (1/2/3): ").strip()

    lang_map = {"1": "en", "2": "hi", "3": "kn"}
    lang = lang_map.get(choice, "en")

    print("\n🌾 Farmer Explanation:")
    print("----------------------------------")
    print(explain(category, lang))
    print("----------------------------------")

    # -------- Save output --------
    output = pd.DataFrame({
        "id": test_df["id"] if "id" in test_df.columns else np.arange(len(test_preds)),
        "prediction": test_preds
    })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output.to_csv(args.out, index=False)

    print(f"\n✅ predictions.csv generated at {args.out}")

if __name__ == "__main__":
    main()
