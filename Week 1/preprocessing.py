
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# STEP 1: SET UP FILE PATHS

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student-mat.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# STEP 2: LOAD AND INSPECT THE ORIGINAL DATASET

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place student-mat.csv beside this Python script."
    )

# The UCI student-mat.csv dataset uses semicolons as separators.
df = pd.read_csv(DATA_FILE, sep=";")

print("FIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nSUMMARY STATISTICS")
print(df.describe(include="all"))


# STEP 3: PREPROCESS THE DATA

# Work on a copy so the original DataFrame remains unchanged.
cleaned_df = df.copy()

print(
    "\nMissing values before cleaning:",
    cleaned_df.isnull().sum().sum()
)

# Handle missing values safely.
for col in cleaned_df.columns:
    if not cleaned_df[col].isnull().any():
        continue

    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
        median_value = cleaned_df[col].median()

        # An entirely empty numerical column has no median.
        if pd.isna(median_value):
            raise ValueError(
                f"Column '{col}' contains only missing values. "
                "Inspect the source data before deciding how to fill it."
            )

        cleaned_df[col] = cleaned_df[col].fillna(median_value)

    else:
        mode_values = cleaned_df[col].mode(dropna=True)

        # Use the most common value, or "Unknown" if none exists.
        fill_value = (
            mode_values.iloc[0] if not mode_values.empty else "Unknown"
        )
        cleaned_df[col] = cleaned_df[col].fillna(fill_value)

# Remove exact duplicate rows.
cleaned_df = cleaned_df.drop_duplicates().copy()

print(
    "Missing values after cleaning:",
    cleaned_df.isnull().sum().sum()
)
print(
    "Duplicate rows after cleaning:",
    cleaned_df.duplicated().sum()
)


# STEP 4: ENCODE CATEGORICAL COLUMNS

categorical_cols = cleaned_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

encoded_df = pd.get_dummies(
    cleaned_df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

print("\nDataset shape after encoding:", encoded_df.shape)
print("Categorical columns encoded:", len(categorical_cols))


# STEP 5: NORMALIZE NUMERICAL FEATURES

# Keep the original grade columns on their original scale.
grade_cols = ["G1", "G2", "G3"]

# Only scale columns that are numerical and not final-grade columns.
numeric_cols = encoded_df.select_dtypes(
    include="number"
).columns.tolist()

features_to_scale = [
    col for col in numeric_cols if col not in grade_cols
]

# MinMaxScaler maps values to a range from 0 to 1.
if features_to_scale:
    scaler = MinMaxScaler()

    encoded_df[features_to_scale] = scaler.fit_transform(
        encoded_df[features_to_scale]
    )

print("\nNormalization completed.")
if features_to_scale:
    print(
        encoded_df[features_to_scale]
        .describe()
        .round(2)
    )


# STEP 6: SAVE THE PREPROCESSED DATASET

processed_file = OUTPUT_DIR / "student_performance_preprocessed.csv"

encoded_df.to_csv(processed_file, index=False)

print(f"\nPreprocessed dataset saved to: {processed_file}")


# STEP 7: VISUALIZE THE DISTRIBUTION OF FINAL GRADES

if "G3" not in df.columns:
    raise ValueError("The dataset does not contain the required G3 column.")

plt.figure(figsize=(8, 5))
plt.hist(df["G3"].dropna(), bins=11, edgecolor="black")
plt.title("Distribution of Final Student Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()

grade_plot = OUTPUT_DIR / "final_grade_distribution.png"
plt.savefig(grade_plot, dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# STEP 8: COMPARE AVERAGE GRADE BY STUDY TIME

if "studytime" not in df.columns:
    raise ValueError(
        "The dataset does not contain the required studytime column."
    )

average_grades = df.groupby("studytime")["G3"].mean()

plt.figure(figsize=(8, 5))
average_grades.plot(kind="bar", edgecolor="black")
plt.title("Average Final Grade by Study Time")
plt.xlabel("Weekly Study Time Category")
plt.ylabel("Average Final Grade (G3)")
plt.xticks(rotation=0)
plt.tight_layout()

studytime_plot = OUTPUT_DIR / "average_grade_by_studytime.png"
plt.savefig(studytime_plot, dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# STEP 9: DISPLAY FINAL RESULTS

print("\nFINAL SUMMARY")
print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", cleaned_df.shape)
print("Encoded dataset shape:", encoded_df.shape)
print("Processed CSV:", processed_file)
print("Grade distribution graph:", grade_plot)
print("Study-time graph:", studytime_plot)
print("\nAll processing steps completed successfully!")

