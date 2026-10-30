"""Synthetic clinical outcomes registry generator.

Creates three simulated source-system tables (demographics, treatment,
outcomes), introduces known data quality errors, and loads the result
into a local DuckDB database. All records are synthetic.
"""
import os

import duckdb
import numpy as np
import pandas as pd

SEED = 42
N = 3000
STUDY_END = pd.Timestamp("2025-12-31")
DB_FILE = "data/registry.duckdb"

rng = np.random.default_rng(SEED)

# Cancer types with their proportion of patients and median survival (months)
cancer_types = {
    "Head and neck": (0.30, 90),
    "Paediatric CNS": (0.25, 200),
    "Chordoma": (0.10, 120),
    "Sarcoma": (0.20, 70),
    "Lymphoma": (0.15, 110),
}


def days(values):
    # Convert day counts to timedeltas so they can be added to date columns
    return pd.to_timedelta(pd.Series(values), unit="D")


def make_clean_tables():
    patient_id = [f"P{i:05d}" for i in range(1, N + 1)]

    names = list(cancer_types.keys())
    shares = [cancer_types[c][0] for c in names]
    cancer = rng.choice(names, size=N, p=shares)
    sex = rng.choice(["F", "M"], size=N)

    # Adult ages by default; paediatric CNS patients are aged 2-16
    age = rng.normal(58, 15, N).clip(18, 90).astype(int)
    kids = cancer == "Paediatric CNS"
    age[kids] = rng.integers(2, 17, kids.sum())

    # Diagnosis spans seven years; date of birth is derived from age
    diagnosis = pd.Timestamp("2018-01-01") + days(rng.integers(0, 365 * 7, N))
    dob = diagnosis - days((age * 365.25).astype(int))

    # Waiting time from diagnosis to treatment start (minimum one week)
    wait = (rng.gamma(4, 10, N) + 7).astype(int)
    treat_start = diagnosis + days(wait)

    # Treatment duration based on the number of fractions (5 per week)
    fractions = rng.choice([20, 25, 30, 33], size=N)
    treat_end = treat_start + days(fractions * 7 // 5)

    # Survival time is exponential, shortened with age and capped at 20 years
    # to keep dates within the range pandas can represent
    median = np.array([cancer_types[c][1] for c in cancer])
    scale = median / np.log(2) * np.exp(-0.02 * (age - 50))
    surv_days = (rng.exponential(scale) * 30.44).astype(int)
    surv_days = np.minimum(surv_days, 365 * 20)

    # Death date cannot fall before treatment has finished
    death = treat_start + days(surv_days)
    day_after_tx = treat_end + days([1] * N)
    death = death.where(death > day_after_tx, day_after_tx)
    dead = death <= STUDY_END

    # Patients alive at study end are censored with a last follow-up date
    # falling within the final year
    censor = STUDY_END - days(rng.integers(0, 365, N))
    alive_fu = censor.where(censor > treat_end, treat_end)
    last_fu = death.where(dead, alive_fu)

    # Maximum toxicity grade recorded during follow-up (0 = none, 3 = severe)
    toxicity = rng.choice([0, 1, 2, 3], size=N, p=[0.35, 0.35, 0.22, 0.08])

    demographics = pd.DataFrame({
        "patient_id": patient_id,
        "sex": sex,
        "date_of_birth": dob,
    })
    treatment = pd.DataFrame({
        "course_id": [f"C{i:05d}" for i in range(1, N + 1)],
        "patient_id": patient_id,
        "cancer_type": cancer,
        "diagnosis_date": diagnosis,
        "treatment_start_date": treat_start,
        "treatment_end_date": treat_end,
        "fractions": fractions,
    })
    outcomes = pd.DataFrame({
        "patient_id": patient_id,
        "last_follow_up_date": last_fu,
        "vital_status": np.where(dead, "Dead", "Alive"),
        "death_date": death.where(dead),
        "max_toxicity_grade": toxicity,
    })
    return demographics, treatment, outcomes


def break_the_data(demo, treat, out):
    # Duplicate patient records: 40 exact copies and 20 with a conflicting
    # date of birth
    picked = demo.sample(60, random_state=SEED)
    exact_copies = picked.iloc[20:]
    wrong_dob = picked.iloc[:20].copy()
    wrong_dob["date_of_birth"] += pd.to_timedelta(rng.integers(1, 400, 20), unit="D")
    demo = pd.concat([demo, exact_copies, wrong_dob], ignore_index=True)

    # Treatment start dates set before the diagnosis date (40 records)
    rows = treat.sample(40, random_state=SEED).index
    treat.loc[rows, "treatment_start_date"] = treat.loc[rows, "diagnosis_date"] - pd.to_timedelta(
        rng.integers(1, 31, 40), unit="D"
    )

    # Patients absent from the outcomes system (80 records removed)
    out = out.drop(out.sample(80, random_state=SEED).index)

    # Outcome records with a missing vital status (50 records)
    rows = out.sample(50, random_state=SEED + 1).index
    out.loc[rows, "vital_status"] = None

    # Orphan treatment records referencing patients who do not exist (15)
    orphans = treat.sample(15, random_state=SEED).copy()
    orphans["patient_id"] = [f"P{99000 + i}" for i in range(15)]
    orphans["course_id"] = [f"C{99000 + i}" for i in range(15)]
    treat = pd.concat([treat, orphans], ignore_index=True)

    print("Errors added: 40 exact duplicates, 20 duplicates with wrong DOB, "
          "40 treatments before diagnosis, 80 missing outcomes, "
          "50 missing vital status, 15 orphan treatments")
    return demo, treat, out


def save_tables(tables):
    os.makedirs("data", exist_ok=True)
    con = duckdb.connect(DB_FILE)

    for name, df in tables.items():
        # Columns with "date" in the name are cast to the DATE type
        cols = [f"CAST({c} AS DATE) AS {c}" if "date" in c else c for c in df.columns]
        con.register("df_view", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT {', '.join(cols)} FROM df_view")
        con.unregister("df_view")

        # Report the row count as a basic load check
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchall()[0][0]
        print(f"{name}: {n} rows")

    con.close()


if __name__ == "__main__":
    demo, treat, out = make_clean_tables()
    demo, treat, out = break_the_data(demo, treat, out)
    save_tables({
        "raw_demographics": demo,
        "raw_treatment": treat,
        "raw_outcomes": out,
    })
    print("Saved to", DB_FILE)