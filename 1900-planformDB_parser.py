import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "path/to/planformDB_2.5.0.edb"  # <-- change this

def load_time_series(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        # Experiments per year (Experiment → Publication.Year)
        exp_years = pd.read_sql_query(
            """
            SELECT p.Year AS Year, COUNT(*) AS Experiments
            FROM Experiment e
            JOIN Publication p ON p.Id = e.Publication
            WHERE p.Year IS NOT NULL
            GROUP BY p.Year
            ORDER BY p.Year
            """,
            conn,
        )
        # Publications per year
        pub_years = pd.read_sql_query(
            """
            SELECT Year, COUNT(*) AS Publications
            FROM Publication
            WHERE Year IS NOT NULL
            GROUP BY Year
            ORDER BY Year
            """,
            conn,
        )
        # Distinct morphologies observed per year
        morph_years = pd.read_sql_query(
            """
            SELECT p.Year AS Year, COUNT(DISTINCT rm.Morphology) AS MorphologiesObserved
            FROM ResultantMorphology rm
            JOIN ResultSet rs ON rs.Id = rm.ResultSet
            JOIN Experiment e ON e.Id = rs.Experiment
            JOIN Publication p ON p.Id = e.Publication
            WHERE p.Year IS NOT NULL AND rm.Frequency > 0
            GROUP BY p.Year
            ORDER BY p.Year
            """,
            conn,
        )
    finally:
        conn.close()

    # Merge and fill
    df = (
        exp_years.merge(pub_years, on="Year", how="outer")
                 .merge(morph_years, on="Year", how="outer")
                 .sort_values("Year")
                 .reset_index(drop=True)
    )
    if not df.empty:
        full_years = pd.DataFrame({"Year": range(int(df["Year"].min()), int(df["Year"].max()) + 1)})
        df = full_years.merge(df, on="Year", how="left").fillna(0)
        for c in ["Experiments", "Publications", "MorphologiesObserved"]:
            df[c] = df[c].astype(int)
        # cumulative total of distinct morphologies seen up to each year
        df["CumulativeMorphologies"] = df["MorphologiesObserved"].cumsum()
    return df

def plot_time_series(df: pd.DataFrame, title: str = "Innovation Timeline: Experiments, Publications, and Total Morphologies"):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Year"], df["Experiments"], label="Experiments per year", color="#f4a261")
    plt.plot(df["Year"], df["Publications"], label="Publications per year", color="#e9c46a")
    plt.plot(df["Year"], df.get("CumulativeMorphologies", df["MorphologiesObserved"].cumsum()),
             label="Cumulative morphologies", color="#e63946")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_yearly = load_time_series(DB_PATH)
    print(df_yearly.head(10).to_string(index=False))
    print(df_yearly.tail(10).to_string(index=False))
    plot_time_series(df_yearly)
