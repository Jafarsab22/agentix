import pandas as pd
df = pd.read_csv("results/df_choice.csv")

# How many cards with frame=1 per screen? Expect ~4 if p≈0.5, never 0 or 8 if we enforce variation.
frame_counts = df.groupby("set_id")["frame"].sum().describe()
print(frame_counts)

# Per-screen variance check for all selected badges
for col in ["frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle"]:
    if col in df.columns:
        bad = df.groupby("set_id")[col].nunique().eq(1).sum()
        total = df["set_id"].nunique()
        print(col, "screens with no variation:", bad, "of", total)
