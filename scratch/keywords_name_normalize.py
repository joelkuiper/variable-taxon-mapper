import re
import pandas as pd

df = pd.read_csv("Keywords.csv")


def snake_to_title_if_underscore(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    if not s or "_" not in s:
        return s  # leave as-is if no underscore
    # Replace one or more underscores/spaces with a single space, then Title Case
    s = re.sub(r"[_\s]+", " ", s)
    return s.title()


df["name_normalized"] = df["name"].apply(snake_to_title_if_underscore)
df.to_csv("Keywords_with_normalized.csv", index=False)
df.head()
