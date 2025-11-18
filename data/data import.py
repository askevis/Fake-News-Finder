
import pandas as pd

#fake = pd.read_csv(r"A:\Downloads\archive (1)\News_Dataset\Fake.csv")
#true = pd.read_csv(r"A:\Downloads\archive (1)\News_Dataset\True.csv")
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Fake shape:", fake.shape)
print("True shape:", true.shape)
fake = fake.drop(columns=["subject", "date"])
true = true.drop(columns=["subject", "date"])

fake["label"] = 1
true["label"] = 0

data = pd.concat([fake, true], axis=0).reset_index(drop=True)

data.to_csv("clean_data.csv", index=False)
print("clean data shape:", data.shape)