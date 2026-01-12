from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
import os


df = pd.read_csv("/content/drive/MyDrive/clean_data.csv")

# make sure labels are correct type
df["label"] = df["label"].astype(int)

# sample fake and real articles.
df_0 = df[df["label"] == 0].sample(n=500, random_state=42)
df_1 = df[df["label"] == 1].sample(n=500, random_state=42)

df_sample = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
print(df_sample["label"].value_counts())


df_sample["text"] = (
    "query: " + df_sample["title"] + " " + df_sample["text"]
)

#data prep
dataset = Dataset.from_pandas(df_sample)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]

#model
os.environ["WANDB_DISABLED"] = "true" # ignore collab warnings
model = SetFitModel.from_pretrained("intfloat/e5-base").to("cuda") # run on gpu

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    batch_size=16,         # smaller batch for CPU stability
    num_iterations=7,
    num_epochs=1,

)

trainer.train()

#preds
preds = trainer.model.predict(test_ds["text"])
labels = test_ds["label"]

acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test F1 Score: {f1:.4f}")
