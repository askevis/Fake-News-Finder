from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
import os


df = pd.read_csv("/content/clean_data.csv")

# Sample 1000 rows
df_sample = df.sample(n=1000, random_state=42)


df_sample["text"] = "query: " + df_sample["title"] + " [SEP] " + df_sample["text"]# Combine title + text , SEP to differentiate

#data prep
dataset = Dataset.from_pandas(df_sample)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]

print(f"Training examples: {len(train_ds)}")
print(f"Testing examples: {len(test_ds)}")

#model
os.environ["WANDB_DISABLED"] = "true" # ignore collab bs
model = SetFitModel.from_pretrained("intfloat/e5-base").to("cuda") # to cuda as to not kill my ram

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    #metric="accuracy",   # will compute accuracy during training
    batch_size=16,         # smaller batch for CPU stability
    num_iterations=20,
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
