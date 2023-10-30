import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from preprocessing import preprocess_agnews, AGNewsDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="agnews")
parser.add_argument("--use_agnews_title", action="store_true")
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--num_epoch", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=3e-5)

args = parser.parse_args()

if args.data_name == "agnews":
    preprocessing_func = preprocess_agnews

train_text, train_label, val_text, val_label = preprocessing_func(
    args.data_name,
    data_type="train",
    use_agnews_title=args.use_agnews_title,
)
test_text, test_label = preprocessing_func(
    args.data_name,
    data_type="test",
    use_agnews_title=args.use_agnews_title,
)
num_labels = len(set(train_label))

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

train_encodings = tokenizer(
    train_text, truncation=True, padding=True, max_length=args.max_length
)
val_encodings = tokenizer(
    val_text, truncation=True, padding=True, max_length=args.max_length
)
test_encodings = tokenizer(
    test_text, truncation=True, padding=True, max_length=args.max_length
)

train_dataset = AGNewsDataset(train_encodings, train_label)
val_dataset = AGNewsDataset(val_encodings, val_label)
test_dataset = AGNewsDataset(test_encodings, test_label)


train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=args.test_batch_size, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=False
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=num_labels,
)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

model.train()
for epoch in tqdm(range(args.num_epoch)):
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    y_preds = []
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        _, predicted = torch.max(outputs[1], 1)
        y_preds.extend(predicted.tolist())

print(f"Acc: {accuracy_score(test_label, y_preds)}")