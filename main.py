from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import py_vncorenlp
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from optimum.bettertransformer import BetterTransformer
import torch


app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8000",
    "https://thienvp.github.io/sentimentAnalysis/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    py_vncorenlp.download_model(save_dir='./')
except:
    pass
rdrsegmenter = py_vncorenlp.VnCoreNLP(save_dir='./')

print('-----------------------------')
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3,)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

print('+++++++++++++++++++++++++')
path = "/home/mmlab/21520810/transformer-fastapi/phobert_8epochs_lr1e-5.pth"
model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

print('****************')

model = BetterTransformer.transform(model, keep_original_model=True)
print("DONE DELPOYMENT")


@app.get("/")
async def root():
    return {"message": "Hello World"}

def preprocess(text):
    text = ''.join(e for e in text if e.isalnum() or e == ' ')
    text = rdrsegmenter.word_segment(text)[0]
    return text
def tokenize_function(text):
    return tokenizer(text ,
                     truncation=True,
                     padding="max_length",
                     return_tensors='pt'
                     )

id2label = {0: "negative", 1: "neutral",2: "positive"}

@app.get("/predict")
async def predict(text):
    text = preprocess(text)
    tokenized_text = tokenize_function(text)
    with torch.no_grad():
        model.eval()
        output = model(**tokenized_text)
        logits = output.logits
        softmax = torch.nn.functional.softmax(logits,dim=1).tolist()
        # print(softmax)
    predicted_class_id = logits.argmax().item()
    return {"message": id2label[predicted_class_id], "score": softmax[0][predicted_class_id]}
    