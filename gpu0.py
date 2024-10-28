
import torch
import evaluate
import argparse
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from tuning import Trainer
from dataset import MyDataset
from model import TextClassificationModel, get_tokenizer
from common import collate_fn, ErrorRateAt95Recall1, setup_seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = Path("model1")

setup_seed(100)
    

def main(args):
    #######model#######
    model, tokenizer = TextClassificationModel(args.pretrained_model_name).to(device), get_tokenizer(args.pretrained_model_name)


    #######data########
    train_dataset = MyDataset('train', args.subject, 0.1)
    validation_dataset = MyDataset('validation', args.subject,  0.1)
    test_dataset = MyDataset('test', args.subject,  0.1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)



    #######train########
    best_accuracy = 0
    trainer = Trainer(model, device)
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_loader)
        val_accuracy, val_loss = trainer.test(validation_loader, validation_dataset)
        print("Epoch {}/{}, total loss:{:.4f}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch+1, args.epochs, train_loss, val_accuracy, val_loss))
        torch.save(model, model_dir / f"model_{epoch}.pt")

        if val_accuracy > best_accuracy:
            torch.save(model, model_dir / f"model_best.pt")
            best_accuracy = val_accuracy



    #######test########
    model = torch.load(model_dir / f"model_best.pt")
    model = model.eval()

    labels = []
    predictions = []
    probs = []
    for input, label in tqdm(test_loader):
        outputs = model(input.to(device))
        probs.append(float(outputs))
        outputs = (outputs >= 0.5).int().flatten().tolist()
        predictions.append(int(outputs[0]))
        labels.append(int(label))

    

    ########evaluate######
    fpr95 = ErrorRateAt95Recall1(labels, probs)

    acc_evaluation = evaluate.load('accuracy')
    acc = acc_evaluation.compute(references=labels, predictions=predictions)

    recall_evaluation = evaluate.load('recall')
    recall = recall_evaluation.compute(references=labels, predictions=predictions)

    precision_evaluation = evaluate.load('precision')
    precision = precision_evaluation.compute(references=labels, predictions=predictions)

    result = pd.DataFrame({"fpr95":fpr95, "ACC":acc, "Recall": recall, "precision":precision})
    result.to_csv("result/{model}_{subject}.csv".format(model=args.pretrained_model_name, subject = args.subject))

    import shutil
    from pathlib import Path
    def  del_file(path):
        for elm in Path(path).glob('*'):
                elm.unlink() if elm.is_file() else shutil.rmtree(elm)
    
    del_file('model')




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #task and prompt
    parser.add_argument('--pretrained_model_name', '-m', choices=['bert', 'robert', "xlnet"], type=str, default="bert", help='Choose pretrained model.')
    parser.add_argument('--batch', type=int, default=32, help='batch size.')
    parser.add_argument('--epochs', type=int, default=2, help='epoch num.')

    parser.add_argument('--subject', type=str, choices=['total', "政治错误"],  default="政治错误", help='subject.')

    args = parser.parse_args()
    main(args)

