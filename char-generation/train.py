from dataset import RNNDataset
from torch.utils.data import DataLoader
from model import RNNSanityCheck
import torch
from sklearn.metrics import accuracy_score
from Mapper import Mapper

def get_loader():
    data_path = "data.txt"
    TextDataset = RNNDataset(data_path)
    emb_size = TextDataset.embedding_size
    loader = DataLoader(TextDataset, batch_size=2, shuffle=True)
    
    return loader, emb_size, TextDataset.padding_idx


def test(model, loader, loss_fn, pad_idx=None):
    
    model.eval()
    outputs = []
    targets = []
    losses = []


    def extract_characters_indices(output, targets):
        output = output.argmax(dim=-1)        
        mask = targets != pad_idx
        output = output[mask]
        targets = targets[mask]
        # for t, o in zip(targets, output):
        #     if t != o:
        #         print(t, o)

        return targets.detach().numpy().tolist(), output.detach().numpy().tolist()



    for j, data in enumerate(loader):
        input, label = data
        hidden_state = model.no_history_state_vector(input.size(0))
        output, hiddent_state = model(input, hidden_state)

        batch, seq_len, vocab_size = output.shape

        logits_flat = output.view(batch*seq_len, vocab_size)
        targets_flat = label.view(batch*seq_len)

        loss = loss_fn(logits_flat, targets_flat)
        _targets, _outputs = extract_characters_indices(logits_flat, targets_flat)
        pred_indices = output.argmax(dim=-1).view(batch, seq_len)
        for i in range(batch):
            print("Pred:", "".join([Mapper.idx2char[idx.item()] for idx in pred_indices[i]]))
            print("True:", "".join([Mapper.idx2char[idx.item()] for idx in label.view(batch, seq_len)[i]]))


        losses.append(loss.item())
        targets.extend(_targets)
        outputs.extend(_outputs)
        
    
    avg_loss = sum(losses) / len(losses)
    acc_score = accuracy_score(outputs, targets)
    return avg_loss, acc_score


def train(loader, emb_size, padding_idx):
    print("<PAD> = ", padding_idx)
    model = RNNSanityCheck(emb_size, 128, n_layers=1, output_size=emb_size)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
    epochs = 400

    for epoch in range(1, epochs+1):
        model.train()
        optim.zero_grad()
        
        for j, data in enumerate(loader):
            input, label = data 
            hidden_state = model.no_history_state_vector(input.size(0))
            output, hidden_state = model(input, hidden_state)


            batch, seq_len, vocab_size = output.shape
            logits_flat = output.view(batch*seq_len, vocab_size)
            target_flat = label.view(batch*seq_len)


            loss = loss_fn(logits_flat, target_flat)
            loss.backward()
            optim.step()
        if epoch%20 == 0:
            print("preds on the run: ", logits_flat.argmax(dim=-1))
        test_loss, test_acc = test(model, loader, loss_fn, padding_idx)
        print(f'Epoch {epoch}, Loss: {test_loss:.3f} - Accuracy: {test_acc:.2f}')

    return model


if __name__ == "__main__":
    loader, emb_size, padding_idx = get_loader()
    train(loader, emb_size, padding_idx)




