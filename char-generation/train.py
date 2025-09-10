from dataset import RNNDataset
from torch.utils.data import DataLoader
from model import RNNSanityCheck
import torch
from sklearn.metrics import accuracy_score
from Mapper import Mapper
from configs import *


def get_loader():
    TextDataset = RNNDataset(DATA_PATH)
    
    loader = DataLoader(TextDataset, batch_size=2, shuffle=True)
    
    return loader


def test(model, loader, loss_fn):
    
    model.eval()
    outputs = []
    targets = []
    losses = []


    def extract_characters_indices(output, targets):
        output = output.argmax(dim=-1)        
        mask = targets != EOS_IDX
        output = output[mask]
        targets = targets[mask]

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

        losses.append(loss.item())
        targets.extend(_targets)
        outputs.extend(_outputs)
        
    
    avg_loss = sum(losses) / len(losses)
    acc_score = accuracy_score(outputs, targets)
    return avg_loss, acc_score


def train(loader):
    model = RNNSanityCheck(EMBEDDING_SIZE, HIDDEN_SIZE, n_layers=N_LAYERS, output_size=EMBEDDING_SIZE)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS+1):
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
        test_loss, test_acc = test(model, loader, loss_fn)
        print(f'Epoch {epoch}, Loss: {test_loss:.3f} - Accuracy: {test_acc:.2f}')

    return model

def save_env(model):
    c2i = Mapper.char2idx
    i2c = Mapper.idx2char
    print("char2index: ", c2i)
    print("index2char: ", i2c)
    torch.save({
        'model_state_dict':model.state_dict(),
        'c2i': c2i,
        'i2c':i2c},
        "env.pth"
        )


if __name__ == "__main__":
    loader = get_loader()
    model = train(loader)
    save_env(model)




