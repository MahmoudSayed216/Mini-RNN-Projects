import torch
from model import RNNSanityCheck
import encoder
from configs import *



def generate(model, c2i, i2c, initial = "a", generate_length=50, EOS='$'):
    sentence = initial.lower()
    encoding = torch.stack(encoder.encode_example(sentence, c2i)).unsqueeze(0)
    hidden_state = model.no_history_state_vector(encoding.size(0))
    for i in range(generate_length):
        output, hidden_state = model(encoding, hidden_state)
        almost_onehot = output.squeeze(0).squeeze(0)
        if len(almost_onehot.shape) == 2: 
            idx = almost_onehot[-1].argmax().item()
        else:
            idx = almost_onehot.argmax().item()

        new_char = i2c[idx]
        if new_char == EOS:
            break
        sentence = sentence + new_char
        encoding = torch.stack(encoder.encode_example(new_char, c2i)).unsqueeze(0)

    print(sentence)



def get_model(env):
    
    model_state_dict = env['model_state_dict']
    model = RNNSanityCheck(EMBEDDING_SIZE, HIDDEN_SIZE, N_LAYERS, EMBEDDING_SIZE)
    model.load_state_dict(model_state_dict)

    model.eval()

    return model

if __name__ == "__main__":
    env = torch.load('env.pth', map_location='cpu', weights_only=False)
    model = get_model(env)
    generate(model,env['c2i'], env['i2c'], initial="m", generate_length=80, EOS='$')