# loads the model
# generate up to n chars
# terminates on EOF

import torch
from model import RNNSanityCheck
import encoder
from configs import *



def generate(model, dictionary, initial = "a", generate_length=50, EOS='$'):
    initial = initial.lower()
    encoding = torch.stack(encoder.encode_example(initial, c2i)).unsqueeze(0)

    hidden_state = model.no_history_state_vector(encoding.size(0))
    # for i in range(generate_length):
    output, hidden_state = model(encoding, hidden_state)
    print(output.shape)
        # print(output.argmax(dim=-1))




def get_model(env):
    
    model_state_dict = env['model_state_dict']
    model = RNNSanityCheck(EMBEDDING_SIZE, HIDDEN_SIZE, N_LAYERS, EMBEDDING_SIZE)
    model.load_state_dict(model_state_dict)

    model.eval()

    return model

if __name__ == "__main__":
    env = torch.load('env.pth', map_location='cpu', weights_only=False)
    model = get_model(env)
    i2c = env['i2c']
    c2i = env['c2i']

    generate(model,c2i, initial="I", generate_length=10, EOS='$')
    # generate(model,c2i, initial="My Love", generate_length=10, EOS='$')
    # generate(model,c2i, initial="Py", generate_length=10, EOS='$')