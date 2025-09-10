from utils import get_number_of_unique_chars

EMBEDDING_SIZE = get_number_of_unique_chars()
LEARNING_RATE = 0.001
HIDDEN_SIZE = 512
N_LAYERS = 3
OUTPUT_SIZE = EMBEDDING_SIZE
EOS = '$'
EOS_IDX = 0
DATA_PATH = "data.txt"
EPOCHS = 600