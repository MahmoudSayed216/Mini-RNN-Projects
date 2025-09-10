class Mapper:

    char2idx = None
    idx2char = None

    @staticmethod
    def map(string):
        string = sorted(list(string))
        Mapper.idx2char = {i: c for i, c in enumerate(string)}
        Mapper.char2idx = {c:i for i , c in Mapper.idx2char.items()}



