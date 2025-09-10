s = ["ding dong", "barabing barabong", "mahmoud sayed hussein abdein"]


def pad_strings(strings):
    max_len = max(strings, key=lambda x : len(x))
    
    padding_sizes = [len(max_len) - len(seq) for seq in strings]
    for i, size in enumerate(padding_sizes):
        strings[i] = strings[i] + "".join('0'*size)

    return strings


s = pad_strings(s)

print(s)