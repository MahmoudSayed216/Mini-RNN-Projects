def get_number_of_unique_chars():
    file = open("data.txt")
    lines = file.readlines()
    lines = [line.strip('\n') for line in lines]
    all_strings = "".join(lines).lower()
    number_of_unique_chars = len(set(all_strings))
    file.close()

    return number_of_unique_chars+1 # for EOS