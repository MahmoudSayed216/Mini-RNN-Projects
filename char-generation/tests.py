file = open("data.txt")
lines = file.readlines()
file.close()


lines = [line.strip('\n') for line in lines]

str_ = "".join(lines)


print(set(str_))
print(len(set(str_)))