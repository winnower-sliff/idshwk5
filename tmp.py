file1="train.txt"
file2="test.txt"
with open(file1) as f:
    with open(file2,"w") as f2:
        for line in f:
            tokens = line.split(",")
            name = tokens[0].strip()
            f2.write(name+'\n')
        