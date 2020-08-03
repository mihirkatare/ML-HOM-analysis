import ast
def parse(path):
    with open(path) as f:
        temp = dict()
        for line in f:
            if(line[0]!="%" and line != "\n"):
                a = line.strip("\n").split("=")
                temp[a[0].strip()] = ast.literal_eval(a[1].strip())
    return temp

# print(parse("Input.txt"))
