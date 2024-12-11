import torch
def test(a):
    a -= 1
a = torch.tensor(2)
test(a)
print(a)
b = 2
test(b)
print(b)

# tensor传的参居然是同步变的