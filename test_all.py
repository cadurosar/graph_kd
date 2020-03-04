import argparse

import torch
import torch.nn as nn
import utils
import os


def main():
    trainloader, testloader = utils.load_data(128)
    for filename in sorted(os.listdir('checkpoint')):
        file = "checkpoint/{}".format(filename)
        model = torch.load(file)["net"].module
        print(filename,model.do_bn)
        
        utils.test(model,testloader, "cuda", "no",show="error")

if __name__ == "__main__":
    main()
