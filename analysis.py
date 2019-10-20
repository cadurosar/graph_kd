import argparse

import torch
import torch.nn as nn
import utils
import numpy as np
import numpy.linalg as linalg


def count_conv2d(m, x, y):
    x = x[0] # remove tuple

    fin = m.in_channels
    fout = m.out_channels 
    sh, sw = m.kernel_size

    # ops per output element
    kernel_mul = sh * sw * fin
    kernel_add = sh * sw * fin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops
    
    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

#    print("Conv2d: S_c={}, F_in={}, F_out={}, P={}, params={}, operations={}".format(sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0] # remove tuple
    
    nelements = x.numel()
    total_sub = 2*nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])
#    print("Batch norm: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))

   
def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
#    print("Linear: F_in={}, F_out={}, params={}, operations={}".format(m.in_features,m.out_features,int(m.total_params.item()),int(total_ops)))
    m.total_ops += torch.Tensor([int(total_ops)])

def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):    
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params
    total_ops = total_ops
    total_params = total_params

    return total_ops, total_params

def representations_to_adj(representations, sigma=1):
    rview = representations.view(representations.size(0),-1)
    distances = torch.cdist(rview,rview)/(2*sigma**2)
    adj = torch.exp(-distances)
    ind = np.diag_indices(adj.shape[0])
    adj[ind[0], ind[1]].data = torch.zeros(adj.shape[0]).cuda()
    degree = torch.pow(adj.sum(dim=1),-0.5)
    degree_matrix = torch.diag(degree)
    return torch.matmul(degree_matrix,torch.matmul(adj,degree_matrix))


def main():
    device = "cuda"
    teacher_file = "checkpoint/WideResNet28-10.pth"
    teacher_model = torch.load(teacher_file)["net"].module

    student_file = "checkpoint/HKD_28-10_teaches_28-1_16_4.pth"
#    student_file = "checkpoint/WideResNet28-1.pth"
#    student_file = "checkpoint/GKD_28-10_teaches_28-1_0_0_p1_25.pth"
    student_model = torch.load(student_file)["net"].module
    student_model.eval()
    teacher_model.eval()
    
    trainloader, testloader = utils.load_data(128)
    identity = torch.eye(1000).cuda()


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_teacher, layers_teacher = teacher_model(inputs)
            outputs_student, layers_student = student_model(inputs)
            for student_layer,teacher_layer in zip(layers_student,layers_teacher):
                adj_teacher = representations_to_adj(teacher_layer)
                laplacian_teacher = adj_teacher
                laplacian_teacher = laplacian_teacher.cpu().numpy()

                w, v = linalg.eig(laplacian_teacher)
                seen = {}
                unique_eigenvalues = []
                for (x, y) in zip(w, v):
                    if x in seen:
                        continue
                    seen[x] = 1
                    unique_eigenvalues.append((x, y))
                fiedler_vector = sorted(unique_eigenvalues)[1][1].reshape(1000,1)

                adj_student = representations_to_adj(student_layer)
                laplacian_student = identity - adj_student
                laplacian_student = laplacian_student.cpu().numpy()
                smoothness = np.dot(fiedler_vector.T,laplacian_student)
                smoothness = np.dot(smoothness,fiedler_vector)
                print(smoothness.sum())
            break


if __name__ == "__main__":
    main()
