import torch
model = torch.hub.load('pytorch/vision:v0.4.2',
                           'resnet18',
                           pretrained=True)
def set_fullname(mod, fullname):
    mod.fullname = fullname
    if len(list(mod.children())) == 0:
        for index, p in enumerate(mod.parameters()):
            p.reserved_name = '%s->p%d' % (fullname, index)
    for child_name, child in mod.named_children():
        child_fullname = '%s->%s' % (fullname, child_name)
        set_fullname(child, child_fullname)
set_fullname(model, 'resnet152')

model = model.eval()

def partition_model(model):
    group_list = []
    before_core = []
    core_complete = False
    after_core = []

    group_list.append(before_core)
    for name, child in model.named_children():
        if 'layer' in name:
            core_complete = True
            for _, child_child in child.named_children():
                group_list.append([child_child])
        else:
            if not core_complete:
                before_core.append(child)
            else:
                after_core.append(child)
    group_list.append(after_core)
    return group_list

for name, child in model.named_children():
    print('line: ', name, child)
print(model)
