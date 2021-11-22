import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D 
import logging



sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)



def pgd(model, images, target, target_ids=None, eps=0.3, alpha=2/255, iters=40, device='cuda'):
    
    images = images.to(device).detach()
    target = target.to(device).float().detach()
    model = model.to(device)
    loss = nn.BCELoss()

    ori_images = images.data.to(device)
        
    for i in range(iters):    
        images.requires_grad = True

        # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
        outputs = sigmoid(model(images)).to(device)
        model.zero_grad()
        cost = 0

        if target_ids:
            cost = loss(outputs[:, target_ids], target[:, target_ids].detach())
        else:
            cost = loss(outputs, target)
        cost.backward()
        # plot_grad_flow(model.named_parameters())

        # perform the step
        adv_images = images - alpha * images.grad.sign()
        # print(images.grad[0])

        # bound the perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        # construct the adversarials by adding perturbations
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

def untargeted_pgd(model, images, eps=0.3, alpha=2/255, iters=40, device='cuda'):
    
    images = images.to(device)
    model = model.to(device)
    loss = nn.BCELoss()
    ori_images = images.data.to(device)
        
    for i in range(iters):    
        images.requires_grad = True
        
        # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
        outputs = sigmoid(model(images)).to(device)

        # This assumes prediction is correct
        target = (outputs.clone() > 0.5).int().float()

        model.zero_grad()
        cost = loss(outputs, target.detach())
        cost.backward()

        # print(images.grad.sign())

        # perform the step
        adv_images = images + alpha * images.grad.sign()

        # bound the perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        # construct the adversarials by adding perturbations
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

# Momentum Induced Fast Gradient Sign Method 
def mi_fgsm(model, images, target, eps=0.3, iters=10, device='cuda'):
    
    # put tensors on the GPU
    images = images.to(device)
    target = target.to(device).float()
    model = model.to(device)
    loss = nn.BCELoss()
    alpha = eps / iters
    mu = 1.0
    g = 0
    
    for i in range(iters):    
        images.requires_grad = True

        # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
        outputs = sigmoid(model(images)).to(device)

        model.zero_grad()
        cost = loss(outputs, target.detach())
        cost.backward()

        # normalize the gradient
        new_g = images.grad / torch.sum(torch.abs(images.grad))

        # update the gradient
        g = mu * g + new_g

        # perform the step, and detach because otherwise gradients get messed up.
        images = (images - alpha * g.sign()).detach()

    # clamp the output
    images = torch.clamp(images, min=0, max=1).detach()
            
    return images


# Fast Gradient Sign Method 
def fgsm(model, images, target, eps=0.3, device='cuda'):
    
    # put tensors on the GPU
    images = images.to(device).detach()
    target = target.to(device).float()
    model = model.to(device)
    loss = nn.BCELoss()
    images.requires_grad = True

    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
    outputs = sigmoid(model(images)).to(device)

    # Compute loss and perform back-prop
    model.zero_grad()
    cost = loss(outputs, target)
    cost.backward()

    # perform the step
    images = images - eps * images.grad.sign()

    # clamp the output
    images = torch.clamp(images, min=0, max=1)

    return images


def ml_cw(model, x, pred, target, **kwargs):
    clip_max = 1
    clip_min = 0

    x = x.detach().cpu().numpy()
    # _, A_pos, A_neg, B_pos, B_neg = get_target_label(pred.detach().cpu().numpy(), target.detach().cpu().numpy())
    y_target = get_target_label(pred.detach().cpu().numpy(), target.detach().cpu().numpy())
    print(y_target)
    kwargs = {'binary_search_steps': 10,
                      'y_target': None,
                      'max_iterations': 1000,
                      'learning_rate': 0.01,
                      'batch_size': 5,
                      'initial_const': 1e5}

    max_iter = kwargs['max_iterations']
    batch_size = kwargs['batch_size']
    learning_rate = kwargs['learning_rate']
    binary_search_steps = kwargs['binary_search_steps']
    init_cons = kwargs['initial_const']

    oimgs = np.clip(x, clip_min, clip_max)
    imgs = (x - clip_min) / (clip_max - clip_min)
    imgs = np.clip(imgs, 0, 1)
    imgs = (imgs * 2) - 1
    imgs = np.arctanh(imgs * .999999)

    x_shape = x.shape[1:]
    num_features = x_shape[0] * x_shape[1] * x_shape[2]
    num_labels = y_target.shape[1]
    num_instaces = y_target.shape[0]


    upper_bound = np.ones(batch_size) * 1e10
    lower_bound = np.zeros(batch_size)
    repeat = binary_search_steps >= 10

    # placeholders for the best l2, score, and instance attack found so far
    o_bestl2 = [1e10] * batch_size
    o_bestscore = [-1] * batch_size
    o_bestattack = np.copy(oimgs)
    o_bestoutput = np.zeros_like(y_target)
    CONST = np.ones(batch_size)*init_cons
    x_t = torch.tensor(imgs)
    y_target_t = torch.tensor(y_target)


    if torch.cuda.is_available():
        model = model.cuda()
        x_t = x_t.cuda()
        y_target_t = y_target_t.cuda()


    for outer_step in range(binary_search_steps):
        modifier = torch.tensor(np.zeros_like(x))
        const_t = torch.tensor(CONST)
        if torch.cuda.is_available():
            modifier = modifier.cuda()
            const_t = const_t.cuda()
        modifier.requires_grad = True
        optimizer = torch.optim.Adam([modifier], lr=learning_rate)
        # completely reset adam's internal state.
        batch = imgs[:batch_size]
        batchlab = y_target[:batch_size]

        bestl2 = [1e10] * batch_size
        bestscore = [-1] * batch_size
        bestoutput = [0] * batch_size

        logging.info("  Binary search step %s of %s",
                      outer_step, binary_search_steps)
        # The last iteration (if we run many steps) repeat the search once.
        if repeat and outer_step == binary_search_steps - 1:
            CONST = upper_bound

        prev = 1e10
        for iteration in range(max_iter):

            output, loss, l2dist, newimg = criterion(model, y_target_t, modifier, x_t, clip_max, clip_min, const_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l = loss.item()
            l2s = l2dist.cpu().detach().numpy()
            scores = output.cpu().detach().numpy()
            nimg = newimg.cpu().detach().numpy()

            if iteration % ((max_iter // 10) or 1) == 0:
                logging.info(("    Iteration {} of {}: loss={:.3g} " +
                               "l2={:.3g} f={:.3g}").format(
                    iteration, max_iter, l,
                    np.mean(l2s), np.mean(scores)))

            # check if we should abort search if we're getting nowhere.
            if True and \
                    iteration % ((max_iter // 10) or 1) == 0:
                if l > prev * .9999:
                    msg = "    Failed to make progress; stop early"
                    logging.info(msg)
                    break
                prev = l

            for e, (l2, sc, ii, bsc) in enumerate(zip(l2s, scores, nimg, bestscore)):
                lab = batchlab[e]
                score = np.sum((((sc - 0.5) * lab) >= 0) + 0)
                if bestscore[e] < score:
                    bestl2[e] = l2
                    bestscore[e] = score
                    bestoutput[e] = sc
                elif bestscore[e] == score and l2 < bestl2[e]:
                    bestl2[e] = l2
                    bestscore[e] = score
                    bestoutput[e] = sc
                if o_bestscore[e] < score:
                    o_bestl2[e] = l2
                    o_bestscore[e] = score
                    o_bestattack[e] = ii
                    o_bestoutput[e] = sc
                if l2 < o_bestl2[e] and o_bestscore[e] == score:
                    o_bestl2[e] = l2
                    o_bestscore[e] = score
                    o_bestattack[e] = ii
                    o_bestoutput[e] = sc
                    # adjust the constant as needed

        for e in range(batch_size):
            lab = batchlab[e]
            sc = np.array(bestoutput[e])
            sc[sc >= 0.5] = 1
            sc[sc < 0.5] = -1
            if np.all(sc == lab) and \
                    bestscore[e] != -1:
                # success, divide const by two
                upper_bound[e] = min(upper_bound[e], CONST[e])
                if upper_bound[e] < 1e9:
                    CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
            else:
                # failure, either multiply by 10 if no solution found yet
                #          or do binary search with the known upper bound
                lower_bound[e] = max(lower_bound[e], CONST[e])
                if upper_bound[e] < 1e9:
                    CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    CONST[e] *= 10
        logging.info("  Successfully generated adversarial examples " +
                      "on {} of {} instances.".format(
                          sum(upper_bound < 1e9), batch_size))
        o_bestl2 = np.array(o_bestl2)
        mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
        logging.info("   Mean successful distortion: {:.4g}".format(mean))

    logging.info('get label gradient')

    return torch.tensor(o_bestattack).cuda()

def criterion(model, y, modifier, x_t, clip_max, clip_min, const):
    newimg = (torch.tanh(modifier + x_t) + 1) / 2
    newimg = newimg * (clip_max - clip_min) + clip_min

    output = model(newimg)
    # distance to the input data
    other = (torch.tanh(x_t) + 1) / \
                 2 * (clip_max - clip_min) + clip_min
    l2dist = torch.sum((newimg - other).pow(2), (1,2,3))
    temp = - y * (output - 0.5)
    loss1 = torch.sum(torch.max(torch.zeros_like(temp), temp), 1)
    # sum up the losses
    loss2 = torch.sum(l2dist)
    loss1 = torch.sum(const * loss1)
    loss = loss1 + loss2

    return output, loss, l2dist, newimg


def ml_deep_fool(model, x, pred, target, iterations=40, **kwargs):
    clip_max = 1
    clip_min = 0
    x_shape = x.shape[1:]
    num_features = x_shape[0] * x_shape[1] * x_shape[2]
    num_labels = target.shape[1]
    num_instaces = target.shape[0]

    x = x.detach().cpu().numpy()
    _, A_pos, A_neg, B_pos, B_neg = get_target_set(pred.detach().cpu().numpy(), target.detach().cpu().numpy())
    y_target = A_pos + A_neg
    x_t = torch.FloatTensor(x)
    target_t = torch.FloatTensor(y_target)
    if torch.cuda.is_available():
        model = model.cuda()
        x_t = x_t.cuda()
        target_t = target_t.cuda()

    iterations = 20
    adv_x = x.copy()
    sample = x
    x_shape = np.shape(adv_x)[1:]

    best_adv_x = adv_x.copy()
    
    # Initialize the loop variables
    i = 0

    output = model(x_t).cpu().detach().numpy()
    current = output.copy()
    current[current>=0.5] = 1
    current[current<0.5] = -1
    target = np.array(target)
    nb_labels = num_labels

    if current.shape == ():
        current = np.array([current])
    w = np.squeeze(np.zeros(sample.shape[1:]))  # same shape as original image
    r_tot = np.zeros(sample.shape)
    original = current  # use original label as the reference

    while i < iterations and np.any(current != target):
  
        x_t = torch.FloatTensor(adv_x)

        gradients, output = get_jacobian(model, x_t, num_labels)
        gradients = np.asarray(gradients)
        gradients = gradients.swapaxes(1, 0)
        predictions_val = output

        for idx in range(sample.shape[0]):
            if np.all(current[idx] == target[idx]):
                continue

            y = target[idx]
            c = -y * 0.5
            w = gradients[idx].reshape(nb_labels, -1).T
            f = predictions_val[idx]
            P = w * (-y)
            q = np.reshape(c - (-y * f), (-1, 1))
            temp = np.matmul(P.T, P)
            zeros = np.zeros(temp.shape[1])
            delete_idx = []
            for j in range(temp.shape[0]):
                if np.all(temp[j] == zeros):
                    delete_idx.append(j)
            P = np.delete(P, delete_idx, axis=1)
            q = np.delete(q, delete_idx, axis=0)
            #print(np.matmul(P.T, P))
            try:
                delta_r = np.matmul(np.matmul(P, np.linalg.inv(np.matmul(P.T, P))), q)
            except:
                continue
            delta_r = np.reshape(delta_r, x_shape)

            r_tot[idx] = r_tot[idx] + delta_r

        adv_x = np.clip(r_tot + sample, clip_min, clip_max)

        x_t = torch.FloatTensor(adv_x)
        if torch.cuda.is_available():
            x_t = x_t.cuda()

        Cst_b = np.sum(np.equal(target, current) + 0, axis=1)
        compare_val = model(x_t).cpu().detach().numpy()
        compare_val[compare_val >= 0.5] = 1
        compare_val[compare_val < 0.5] = -1
        Cst_i = np.sum(np.equal(target, compare_val) + 0, axis=1)

        for idx in range(sample.shape[0]):
            if Cst_i[idx] > Cst_b[idx]:
                best_adv_x[idx] = adv_x[idx]
                current[idx] = compare_val[idx]
            elif Cst_i[idx] == Cst_b[idx] and np.linalg.norm(adv_x[idx]) < np.linalg.norm(best_adv_x[idx]):
                best_adv_x[idx] = adv_x[idx]
                current[idx] = compare_val[idx]

        if current.shape == ():
            current = np.array([current])
        # Update loop variables
        i = i + 1

    adv_x = np.clip(best_adv_x, clip_min, clip_max)

    return torch.tensor(adv_x).cuda()

def get_jacobian(model, x, noutputs):
    num_instaces = x.size()[0]
    v = torch.eye(noutputs).cuda()
    jac = []

    if torch.cuda.is_available():
        x = x.cuda()
    x.requires_grad = True
    y = model(x)
    retain_graph = True
    for i in range(noutputs):
        if i == noutputs - 1:
            retain_graph = False
        y.backward(torch.unsqueeze(v[i], 0).repeat(num_instaces, 1), retain_graph=retain_graph)
        g = x.grad.cpu().detach().numpy()
        x.grad.zero_()
        jac.append(g)
    jac = np.asarray(jac)
    y = y.cpu().detach().numpy()
    return jac, y


def get_target_label(pred, target):

    y = pred.copy()
    y[y == 0] = -1
    ineq = np.logical_xor(pred, target).nonzero()
    for i in range(len(ineq[0])):
        y[ineq[0][i], ineq[1][i]] = -1 * y[ineq[0][i], ineq[1][i]]

    print(pred[0])
    print(target[0])
    print(y[0])
    
    return y


def get_target_set(y, y_target):
    y[y == 0] = -1
    A_pos = np.logical_and(np.not_equal(y, y_target), y == 1) + 0
    A_neg = np.logical_and(np.not_equal(y, y_target), y == -1) + 0
    B_pos = np.logical_and(np.equal(y, y_target), y == 1) + 0
    B_neg = np.logical_and(np.equal(y, y_target), y == -1) + 0

    y_tor = A_pos * -2 + -1 * B_neg + 1 * B_pos + 2 * A_neg
    return y_tor, A_pos, A_neg, B_pos, B_neg