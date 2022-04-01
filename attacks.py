import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D 
import logging
import mosek
import gc
from multiprocessing import Pool
from mlc_attack_losses import LinearLoss
import math
import matplotlib.pyplot as plt
import seaborn as sns

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)


# def get_weights(flipups, flipdowns, number_of_attacked_labels, target_vector, random=False):
    
#     rankings = target_vector * flipups[:,0] + (~target_vector) * flipdowns[:,0]
#     rankings = torch.argsort(rankings, dim=1, descending=True)
#     weights = torch.zeros(target_vector.shape)
#     if random == True:
#         weights[:, np.random.permutation(target_vector.shape[1])[0:number_of_attacked_labels]] = 1
#     else:
#         weights[:, rankings[:, 0:number_of_attacked_labels]] = 1
#     return weights


def get_weights(outputs, number_of_attacked_labels, target_vector, random=False):
    rankings = (1-target_vector) * outputs + target_vector * (1-outputs)
    rankings = torch.argsort(rankings, dim=1, descending=False)
    weights = torch.zeros(target_vector.shape)
    if random == True:
        weights[:, np.random.permutation(target_vector.shape[1])[0:number_of_attacked_labels]] = 1
    else:
        weights[:, rankings[:, 0:number_of_attacked_labels]] = 1
    return weights

def pgd(model, images, target, loss_function=torch.nn.BCELoss(), eps=0.3, alpha=2/255, iters=10, device='cuda'):

    loss = loss_function
    images = images.to(device).detach()
    target = target.to(device).float().detach()
    model = model.to(device)
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

        # perform the step
        adv_images = images - alpha * images.grad.sign()
        # print(images.grad[0])

        # bound the perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        # construct the adversarials by adding perturbations
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images


# Momentum Induced Fast Gradient Sign Method 
def mi_fgsm(model, images, target, loss_function=torch.nn.BCELoss(), eps=0.3, device='cuda'):
    
    # put tensors on the GPU
    images = images.to(device)
    target = target.to(device).float()
    model = model.to(device)

    L = loss_function

    alpha = 1/256
    iters = int(eps / alpha)
    # iters = 10
    # alpha = eps / iters
    mu = 1.0
    g = 0
    
    for i in range(iters):    
        images.requires_grad = True

        # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!

        outputs = sigmoid(model(images)).to(device)
        model.zero_grad()
        cost = L(outputs, target.detach())
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


# Momentum Induced Fast Gradient Sign Method 
# def smart_mi_fgsm(model, images, target, flips_ratio, eps=0.3, iters=10, device='cuda'):
       
#     # put tensors on the GPU
#     images = images.to(device)
#     target = target.to(device).float()
#     model = model.to(device)

#     alpha = eps / iters
#     mu = 1.0
#     g = 0

#     # flips_ratio = torch.sum(torch.logical_xor((sigmoid(model(images)) > 0.5).int(), (sigmoid(model(mi_fgsm(model, images, target, eps=0.3, iters=10, device='cuda'))) > 0.5).int())).item() / 80
#     a = 1000 - np.exp(np.log(1000) * flips_ratio)

#     L = SmartLoss(a)
    
#     for i in range(iters):    
#         images.requires_grad = True

#         # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!

#         outputs = model(images).to(device)

#         model.zero_grad()
#         cost = L(outputs, target.detach())
#         cost.backward()

#         # normalize the gradient
#         new_g = images.grad / torch.sum(torch.abs(images.grad))

#         # update the gradient
#         g = mu * g + new_g

#         # perform the step, and detach because otherwise gradients get messed up.
#         images = (images - alpha * g.sign()).detach()

#     # clamp the output
#     images = torch.clamp(images, min=0, max=1).detach()
            
#     return images

# def get_weights_from_correlations(flipup_correlations, flipdown_correlations, outputs, gamma, number_of_labels, target):


#     weights = torch.zeros(target.shape)
#     outputs = (target) * outputs + (1-target) * (1-outputs)
#     outputs = outputs.detach().cpu().numpy()
#     target = target.cpu().numpy()

#     # Construct attack correlation matrix
#     # negative_indices = (target == 0).nonzero()[:, 1]
#     # positive_indices = (target == 1).nonzero()[:, 1]
#     negative_indices = np.where(target == 0)
#     positive_indices = np.where(target == 1)
#     negative_indices_per_batch = []
#     positive_indices_per_batch = []

    

#     for i in range(target.shape[0]):
#         negative_indices_per_batch.append(negative_indices[1][np.where(negative_indices[0] == i)])
#         positive_indices_per_batch.append(positive_indices[1][np.where(positive_indices[0] == i)])


#         current_positive_indices = positive_indices_per_batch[i]
#         current_negative_indices = negative_indices_per_batch[i]

#         attack_correlations = np.zeros(flipup_correlations.shape)
#         attack_correlations[current_positive_indices] = flipup_correlations[current_positive_indices]
#         attack_correlations[current_negative_indices] = flipdown_correlations[current_negative_indices]

        
#         normalized_confidences = np.abs(outputs[i]) / np.max(np.abs(outputs[i]))
#         normalized_confidences = np.transpose(np.squeeze(normalized_confidences))
#         confidence_rankings = np.argsort(np.abs(outputs[i]))

#         # Greedy correlated label select
#         root_label = confidence_rankings[len(confidence_rankings) - 1].item()

#         label_set = [root_label]

#         for j in range(number_of_labels-1):

#             correlation_to_set = attack_correlations[:, label_set].sum(axis=1)
#             correlation_from_set = attack_correlations[label_set, :].sum(axis=0)
#             correlation_factors = correlation_to_set + correlation_from_set
#             normalized_correlation_factors = correlation_factors / np.max(correlation_factors)
#             factors = gamma * normalized_correlation_factors + (1-gamma) * normalized_confidences
#             ranking = np.argsort(factors)
#             updated_ranking = [x for x in ranking if x not in label_set]
#             label_set.append(updated_ranking[len(updated_ranking)-1])

    
#         weights[i, label_set] = 1
#     return weights

def get_weights_from_correlations(flipup_correlations, flipdown_correlations, target, outputs, number_of_labels, gamma, number_of_branches, branch_depth):

    weights = torch.zeros(target.shape)
    outputs = (target) * outputs + (1-target) * (1-outputs)
    outputs = outputs.detach().cpu().numpy()
    target = target.cpu().numpy()

    for i in range(target.shape[0]):
        weights[i, get_easiest_n_labels(target[i,:], flipup_correlations, flipdown_correlations, outputs[i,:], number_of_labels, gamma, number_of_branches, branch_depth)] = 1
    return weights


def get_easiest_n_labels(target, flipup_correlations, flipdown_correlations, outputs, number_of_labels, gamma, number_of_branches, branch_depth):    

    for i in range(target.shape[0]):
        
        negative_indices = np.where(target == 0)
        positive_indices = np.where(target == 1)

        instance_correlation_matrix = np.zeros(flipup_correlations.shape)
        instance_correlation_matrix[positive_indices] = flipup_correlations[positive_indices]
        instance_correlation_matrix[negative_indices] = flipdown_correlations[negative_indices]

        normalized_confidences = np.abs(outputs) / np.max(np.abs(outputs))

        return look_ahead_easiest_n_labels(normalized_confidences, instance_correlation_matrix, number_of_labels, gamma, number_of_branches, branch_depth)
        
class TreeOfLists():

    def __init__(self):
        self.baselist = []
        self.added_labels = []
        self.children = []

    def add_child(self, label):
        child = TreeOfLists()
        child.baselist = self.baselist.copy()
        child.added_labels = self.added_labels.copy()
        child.added_labels.append(label)
        self.children.append(child)

    def get_list(self):
        total_list = self.baselist.copy() + self.added_labels.copy()
        return total_list


def look_ahead_easiest_n_labels(normalized_confidences, instance_correlation_matrix, number_of_labels, gamma, number_of_branches, branch_depth):

    confidence_rankings = np.argsort(normalized_confidences)
    root_label = confidence_rankings[len(confidence_rankings) - 1].item()

    # Initialize the label set with easiest/closest label
    base_label_set = [root_label]

    # We iteratively add a label until pre-specified length is reached
    for l in range(number_of_labels-1):

        # We have 'number_of_branches' branches to explore up until depth 'branch_depth' for the best option
        root = TreeOfLists()
        root.baselist = base_label_set.copy()
        parents = [root]
        children = []
        depth = min(branch_depth, number_of_labels - len(base_label_set))

        # Look mutiple levels ahead and pick the best option to add to the list
        for d in range(depth):

            for parent in parents:

                current_label_set = parent.get_list()

                ## COMPUTE THE CURRENT LABEL RANKINGS FOR THIS PARENT

                # We compute the correlations from and to the set by using the correlation matrix, we then select the best option 
                correlation_to_set = instance_correlation_matrix[:, current_label_set].sum(axis=1)
                correlation_from_set = instance_correlation_matrix[current_label_set, :].sum(axis=0)
                correlation_factors = correlation_to_set + correlation_from_set
                normalized_correlation_factors = correlation_factors / np.max(correlation_factors)

                # gamma determines the priority distribution between label confidence and correlation
                scores = gamma * normalized_correlation_factors + (1-gamma) * normalized_confidences
                ranking = np.argsort(scores)
                updated_ranking = [x for x in ranking if x not in current_label_set]

                ## FOR EACH BRANCH ADD A TOP LABEL FROM THE RANKING
                for b in range(number_of_branches):
                    added_label = updated_ranking[len(updated_ranking)-1-b]
                    parent.add_child(added_label)

                children.extend(parent.children)

            parents = children
            children = []

        # find the best leaf node and use its parent from the first sub-root level as a next added label
        max_obj_value = 0
        best_option = None
        for p in parents:
            obj_value = objective_function(p.get_list(), instance_correlation_matrix, normalized_confidences, gamma)
            if obj_value > max_obj_value:
                max_obj_value = obj_value
                best_option = p

        base_label_set.append(best_option.added_labels[0])
 
    return base_label_set


def objective_function(label_set, instance_correlation_matrix, normalized_confidences, gamma):
    correlation_score = 0
    for label in label_set:
        correlation_score = correlation_score + instance_correlation_matrix[label, label_set].sum()
    confidence_score = normalized_confidences[label_set].sum()

    return gamma * correlation_score + (1-gamma) * confidence_score


def correlation_mi_fgsm(model, images, flipup_correlations, flipdown_correlations, gamma, number_of_labels, number_of_branches, branch_depth, random=False, device='cuda'):

    # put tensors on the GPU
    images = images.to(device).detach()
    model = model.to(device)

    alpha = 1/256
    mu = 1.0
    g = 0

    with torch.no_grad():
        original_output = sigmoid(model(images)).detach()
        original_pred = (original_output > 0.5).int().to(device).detach()
        target = (1 - original_pred).to(device).float().detach()
        rankings = torch.argsort(torch.abs(original_output), descending=True).detach()
        if random is True:
            weights = torch.zeros(target.shape).to(device)
            weights[:, np.random.permutation(target.shape[1])[0:number_of_labels]] = 1
        else:
            weights = get_weights_from_correlations(flipup_correlations, flipup_correlations, target, original_output, number_of_labels, gamma, number_of_branches, branch_depth,).to(device)
    
    L = torch.nn.BCELoss(weight=weights)

    done = False
    iters = 0
    epsilon_values = np.zeros(target.shape[0])

    while not done:    

        images.requires_grad = True

        # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
        outputs = sigmoid(model(images)).to(device)
        model.zero_grad()
        cost = L(outputs, target.detach())
        cost.backward()


        # normalize the gradient
        new_g = images.grad / torch.sum(torch.abs(images.grad))

        # update the gradient
        g = mu * g + new_g

        # perform the step, and detach because otherwise gradients get messed up.
        images = (images - alpha * g.sign()).detach()

        # clamp the output
        images = torch.clamp(images, min=0, max=1).detach()

        with torch.no_grad():
            pred = (sigmoid(model(images)) > 0.5).int().to(device)
            flips = torch.sum(torch.logical_xor(pred, original_pred) * weights, dim=1)
            for i  in range(target.shape[0]):
                if flips[i] >= number_of_labels and epsilon_values[i] == 0:
                    epsilon_values[i] = iters * alpha
            if flips.sum() >= target.shape[0] * number_of_labels:
                done = True
        iters = iters + 1
        if iters > 100:
            done = True
            print("couldn't flip all labels, ", len([x for x in list(epsilon_values) if x == 0]), gamma)
            
    return [x for x in list(epsilon_values) if x != 0]



# Fast Gradient Sign Method 
def fgsm(model, images, target, loss_function=torch.nn.BCELoss(), eps=0.3, device='cuda'):
    
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

# Code taken from https://github.com/hinanmu/MLALP/
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

# Code taken from https://github.com/hinanmu/MLALP/
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

# Code taken from https://github.com/hinanmu/MLALP/
def ml_deep_fool(model, x, pred, target, iterations=40, **kwargs):
    clip_max = 1
    clip_min = 0
    x_shape = x.shape[1:]
    num_features = x_shape[0] * x_shape[1] * x_shape[2]
    num_labels = target.shape[1]
    num_instaces = target.shape[0]

    x = x.detach().cpu().numpy()
    
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

# Code taken from https://github.com/hinanmu/MLALP/
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
        y[ineq[0][i], ineq[1][i]] = -y[ineq[0][i], ineq[1][i]]

    return y


# Code taken from https://github.com/hinanmu/MLALP/
def get_target_set(y, y_target):
    y[y == 0] = -1
    A_pos = np.logical_and(np.not_equal(y, y_target), y == 1) + 0
    A_neg = np.logical_and(np.not_equal(y, y_target), y == -1) + 0
    B_pos = np.logical_and(np.equal(y, y_target), y == 1) + 0
    B_neg = np.logical_and(np.equal(y, y_target), y == -1) + 0

    y_tor = A_pos * -2 + -1 * B_neg + 1 * B_pos + 2 * A_neg
    return y_tor, A_pos, A_neg, B_pos, B_neg

# Code taken from https://github.com/hinanmu/MLALP/
def ml_lp(model, x, pred, target, **kwargs):

    _, A_pos, A_neg, B_pos, B_neg = get_target_set(pred.detach().cpu().numpy(), target.detach().cpu().numpy())
    A_m = A_pos + A_neg
    x = x.detach().cpu().numpy()
    logging.info('prepare attack')
    clip_max = 1
    clip_min = 0
    y_target = get_target_label(pred.detach().cpu().numpy(), target.detach().cpu().numpy())
    max_iter = 10
    x_shape = x.shape[1:]
    y_target[y_target == -1] = 0

    num_features = x_shape[0] * x_shape[1] * x_shape[2]
    num_labels = y_target.shape[1]
    num_instaces = y_target.shape[0]

    logging.info('get label gradient')

    x_t = torch.FloatTensor(x)
    y_target_t = torch.FloatTensor(y_target)
    if torch.cuda.is_available():
        x_t = x_t.cuda()
        y_target_t = y_target_t.cuda()

    # the probability and prediction of original batch x
    output = model(x_t).cpu().detach().numpy()
    pred = output.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    # the variabes save best adversarial examples and its output
    best_adv_x = x.copy()
    best_pred = pred.copy()
    adv_pred = pred

    iteration = 0
    r_change = np.zeros_like(x)
    threshold_value = np.ones((num_instaces, num_labels)) * 0.5

    # Probability of attack label output after each attack, attack sample, attack output
    attack_bool = (A_m == 1)
    adv_output = output.copy()  # shape (batch, n_label)
    before_adv_output = np.copy(adv_output[attack_bool])  # shape size (batch)
    adv_x = x.copy()  # shape (batch, channels, height, width)

    error_idx = []
    while iteration < max_iter:
        if len(error_idx) == num_instaces:
            break
        for i in range(num_instaces):
            if (best_pred[i] == y_target[i]).all() and (i not in error_idx):
                log_info = '    {}     iteration， example   {}  found attack sample,  rmsd:   {} , norm:  {} , max:    {}    , mean:    {}  '.format(
                    iteration, i,
                    np.sqrt(np.mean(np.square(((best_adv_x[i] / 2 + 0.5) * 255) - ((x[i] / 2 + 0.5) * 255)))),
                    np.linalg.norm(best_adv_x[i] - x[i]),
                    np.max(np.abs(best_adv_x[i] - x[i])),
                    np.mean(np.abs(best_adv_x[i] - x[i])))

                error_idx.append(i)
                logging.info(log_info)

        x_t = torch.FloatTensor(adv_x)
        if torch.cuda.is_available():
            x_t = x_t.cuda()

        # get jacobian matrix
        # gradients shape [n_label, batch, x_features]
        # output shape [batch, n_label]
        gradients, output = get_jacobian_loss(model, x_t, y_target_t, num_labels)

        # gradient_samples shape [batch, n_label, x_features]
        jac_grad, output = get_jacobian_loss(model, x_t, y_target_t, num_labels)
        jac_grad = np.asarray(jac_grad).reshape((num_labels, num_instaces, num_features))
        jac_grad = jac_grad.swapaxes(1, 0)

        # multi-process solving, one sample has one process
        item_list = []
        for i in range(num_instaces):
            item = []
            item.append(np.expand_dims(jac_grad[i], 0))
            item.append(np.expand_dims(output[i], 0))
            item.append(np.expand_dims(y_target[i], 0))
            item.append(threshold_value[i])
            item.append(np.expand_dims(r_change[i], 0))
            item_list.append(item)
        with Pool(processes=num_instaces) as pool:
            result = pool.starmap(mosek_inner_point_solver, item_list)

        for i in range(num_instaces):
            if i in error_idx:
                result[i] = np.zeros((1, num_features))
        result = np.asarray(result)
        result = result.reshape((num_instaces, x_shape[0], x_shape[1], x_shape[2]))
        temp_adv_x = np.clip(adv_x + result, a_min=clip_min, a_max=clip_max)

        x_t = torch.FloatTensor(temp_adv_x)
        if torch.cuda.is_available():
            x_t = x_t.cuda()

        temp_adv_output = model(x_t).cpu().detach().numpy()

        for i in range(num_instaces):
            if i in error_idx:
                continue

            if temp_adv_output[attack_bool][i] == output[attack_bool][i] and iteration >= 5:
                msg = 'example    {}  failed to solve'.format(i)
                logging.info(msg)
                error_idx.append(i)
                continue
            if np.all(result[i] == 0):
                msg = 'example    {}  failed to solve'.format(i)
                logging.info(msg)
                error_idx.append(i)
                continue

            if (before_adv_output[i] <= temp_adv_output[attack_bool][i] or temp_adv_output[attack_bool][
                i] < 0.6) and threshold_value[attack_bool][i] > 0.1:
                temp = threshold_value[attack_bool]
                temp[i] = temp[i] - 0.05
                threshold_value[attack_bool] = temp[i]

        adv_x = temp_adv_x
        # r_change = adv_x[i] - x[i]
        adv_output = temp_adv_output
        before_adv_output = temp_adv_output[attack_bool]

        pred_temp = adv_output.copy()
        pred_temp[pred_temp >= 0.5] = 1
        pred_temp[pred_temp < 0.5] = 0
        adv_pred = pred_temp

        for i in range(num_instaces):
            if i in error_idx:
                continue

            eq_value_1 = np.sum((adv_pred[i] == y_target[i]) + 0)
            eq_value_2 = np.sum((best_pred[i] == y_target[i]) + 0)
            if eq_value_1 > eq_value_2:
                best_adv_x[i] = adv_x[i]
                best_pred[i] = adv_pred[i]
            elif eq_value_1 == eq_value_2 and np.sqrt(np.mean(np.square(adv_x[i] - x[i]))) < np.sqrt(
                    np.mean(np.square(
                            best_adv_x[i] - x[i]))):
                best_adv_x[i] = adv_x[i]
                best_pred[i] = adv_pred[i]
            elif eq_value_1 < eq_value_2:
                adv_x[i] = x[i]
                best_adv_x[i] = x[i]
                best_pred[i] = pred[i]
        iteration = iteration + 1

    return best_adv_x

# Code taken from https://github.com/hinanmu/MLALP/
def get_jacobian_loss(model, x, y_target_t, noutputs):
    num_instaces = x.size()[0]
    v = torch.eye(noutputs).cuda()
    jac = []

    if torch.cuda.is_available():
        x = x.cuda()
    x.requires_grad = True
    y = model(x)
    y = torch.clamp(y, 1e-6, 1 - 1e-6)
    loss = -(y_target_t * torch.log(y) + (1 - y_target_t) * torch.log(1 - y))

    retain_graph = True
    for i in range(noutputs):
        if i == noutputs - 1:
            retain_graph = False
        loss.backward(torch.unsqueeze(v[i], 0).repeat(num_instaces, 1), retain_graph=retain_graph)
        g = x.grad.cpu().detach().numpy()
        # 梯度清零
        x.grad.zero_()
        jac.append(g)
    jac = np.asarray(jac)
    y = y.cpu().detach().numpy()
    return jac, y


# Code taken from https://github.com/hinanmu/MLALP/
def mosek_inner_point_solver(A, output, y_target, threshold_value, r_change):
    """
    Solving Linear Programming on a Sample use Mosek
    the tutorial of Mosek on (https://docs.mosek.com/9.1/pythonapi/tutorial-lo-shared.html)
    :param A: jac matrix
        shape: [1, num_labels, input_dimension]
    :param output: label confidence
        shape: [1, num_labels]
    :param y_target: target label
        shape: [1, num_labels]
        value: {0, 1}
    :param threshold_value: target label confidence, default is threshold(0.5)
        shape: [1, num_labels]
        value: 0.5
    :param r_change:
    :return:
    """
    num_labels = y_target.shape[-1]
    num_instances = A.shape[0]
    d = A.shape[-1]
    inf = 0.0

    output = np.clip(output, 1e-6, 1.0 - (1e-6))
    loss_current = -(y_target * np.log(output) + (1 - y_target) * np.log(1 - output))
    loss_target = -(y_target * np.log(np.ones((num_labels)) * threshold_value) + (1 - y_target) * np.log(
        1 - np.ones((num_labels)) * threshold_value))
    delta_loss = loss_target - loss_current
    r_change = r_change.reshape(num_instances, d)

    tasks = []
    with mosek.Env() as env:
        for i in range(num_instances):
            task = mosek.Task(env, 0, 0)
            # task.set_Stream(mosek.streamtype.log, streamprinter)
            task.putintparam(mosek.iparam.intpnt_multi_thread, mosek.onoffkey.off)
            A = A[i]

            delta_loss = delta_loss[i]

            # delete the full zero rows in jac matrix
            delete_idx = []
            for j in range(num_labels):
                if (A[j] == 0).all():
                    delete_idx.append(j)
            if len(delete_idx) != 0:
                A = np.delete(A, delete_idx, axis=0)
                delta_loss = np.delete(delta_loss, delete_idx, axis=0)
            rows = A.shape[0]


            # Set the boundry of all variables
            # boundkey.fr [-inf, +inf]
            # boundkey.lo [value, +inf]
            # boundry for r1, r2,..., rd, z
            # bkx: boundry type
            # blx: low boundry
            # blx: up boundry
            bkx = [mosek.boundkey.ra for i in range(d)] + [mosek.boundkey.lo]
            blx = [-1 for i in range(d)] + [0.0]
            bux = [1 for i in range(d)] + [1.]

            # Set the boundry of all constraints
            # boundkey.up [-inf, value]
            # bkc: boundry type
            # blc: low boundry
            # blc: up boundry
            bkc = [mosek.boundkey.up for i in range(rows + d * 2)]
            blc = [-inf for i in range(rows + d * 2)]
            buc = delta_loss.tolist() + [0.0 for i in range(d * 2)]

            #  sparse matrix ordinal value, matrix stored by column.
            # shape: n_x_feature, n_label
            # [[0, 1, 2, ..., 19],
            #   ...
            # [0, 1, 2, ..., 19]]
            zeros_to_rows = np.tile(np.arange(rows).reshape(1, -1), (d, 1))

            # [20, 22, 24, ...,]
            rows_to_rows_add_d = np.arange(start=rows, stop=rows + d * 2, step=2).reshape(-1, 1)
            # [21, 23, 25, ...,]
            rows_add1_to_rows_add_d = (rows_to_rows_add_d + 1).reshape(-1, 1)
            B = np.c_[zeros_to_rows, rows_to_rows_add_d, rows_add1_to_rows_add_d]
            asub = B.tolist()

            ones_d = np.ones(d).reshape(-1, 1)
            A = np.c_[A.T, ones_d, -ones_d]
            aval = A.tolist()

            asub.extend([[i for i in range(rows, rows + d * 2)]])
            aval.extend([[-1 for i in range(rows, rows + d * 2)]])
            c = [0.0 for i in range(d)] + [1]
            numvar = len(bkx)
            numcon = len(bkc)
            task.appendcons(numcon)
            task.appendvars(numvar)

            #garbage collection
            del A
            del B
            del output
            del zeros_to_rows
            del rows_to_rows_add_d
            del rows_add1_to_rows_add_d
            gc.collect()


            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                task.putacol(j,  # Variable (column) index.
                             asub[j],  # Row index of non-zeros in column j.
                             aval[j])  # Non-zero Values of column j.
            for j in range(numcon):
                task.putconbound(j, bkc[j], blc[j], buc[j])

            task.putobjsense(mosek.objsense.minimize)
            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)

            tasks.append(task)

        # if the solution fails, then result is full zero
        result = np.zeros((num_instances, d))
        for i in range(num_instances):
            tasks[i].optimize()
            # tasks[i].solutionsummary(mosek.streamtype.msg)
            # Get status information about the solution
            solsta = tasks[i].getsolsta(mosek.soltype.itr)
            if (solsta == mosek.solsta.optimal):
                xx = [0.] * numvar
                tasks[i].getxx(mosek.soltype.itr,  # Request the basic solution.
                               xx)
                xx = np.asarray(xx)
                result[i] = xx[:-1]
            elif (solsta == mosek.solsta.dual_infeas_cer or
                  solsta == mosek.solsta.prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")
    return result

