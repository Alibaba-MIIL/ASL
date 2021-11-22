import numpy as np
import logging
import torch

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

def get_target_set(y, y_target):
    y[y == 0] = -1
    A_pos = np.logical_and(np.not_equal(y, y_target), y == 1) + 0
    A_neg = np.logical_and(np.not_equal(y, y_target), y == -1) + 0
    B_pos = np.logical_and(np.equal(y, y_target), y == 1) + 0
    B_neg = np.logical_and(np.equal(y, y_target), y == -1) + 0

    y_tor = A_pos * -2 + -1 * B_neg + 1 * B_pos + 2 * A_neg
    return y_tor, A_pos, A_neg, B_pos, B_neg