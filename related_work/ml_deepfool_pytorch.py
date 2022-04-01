#@Time      :2019/10/26 11:26
#@Author    :zhounan
#@FileName  :ml_deepfool_pytorch.py
import numpy as np
import logging
import torch

class MLDeepFool(object):
    def __init__(self, model, dtypestr='float32', **kwargs):
        self.model = model

    def generate_np(self, x, A_m, **kwargs):
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']
        print(y_target)
        max_iter = kwargs['max_iter']
        x_shape = x.shape[1:]
        num_features = x_shape[0] * x_shape[1] * x_shape[2]
        num_labels = y_target.shape[1]
        num_instaces = y_target.shape[0]

        x_t = torch.FloatTensor(x)
        y_target_t = torch.FloatTensor(y_target)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            x_t = x_t.cuda()
            y_target_t = y_target_t.cuda()

        max_iter = 20
        adv_x = x.copy()
        sample = x
        x_shape = np.shape(adv_x)[1:]

        best_adv_x = adv_x.copy()
        # Initialize the loop variables
        iteration = 0

        output = torch.sigmoid(self.model(x_t)).cpu().detach().numpy()
        current = output.copy()
        current[current>=0.5] = 1
        current[current<0.5] = -1
        y_target = np.array(y_target)
        nb_labels = num_labels

        if current.shape == ():
            current = np.array([current])
        w = np.squeeze(np.zeros(sample.shape[1:]))  # same shape as original image
        r_tot = np.zeros(sample.shape)
        original = current  # use original label as the reference

        logging.debug(
            "Starting DeepFool attack up to %s iterations", max_iter)
        # Repeat this main loop until we have achieved misclassification

        while iteration < max_iter and np.any(current != y_target):
            # if iteration % 5 == 0 and iteration > 0:
            #     logging.info("Attack result at iteration %s is %s", iteration, current)
            logging.info("%s out of %s become adversarial examples at iteration %s",
                         sum(np.any(current != original, axis=1)),
                         sample.shape[0],
                         iteration)

            x_t = torch.FloatTensor(adv_x)

            gradients, output = get_jacobian(self.model, x_t, num_labels)
            gradients = np.asarray(gradients)
            gradients = gradients.swapaxes(1, 0)
            predictions_val = output

            for idx in range(sample.shape[0]):
                if np.all(current[idx] == y_target[idx]):
                    continue

                y = y_target[idx]
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

            adv_x = np.clip(r_tot + sample, self.clip_min, self.clip_max)

            x_t = torch.FloatTensor(adv_x)
            if torch.cuda.is_available():
                x_t = x_t.cuda()

            Cst_b = np.sum(np.equal(y_target, current) + 0, axis=1)
            compare_val = torch.sigmoid(self.model(x_t)).cpu().detach().numpy()
            compare_val[compare_val >= 0.5] = 1
            compare_val[compare_val < 0.5] = -1
            Cst_i = np.sum(np.equal(y_target, compare_val) + 0, axis=1)

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
            iteration = iteration + 1

        # need more revision, including info like how many succeed
        # logging.info("Attack result at iteration %s is %s", iteration, current)
        logging.info("%s out of %s become adversarial examples at iteration %s",
                     sum(np.any(current != original, axis=1)),
                     sample.shape[0],
                     iteration)
        # need to clip this image into the given range
        # adv_x = np.clip((1 + overshoot) * r_tot + sample, clip_min, clip_max)
        adv_x = np.clip(best_adv_x, self.clip_min, self.clip_max)

        return best_adv_x


def get_jacobian(model, x, noutputs):
    num_instaces = x.size()[0]
    v = torch.eye(noutputs).cuda()
    jac = []

    if torch.cuda.is_available():
        x = x.cuda()
    x.requires_grad = True
    y = torch.sigmoid(model(x))
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