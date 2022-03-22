#@Time      :2019/10/26 11:26
#@Author    :zhounan
#@FileName  :ml_rank1_pytorch.py
import numpy as np
import torch
import logging

class MLRank1(object):
    def __init__(self, model, dtypestr='float32', **kwargs):
        self.model = model

    def generate_np(self, x, **kwargs):
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']
        max_iter = kwargs['max_iterations']
        batch_size = kwargs['batch_size']
        learning_rate = kwargs['learning_rate']
        binary_search_steps = kwargs['binary_search_steps']
        init_cons = kwargs['initial_const']
        y_tor = kwargs['y_tor']

        oimgs = np.clip(x, self.clip_min, self.clip_max)

        imgs = (x - self.clip_min) / (self.clip_max - self.clip_min)
        imgs = np.clip(imgs, 0, 1)
        imgs = (imgs * 2) - 1
        imgs = np.arctanh(imgs * .999999)

        x_shape = x.shape[1:]
        num_features = x_shape[0] * x_shape[1] * x_shape[2]
        num_labels = y_target.shape[1]
        num_instaces = y_target.shape[0]


        upper_bound = np.ones(batch_size) * 1e10
        lower_bound = np.zeros(batch_size)
        self.repeat = binary_search_steps >= 10

        # placeholders for the best l2, score, and instance attack found so far
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(oimgs)
        o_bestoutput = np.zeros_like(y_target)
        CONST = np.ones(batch_size)*init_cons
        x_t = torch.tensor(imgs)
        y_target_t = torch.tensor(y_target)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
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
            batchlab_tor = y_tor[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            bestoutput = [0] * batch_size

            logging.info("  Binary search step %s of %s",
                          outer_step, binary_search_steps)
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == binary_search_steps - 1:
                CONST = upper_bound

            prev = 1e10
            for iteration in range(max_iter):

                output, loss, l2dist, newimg = criterion(self.model, y_target_t, modifier, x_t, self.clip_max, self.clip_min, const_t)
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
                    tor = batchlab_tor[e]
                    score = np.dot(tor, sc) / (np.linalg.norm(tor) * np.linalg.norm(sc))
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

        return o_bestattack

def criterion(model, y, modifier, x_t, clip_max, clip_min, const):
    newimg = (torch.tanh(modifier + x_t) + 1) / 2
    newimg = newimg * (clip_max - clip_min) + clip_min

    output = model(newimg)
    # distance to the input data
    other = (torch.tanh(x_t) + 1) / \
                 2 * (clip_max - clip_min) + clip_min
    l2dist = torch.sum((newimg - other).pow(2), (1,2,3))

    y_i = torch.eq(y, torch.ones_like(y))
    y_not_i = torch.eq(y, -(torch.ones_like(y)))
    omega_pos = output * y_i
    omega_neg = output * y_not_i

    omega_neg_max = torch.max(omega_neg, 1).values
    omega_pos_temp = omega_pos
    omega_pos_temp[omega_pos_temp==0] = 1
    omega_pos_min = torch.min(omega_pos_temp, 1).values

    # get indices to check
    loss1 = torch.max(torch.zeros_like(omega_neg_max), omega_neg_max - omega_pos_min)
    # sum up the losses
    loss2 = torch.sum(l2dist)
    loss1 = torch.sum(const * loss1)
    loss = loss1 + loss2

    return output, loss, l2dist, newimg