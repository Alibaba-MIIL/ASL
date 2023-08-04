import tensorflow as tf

def AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0): # Wrapper for ASL function
  """"
  Tensorflow adaptation of "Official Pytorch Implementation of: 'Asymmetric Loss For Multi-Label Classification'(ICCV, 2021) paper" --> https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
  Returns a loss function with asymmetric, specifiable emphases for false negatives & false positives. Output can be passed in as loss function for model.compile().
  ----------
  Parameters
  ----------
  gamma_neg: asymmetric emphasis on false negatives
  gamma_pos: assymetric emphasis on false positives
  """
  
  # Return ASL function with custom emphases
  def ASL_func(y, x):
    """"
    Parameters
    ----------
    x: input logits (y hat)
    y: targets (multi-label binarized vector)
    """

    # Calculating Probabilities
    xs_pos = x
    xs_neg = 1 - x

    # Basic CE calculation
    los_pos = y * tf.math.log(xs_pos)
    los_neg = (1 - y) * tf.math.log(xs_neg)
    loss = los_pos + los_neg

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_gamma = gamma_pos * y + gamma_neg * (1 - y)
        one_sided_w = tf.math.pow(1 - pt, one_sided_gamma)
        loss *= one_sided_w

    return -tf.math.reduce_sum(loss)
  return ASL_func