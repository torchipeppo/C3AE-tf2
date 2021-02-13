'''
una cosa che credevamo servisse, ma non serve
la tengo per non buttarla
'''

# softmax multidimensionale
# https://github.com/tensorflow/tensorflow/issues/210
# TODO se non funziona scrivere e aggiungere @layer
def mdim_softmax(self, target, axis, name=None):
    max_axis = tf.reduce_max(target, axis, keepdims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
    softmax = tf.div(target_exp, normalize, name)
    return softmax