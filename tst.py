import tensorflow as tf



next_img, next_label = iterator.get_next()
image_splits = tf.split(next_img, num_gpus)
label_splits = tf.split(next_label, num_gpus)
tower_grads = []
tower_loss = []
counter = 0
for d in gpu_id:
    with tf.device('/gpu:%s' % d):
        with tf.name_scope('%s_%s' % ('tower', d)):
            cross_entropy = build_train_model(image_splits[counter], label_splits[counter], for_training=True)
            counter += 1
            with tf.variable_scope("loss"):
                grads = opt.compute_gradients(cross_entropy)
                tower_grads.append(grads)
                tower_loss.append(cross_entropy)
                tf.get_variable_scope().reuse_variables()

mean_loss = tf.stack(axis=0, values=tower_loss)
mean_loss = tf.reduce_mean(mean_loss, 0)
mean_grads = util.average_gradients(tower_grads)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = opt.apply_gradients(mean_grads, global_step=global_step)
