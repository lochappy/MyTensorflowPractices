"""
Adapted from https://github.com/VisualComputingInstitute/triplet-reid/blob/master/loss.py
"""
import numbers
import tensorflow as tf


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))
cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
]


def get_at_indices(tensor, indices):
    """ Like `tensor[np.arange(len(tensor)), indices]` in numpy. """
    counter = tf.range(tf.shape(indices, out_type=indices.dtype)[0])
    return tf.gather_nd(tensor, tf.stack((counter, indices), -1))


def batch_hard(dists, pids, margin, batch_precision_at_k=None):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        margin: The value of the margin if a number, alternatively the string
            'soft' for using the soft-margin formulation, or `None` for not
            using a margin at all.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """
    with tf.name_scope("batch_hard"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_max(dists*tf.cast(positive_mask, tf.float32), axis=1)
        #def reduceMinWithMask(x):
        #    x[0].set_shape([None])
        #    x[1].set_shape([None])
        #    return tf.reduce_min(tf.boolean_mask(x[0], x[1]))
        #closest_negative = tf.map_fn(reduceMinWithMask,(dists, negative_mask), tf.float32)
        #closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),(dists, negative_mask), tf.float32)
        # Another way of achieving the same, though more hacky:
        closest_negative = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = furthest_positive - closest_negative
        if isinstance(margin, numbers.Real):
            diff = tf.maximum(diff + margin, 0.0)
        elif margin == 'soft':
            diff = tf.nn.softplus(diff)
        elif margin.lower() == 'none':
            pass
        else:
            raise NotImplementedError(
                'The margin {} is not implemented in batch_hard'.format(margin))

    if batch_precision_at_k is None:
        return diff

    # For monitoring, compute the within-batch top-1 accuracy and the
    # within-batch precision-at-k, which is somewhat more expressive.
    with tf.name_scope("monitoring"):
        # This is like argsort along the last axis. Add one to K as we'll
        # drop the diagonal.
        _, indices = tf.nn.top_k(-dists, k=batch_precision_at_k+1)

        # Drop the diagonal (distance to self is always least).
        indices = indices[:,1:]

        # Generate the index indexing into the batch dimension.
        # This is simething like [[0,0,0],[1,1,1],...,[B,B,B]]
        batch_index = tf.tile(
            tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
            (1, tf.shape(indices)[1]))

        # Stitch the above together with the argsort indices to get the
        # indices of the top-k of each row.
        topk_indices = tf.stack((batch_index, indices), -1)

        # See if the topk belong to the same person as they should, or not.
        topk_is_same = tf.gather_nd(same_identity_mask, topk_indices)

        # All of the above could be reduced to the simpler following if k==1
        #top1_is_same = get_at_indices(same_identity_mask, top_idxs[:,1])

        topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
        top1 = tf.reduce_mean(topk_is_same_f32[:,0])
        prec_at_k = tf.reduce_mean(topk_is_same_f32)

        # Finally, let's get some more info that can help in debugging while
        # we're at it!
        negative_dists = tf.boolean_mask(dists, negative_mask)
        positive_dists = tf.boolean_mask(dists, positive_mask)

        return diff, top1, prec_at_k, topk_is_same, negative_dists, positive_dists
    
def normalTriplet(dists, pids, margin, batch_precision_at_k=None):
    """
    Author: lochappy <ttanloc@gmail.com>
    Computes the normal triplet loss between positive and negative pairs

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        margin: The value of the margin if a number, alternatively the string
            'soft' for using the soft-margin formulation, or `None` for not
            using a margin at all.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """
    with tf.name_scope("normalTriplet"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))
        
        def computeDistanceOfPosNegPairWithMargin(x):
            x[0].set_shape([None])
            x[1].set_shape([None])
            x[2].set_shape([None])
            pos_pair_dist = tf.expand_dims(tf.boolean_mask(x[0], x[1]),axis=1)
            neg_pair_dist = tf.expand_dims(tf.boolean_mask(x[0], x[2]),axis=0)
            
            #check if there is no positive pairs in the batch
            noPosPairs = tf.less(tf.shape(pos_pair_dist)[0],1)
            #check if there is no negative pairs in the batch
            noNegPairs = tf.less(tf.shape(neg_pair_dist)[1],1)
            
            def PosNegPairs(): return tf.maximum(pos_pair_dist + margin - neg_pair_dist,0.0)
            def noPosPair(): return tf.maximum(margin - neg_pair_dist,0.0)
            def noNegPair(): return tf.maximum(pos_pair_dist + margin,0.0)
            pos_neg_dist = tf.case({noPosPairs:noPosPair, \
                                    noNegPairs:noNegPair}, \
                                   default=PosNegPairs,exclusive=True)
            
            return tf.reduce_sum(pos_neg_dist)

        def computeDistanceOfPosNegPairWithSoftplus(x):
            x[0].set_shape([None])
            x[1].set_shape([None])
            x[2].set_shape([None])
            pos_pair_dist = tf.expand_dims(tf.boolean_mask(x[0], x[1]),axis=1)
            neg_pair_dist = tf.expand_dims(tf.boolean_mask(x[0], x[2]),axis=0)

            #check if there is no positive pairs in the batch
            noPosPairs = tf.less(tf.shape(pos_pair_dist)[0],1)
            #check if there is no negative pairs in the batch
            noNegPairs = tf.less(tf.shape(neg_pair_dist)[1],1)
            
            def PosNegPairs(): return tf.nn.softplus(pos_pair_dist - neg_pair_dist)
            def noPosPair():   return tf.nn.softplus(-neg_pair_dist)
            def noNegPair():   return tf.nn.softplus(pos_pair_dist)
            pos_neg_dist = tf.case({noPosPairs:noPosPair, \
                                    noNegPairs:noNegPair}, \
                                   default=PosNegPairs,exclusive=True)
            #pos_neg_dist = tf.reduce_sum(tf.nn.softplus(pos_pair_dist - neg_pair_dist ))
            return tf.reduce_sum(pos_neg_dist)
        
        if isinstance(margin, numbers.Real):
            diff = tf.map_fn(computeDistanceOfPosNegPairWithMargin,(dists, positive_mask, negative_mask), tf.float32)
        elif margin == 'soft':
            diff = tf.map_fn(computeDistanceOfPosNegPairWithSoftplus,(dists, positive_mask, negative_mask), tf.float32)
        else:
            raise NotImplementedError(
                    'The margin {} is not implemented in normalTriplet'.format(margin))


    if batch_precision_at_k is None:
        return diff

    # For monitoring, compute the within-batch top-1 accuracy and the
    # within-batch precision-at-k, which is somewhat more expressive.
    with tf.name_scope("monitoring"):
        # This is like argsort along the last axis. Add one to K as we'll
        # drop the diagonal.
        _, indices = tf.nn.top_k(-dists, k=batch_precision_at_k+1)

        # Drop the diagonal (distance to self is always least).
        indices = indices[:,1:]

        # Generate the index indexing into the batch dimension.
        # This is simething like [[0,0,0],[1,1,1],...,[B,B,B]]
        batch_index = tf.tile(
            tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
            (1, tf.shape(indices)[1]))

        # Stitch the above together with the argsort indices to get the
        # indices of the top-k of each row.
        topk_indices = tf.stack((batch_index, indices), -1)

        # See if the topk belong to the same person as they should, or not.
        topk_is_same = tf.gather_nd(same_identity_mask, topk_indices)

        # All of the above could be reduced to the simpler following if k==1
        #top1_is_same = get_at_indices(same_identity_mask, top_idxs[:,1])

        topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
        top1 = tf.reduce_mean(topk_is_same_f32[:,0])
        prec_at_k = tf.reduce_mean(topk_is_same_f32)

        # Finally, let's get some more info that can help in debugging while
        # we're at it!
        negative_dists = tf.boolean_mask(dists, negative_mask)
        positive_dists = tf.boolean_mask(dists, positive_mask)

        return diff, top1, prec_at_k, topk_is_same, negative_dists, positive_dists

def tripletLoss(embeddingFeature,label,margin=-0.5, metric='euclidean',tripletType='normal'):
    """
    Author: lochappy<ttanloc@gmail.com>
    rgs:
        embeddingFeature (2D tensor): Embedding feature vector, shape(B,numfeatures)
        label (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        margin: The value of the margin if a number, alternatively the string
            'soft' for using the soft-margin formulation, or `None` for not
            using a margin at all.
        metric (string): Which distance metric to use. The currently supported metrics are:
                            - 'euclidean', although with a fudge-factor epsilon.
                            - 'sqeuclidean', the squared euclidean.
                            - 'cityblock', the manhattan or L1 distance.
        tripletType (string):which type of triplet loss is used. The currently supported losses
                            are: normal and batch_hard

    Returns:
        A scalar which is the mean loss value of all samples.
    """
    # Create the loss in two steps:
    # 1. Compute all pairwise distances according to the specified metric.
    # 2. For each anchor along the first dimension, compute its loss.
    with tf.name_scope('triplet_loss'):
        dists = cdist(embeddingFeature, embeddingFeature, metric=metric)
        if tripletType == 'normal':
            losses = normalTriplet(dists, label, margin)
        elif tripletType == 'batch_hard':
            losses = batch_hard(dists, label, margin)
        losses = tf.reduce_mean(losses)

    return losses

LOSS_CHOICES = {
    'batch_hard': batch_hard,
}
