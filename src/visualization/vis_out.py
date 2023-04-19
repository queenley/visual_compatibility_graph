"""
This script loads a trained model and tests it for the FITB task.
"""

import json
import argparse
import numpy as np
from skimage.transform import resize
from skimage.color import gray2rgb, rgba2rgb
import skimage.io
from collections import namedtuple
import os
import sys
sys.path.append("./")
from tqdm import tqdm

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from src.utils.utils import get_degree_supports, sparse_to_tuple, normalize_nonsym_adj
from src.utils.utils import construct_feed_dict
from src.models.gnn import CompatibilityGAE
from src.features import DataLoaderPolyvore


def test_fitb(_args):
    _args = namedtuple("Args", _args.keys())(*_args.values())
    load_from = _args.load_from
    config_file = load_from + '/results.json'

    with open(config_file) as f:
        config = json.load(f)

    with open('./src/data/polyvore/dataset/id2idx_test.json') as f:
        id2idx_test = json.load(f)

    with open('./src/data/polyvore/dataset/id2their_test.json') as f:
        id2their_test = json.load(f)

    NUMCLASSES = 2
    BN_AS_TRAIN = False
    ADJ_SELF_CONNECTIONS = True

    def norm_adj(adj_to_norm):
        return normalize_nonsym_adj(adj_to_norm)

    dl = DataLoaderPolyvore()
    train_features, adj_train, train_labels, train_r_indices, train_c_indices = dl.get_phase('train')
    val_features, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase('valid')
    test_features, adj_test, test_labels, test_r_indices, test_c_indices = dl.get_phase('test')
    adj_q, q_r_indices, q_c_indices, q_labels, q_ids, q_valid = dl.get_test_questions()
    train_features, mean, std = dl.normalize_features(train_features, get_moments=True)
    test_features = dl.normalize_features(test_features, mean=mean, std=std)

    train_support = get_degree_supports(adj_train, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    val_support = get_degree_supports(adj_val, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    test_support = get_degree_supports(adj_test, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)
    q_support = get_degree_supports(adj_q, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS)

    for i in range(1, len(train_support)):
        train_support[i] = norm_adj(train_support[i])
        val_support[i] = norm_adj(val_support[i])
        test_support[i] = norm_adj(test_support[i])
        q_support[i] = norm_adj(q_support[i])

    num_support = len(train_support)
    placeholders = {
        'row_indices': tf.placeholder(tf.int32, shape=(None,)),
        'col_indices': tf.placeholder(tf.int32, shape=(None,)),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'weight_decay': tf.placeholder_with_default(0., shape=()),
        'is_train': tf.placeholder_with_default(True, shape=()),
        'support': [tf.sparse_placeholder(tf.float32, shape=(None, None)) for _ in range(num_support)],
        'node_features': tf.placeholder(tf.float32, shape=(None, None)),
        'labels': tf.placeholder(tf.float32, shape=(None,))
    }

    model = CompatibilityGAE(placeholders,
                             input_dim=train_features.shape[1],
                             num_classes=NUMCLASSES,
                             num_support=num_support,
                             hidden=config['hidden'],
                             learning_rate=config['learning_rate'],
                             logging=True,
                             batch_norm=config['batch_norm'])

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def get_image_id(_id):
        K_1 = None
        for k, _v in id2idx_test.items():
            if _v == _id:
                K_1 = k
                break

        return id2their_test[K_1]

    def save_image(_id):
        outfit_id, index = _id.split('_')  # outfitID_index
        images_path = './data/polyvore/images/'
        image_path = images_path + outfit_id + '/' + '{}.jpg'.format(index)
        _im = None
        if os.path.exists(image_path):
            _im = skimage.io.imread(image_path)
            if len(_im.shape) == 2:
                _im = gray2rgb(_im)
            if _im.shape[2] == 4:
                _im = rgba2rgb(_im)
            _im = resize(_im, (256, 256))
        else:
            print(image_path)
        return _im

    with tf.Session() as sess:
        saver.restore(sess, load_from + '/' + 'best_epoch.ckpt')

        kwargs = {'K': _args.k, 'subset': _args.subset,
                  'resampled': _args.resampled, 'expand_outfit': _args.expand_outfit}

        for idx, (question_adj, out_ids, choices_ids, labels, questions) in tqdm(
                enumerate(dl.yield_test_questions_K_edges(**kwargs))):
            shape_choices = choices_ids.shape[0]
            q_support = get_degree_supports(question_adj, config['degree'], adj_self_con=ADJ_SELF_CONNECTIONS,
                                            verbose=False)
            for i in range(1, len(q_support)):
                q_support[i] = norm_adj(q_support[i])
            q_support = [sparse_to_tuple(sup) for sup in q_support]

            q_feed_dict = construct_feed_dict(placeholders, test_features, q_support,
                                              q_labels, out_ids, choices_ids, 0., is_train=BN_AS_TRAIN)

            # compute the output (correct or not) for the current FITB question
            preds = sess.run(model.outputs, feed_dict=q_feed_dict)
            preds = sigmoid(preds)
            outs = preds.reshape((-1, shape_choices))
            outs = outs.mean(axis=0)  # pick the item with average largest probability, averaged accross all edges

            gt = labels.reshape((-1, shape_choices)).mean(axis=0)
            predicted = outs.argmax()
            gt = gt.argmax()

            if not os.path.exists(f"./{_args.result}"):
                os.mkdir(f"./{_args.result}")
            if not os.path.exists(f"./{_args.result}/{idx}"):
                os.mkdir(f"./{_args.result}/{idx}")
            if not os.path.exists(f"./{_args.result}/{idx}/questions"):
                os.mkdir(f"./{_args.result}/{idx}/questions")
            if not os.path.exists(f"./{_args.result}/{idx}/choices"):
                os.mkdir(f"./{_args.result}/{idx}/choices")

            for q in questions:
                id = get_image_id(q)
                if id == get_image_id(choices_ids[predicted]) or id == get_image_id(choices_ids[gt]):
                    print(id)
                im = save_image(id)
                skimage.io.imsave(f"./{_args.result}/{idx}/questions/{id}.png", im)

            for v in np.unique(choices_ids):
                id = get_image_id(v)
                if id == get_image_id(choices_ids[gt]):
                    im = save_image(id)
                    skimage.io.imsave(f"./{_args.result}/{idx}/gt_{id}.png", im)
                if id == get_image_id(choices_ids[predicted]):
                    im = save_image(id)
                    skimage.io.imsave(f"./{_args.result}/{idx}/pd_{id}.png", im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=1,
                        help="K used for the variable number of edges case")
    parser.add_argument('-eo', '--expand_outfit', dest='expand_outfit', action='store_true',
                        help='Expand the outfit nodes as well, rather than using them by default')
    parser.add_argument('-resampled', '--resampled', dest='resampled', action='store_true',
                        help='Runs the test with the resampled FITB tasks (harder)')
    parser.add_argument('-subset', '--subset', dest='subset', action='store_true',
                        help='Use only a subset of the nodes that form the outfit (3 of them) and use the others as '
                             'connections')
    parser.add_argument("-lf", "--load_from", type=str, required=True, default=None, help="Model used.")
    parser.add_argument("--result", type=str, required=True, default="./result", help="path result to save")
    args = parser.parse_args()
    test_fitb(vars(args))
