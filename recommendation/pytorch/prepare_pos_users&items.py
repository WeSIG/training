import multiprocessing as mp
import multiprocessing.dummy
import os
import pickle
import timeit
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import numpy_indexed as npi
# import psutil


CACHE_FN = "alias_tbl_{}x{}_"
NEG_ELEMS_BATCH_SZ = 100000


def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model converter")
    parser.add_argument('data', type=str,
                        help='path to test and training data files')
    parser.add_argument('--valid-negative', type=int, default=999,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('--user_scaling', default=16, type=int)
    parser.add_argument('--item_scaling', default=32, type=int)
    parser.add_argument('--use_sampler_cache', action='store_true',
                        help='Use exiting pre-processed sampler cache. See CACHE_FN variable and use.')
    parser.add_argument('--seed', '-s', default=0, type=int,
                        help='manually set random seed for numpy')
    return parser.parse_args()


def generate_negatives(sampler, num_negatives, users):
    result = []

    neg_users = np.repeat(users, num_negatives)
    num_batches = (neg_users.shape[0] // NEG_ELEMS_BATCH_SZ) + 1
    user_batches = np.array_split(neg_users, num_batches)

    neg_users_items = np.empty([num_negatives], object)
    for i in range(num_batches):
        result.append(sampler.sample_negatives(user_batches[i]))
    result = np.array([neg_users, np.concatenate(result)])
    return result.transpose()


def generate_negatives_flat(sampler, num_negatives, users):
    num_threads = int(0.8 * multiprocessing.cpu_count())
    print(datetime.now(), "Generating negatives using {} threads.".format(num_threads))

    users = np.tile(users, num_negatives)
    users_shape = users.shape

    num_batches = (users.shape[0] // int(1e5)) + 1
    st = timeit.default_timer()
    user_batches = np.array_split(users, num_batches)
    print(".. split users into {} batches, time: {:.2f} sec".format(num_batches, timeit.default_timer() - st))

    # Real multi-processing requires us to move the large sampler object to
    # shared memory. Using threading for now.
    with mp.dummy.Pool(num_threads) as pool:
        results = pool.map(sampler.sample_negatives, user_batches)

    return np.concatenate(results).astype(np.int64)

def process_data(num_items, min_items_per_user, iter_fn):
    user_id = -1
    user_id2 = -1
    positive_users = []
    positive_items = []
    for user_items in iter_fn():
        user_id2 += 1
        if len(user_items) < min_items_per_user:
            continue
        user_id += 1
        user_items = np.sort(user_items)
        positive_users.append(np.ones_like(user_items) * user_id)
        positive_items.append(user_items)
        if user_id % 10000 == 0:
            print("user id {} processed".format(user_id))
    positive_users = np.concatenate(positive_users)
    positive_items = np.concatenate(positive_items)
    print('Generating Alias_Sampler')
    return positive_users, positive_items




def process_raw_data(args):
    train_ratings = [np.array([], dtype=np.int64)] * args.user_scaling  # user_scaling = 16
    test_ratings_chunk = [np.array([], dtype=np.int64)] * args.user_scaling
    test_chunk_size = [0] * args.user_scaling
    for chunk in range(args.user_scaling):
        print(datetime.now(), "Loading data chunk {} of {}".format(chunk + 1, args.user_scaling))
        train_ratings[chunk] = np.load(args.data + '/trainx'
                                       + str(args.user_scaling) + 'x' + str(args.item_scaling)
                                       + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0']
        test_ratings_chunk[chunk] = np.load(args.data + '/testx'
                                            + str(args.user_scaling) + 'x' + str(args.item_scaling)
                                            + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0']
        test_chunk_size[chunk] = test_ratings_chunk[chunk].shape[0]
        #print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)

    # Due to the fractal graph expansion process, some generated users do not
    # have any ratings. Therefore, nb_users should not be max_user_index+1.
    nb_users_per_chunk = [len(np.unique(x[:, 0])) for x in train_ratings]  # (16, 1)
    nb_users = sum(nb_users_per_chunk)
    # nb_users = len(np.unique(train_ratings[:, 0]))

    nb_maxs_per_chunk = [np.max(x, axis=0)[1] for x in train_ratings]  # (16, 1)
    nb_items = max(nb_maxs_per_chunk) + 1  # Zero is valid item in output from expansion

    nb_train_elems = sum([x.shape[0] for x in train_ratings])

    print(datetime.now(), "Number of users: {}, Number of items: {}".format(nb_users,
                                                                            nb_items))  # Number of users: 2197225, Number of items: 855776
    print(datetime.now(), "Number of ratings: {}".format(nb_train_elems))  # Number of ratings: 1223962043

    train_input = [npi.group_by(x[:, 0]).split(x[:, 1]) for x in train_ratings]  # 内存大户，9GB
    #print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    del train_ratings
    del test_ratings_chunk
    #print(u'del后，内存使用：', psutil.Process(os.getpid()).memory_info().rss)

    def iter_fn_simple():
        for train_chunk in train_input:
            for _, items in enumerate(train_chunk):
                yield items

    # TODO Memory error 110GB
    pos_users, pos_items = process_data(
        num_items=nb_items, min_items_per_user=1, iter_fn=iter_fn_simple)
    assert len(pos_users) == nb_train_elems, "Cardinality difference with original data and sample table data."

    # 60GB
    fn_prefix = args.data + '/' + CACHE_FN.format(args.user_scaling, args.item_scaling)
    pos_users_cache = fn_prefix + "cached_pos_users.pkl"
    pos_items_cache = fn_prefix + "cached_pos_items.pkl"

    with open(pos_users_cache, 'wb') as f:
        print('saving pos_users........')
        pickle.dump(pos_users, f, pickle.HIGHEST_PROTOCOL)
        print('done!')
        del pos_users

    with open(pos_items_cache, 'wb') as f:
        print('saving pos_items........')
        pickle.dump(pos_items, f, pickle.HIGHEST_PROTOCOL)
        print('done!')
        del pos_items

    return test_chunk_size


def main():
    args = parse_args()
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        np.random.seed(seed=args.seed)

    if not args.use_sampler_cache:
        test_chunk_size = process_raw_data(args)  # TODO Memory error
    else:
        fn_prefix = args.data + '/' + CACHE_FN.format(args.user_scaling, args.item_scaling)
        sampler_cache = fn_prefix + "cached_sampler.pkl"
        pos_users_cache = fn_prefix + "cached_pos_users.pkl"
        pos_items_cache = fn_prefix + "cached_pos_items.pkl"
        nb_items_cache = fn_prefix + "cached_nb_items.pkl"
        test_chunk_size_cache = fn_prefix + "cached_test_chunk_size.pkl"
        print(datetime.now(), "Loading preprocessed sampler.")
        if os.path.exists(args.data):
            print("Using alias file: {}".format(args.data))
            with open(sampler_cache, "rb") as f:
                sampler = pickle.load(f)
            with open(pos_users_cache, 'rb') as f:
                pos_users = pickle.load(f)
            with open(pos_items_cache, 'rb') as f:
                pos_items = pickle.load(f)
            with open(nb_items_cache, 'rb') as f:
                nb_items = pickle.load(f)
            with open(test_chunk_size_cache, 'rb') as f:
                test_chunk_size = pickle.load(f)



if __name__ == '__main__':
    main()
