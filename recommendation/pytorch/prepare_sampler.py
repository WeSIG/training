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

from alias_generator import compute_alias_table, AliasSample

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
    del train_ratings
    del test_ratings_chunk

    def iter_fn_simple():
        for train_chunk in train_input:
            for _, items in enumerate(train_chunk):
                yield items

    # TODO Memory error 110GB
    sampler= process_data(
        num_items=nb_items, min_items_per_user=1, iter_fn=iter_fn_simple)
    print("num_reg: {}, region_card: {}".format(sampler.num_regions.dtype,
                                                sampler.region_cardinality.dtype))
    print("region_starts: {}, alias_index: {}, alias_p: {}".format(
        sampler.region_starts.dtype, sampler.alias_index.dtype,
        sampler.alias_split_p.dtype))
    # 60GB
    fn_prefix = args.data + '/' + CACHE_FN.format(args.user_scaling, args.item_scaling)
    sampler_cache = fn_prefix + "cached_sampler.pkl"
    nb_items_cache = fn_prefix + "cached_nb_items.pkl"
    test_chunk_size_cache = fn_prefix + "cached_test_chunk_size.pkl"

    with open(sampler_cache, "wb") as f:
        # pickle.dump([sampler, pos_users, pos_items, nb_items, test_chunk_size], f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(sampler, f, pickle.HIGHEST_PROTOCOL)
        # del sampler
    with open(nb_items_cache, 'wb') as f:
        pickle.dump(nb_items, f, pickle.HIGHEST_PROTOCOL)
        del nb_items
    with open(test_chunk_size_cache, 'wb')as f:
        pickle.dump(test_chunk_size, f, pickle.HIGHEST_PROTOCOL)
        # del test_chunk_size

    return sampler, test_chunk_size


def process_data(num_items, min_items_per_user, iter_fn):
    user_id = -1
    user_id2 = -1
    num_regions, region_cardinality, region_starts, alias_index, alias_split_p = [], [], [], [], []
    for user_items in iter_fn():
        user_id2 += 1
        if len(user_items) < min_items_per_user:
            continue
        user_id += 1
        user_items = np.sort(user_items)


        bounds = np.concatenate([[-1], user_items, [num_items]])

        neg_region_starts = bounds[:-1] + 1
        neg_region_cardinality = bounds[1:] - bounds[:-1] - 1

        # Handle contiguous positives
        if np.min(neg_region_cardinality) == 0:
            filter_ind = neg_region_cardinality > 0
            neg_region_starts = neg_region_starts[filter_ind]
            neg_region_cardinality = neg_region_cardinality[filter_ind]
        user_alias_index, user_alias_split_p = compute_alias_table(neg_region_cardinality)

        num_regions.append(len(user_alias_index))
        region_cardinality.append(neg_region_cardinality)
        region_starts.append(neg_region_starts)
        alias_index.append(user_alias_index)
        alias_split_p.append(user_alias_split_p)

        if user_id % 10000 == 0:
            print("user id {} processed".format(user_id))

    offsets = np.cumsum([0] + num_regions, dtype=np.int32)[:-1]
    num_regions = np.array(num_regions)
    region_cardinality = np.concatenate(region_cardinality)
    region_starts = np.concatenate(region_starts)
    alias_index = np.concatenate(alias_index)
    alias_split_p = np.concatenate(alias_split_p)
    print('Generating Alias_Sampler')
    return AliasSample(
        offsets=offsets,
        num_regions=num_regions,
        region_cardinality=region_cardinality,
        region_starts=region_starts,
        alias_index=alias_index,
        alias_split_p=alias_split_p,
    )
    # TODO Memory error, 思路：将对应的numpy提前准备好，并del掉对应的list

def main():
    args = parse_args()
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        np.random.seed(seed=args.seed)

    if not args.use_sampler_cache:
        sampler, test_chunk_size = process_raw_data(args)  # TODO Memory error
        print('Done!!')
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
