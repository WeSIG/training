import multiprocessing as mp
import multiprocessing.dummy
import os
import pickle
import timeit
from argparse import ArgumentParser
from datetime import datetime
import psutil

import numpy as np
import numpy_indexed as npi

from alias_generator import process_data

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
        print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)

    # Due to the fractal graph expansion process, some generated users do not
    # have any ratings. Therefore, nb_users should not be max_user_index+1.
    nb_users_per_chunk = [len(np.unique(x[:, 0])) for x in train_ratings] # (16, 1)
    nb_users = sum(nb_users_per_chunk)
    # nb_users = len(np.unique(train_ratings[:, 0]))

    nb_maxs_per_chunk = [np.max(x, axis=0)[1] for x in train_ratings] # (16, 1)
    nb_items = max(nb_maxs_per_chunk) + 1  # Zero is valid item in output from expansion

    nb_train_elems = sum([x.shape[0] for x in train_ratings])

    print(datetime.now(), "Number of users: {}, Number of items: {}".format(nb_users,
                                                                            nb_items))  # Number of users: 2197225, Number of items: 855776
    print(datetime.now(), "Number of ratings: {}".format(nb_train_elems))  # Number of ratings: 1223962043

    train_input = [npi.group_by(x[:, 0]).split(x[:, 1]) for x in train_ratings] # 内存大户，9GB
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    del train_ratings
    del test_ratings_chunk
    print(u'del后，内存使用：', psutil.Process(os.getpid()).memory_info().rss)

    def iter_fn_simple():
        for train_chunk in train_input:
            for _, items in enumerate(train_chunk):
                yield items

    # TODO Memory error 110GB
    sampler, pos_users, pos_items = process_data(
        num_items=nb_items, min_items_per_user=1, iter_fn=iter_fn_simple)
    assert len(pos_users) == nb_train_elems, "Cardinality difference with original data and sample table data."

    print("pos_users type: {}, pos_items type: {}, s.offsets: {}".format(
        pos_users.dtype, pos_items.dtype, sampler.offsets.dtype))
    print("num_reg: {}, region_card: {}".format(sampler.num_regions.dtype,
                                                sampler.region_cardinality.dtype))
    print("region_starts: {}, alias_index: {}, alias_p: {}".format(
        sampler.region_starts.dtype, sampler.alias_index.dtype,
        sampler.alias_split_p.dtype))
    # 60GB
    fn_prefix = args.data + '/' + CACHE_FN.format(args.user_scaling, args.item_scaling)
    sampler_cache = fn_prefix + "cached_sampler.pkl"
    pos_users_cache = fn_prefix + "cached_pos_users.pkl"
    pos_items_cache = fn_prefix+"cached_pos_items.pkl"
    nb_items_cache = fn_prefix+"cached_nb_items.pkl"
    test_chunk_size_cache = fn_prefix+"cached_test_chunk_size.pkl"

    with open(sampler_cache, "wb") as f:
        pickle.dump([sampler, pos_users, pos_items, nb_items, test_chunk_size], f, pickle.HIGHEST_PROTOCOL)
        del sampler
    with open(pos_users_cache,'wb') as f:
        pickle.dump(pos_users,f,pickle.HIGHEST_PROTOCOL)
        del pos_users
    with open(pos_items_cache,'wb') as f:
        pickle.dump(pos_items,f,pickle.HIGHEST_PROTOCOL)
        del pos_items
    with open(nb_items_cache,'wb') as f:
        pickle.dump(nb_items,f,pickle.HIGHEST_PROTOCOL)
        del nb_items
    with open(test_chunk_size_cache,'wb')as f:
        pickle.dump(test_chunk_size,f,pickle.HIGHEST_PROTOCOL)
        del test_chunk_size


    return sampler, test_chunk_size


def main():
    args = parse_args()
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        np.random.seed(seed=args.seed)

    if not args.use_sampler_cache:
        sampler, test_chunk_size = process_raw_data(args)  # TODO Memory error
    else:
        fn_prefix = args.data + '/' + CACHE_FN.format(args.user_scaling, args.item_scaling)
        sampler_cache = fn_prefix + "cached_sampler.pkl"
        print(datetime.now(), "Loading preprocessed sampler.")
        if os.path.exists(args.data):
            print("Using alias file: {}".format(args.data))
            with open(sampler_cache, "rb") as f:
                sampler, pos_users, pos_items, nb_items, test_chunk_size = pickle.load(f)

    print(datetime.now(), 'Generating negative test samples...')
    test_negatives = [np.array([], dtype=np.int64)] * args.user_scaling
    test_user_offset = 0
    for chunk in range(args.user_scaling):
        neg_users = np.arange(test_user_offset,
                              test_user_offset + test_chunk_size[chunk])  # test_chunk_size: shape (1, 16), including size of every chunk
        test_negatives[chunk] = generate_negatives(
            sampler,
            args.valid_negative, # 999
            neg_users)
        file_name = (args.data + '/test_negx' + str(args.user_scaling) + 'x'
                     + str(args.item_scaling) + '_' + str(chunk) + '.npz')
        np.savez_compressed(file_name, test_negatives[chunk])

        print(datetime.now(), 'Chunk', chunk + 1, 'of', args.user_scaling, 'saved.')
        test_user_offset += test_chunk_size[chunk]

    print(datetime.now(), "Number of test users: {}".format(test_user_offset))


if __name__ == '__main__':
    main()
