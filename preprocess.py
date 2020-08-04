import argparse
import os
from multiprocessing import cpu_count

from utils.convert_hfdata import convert_to_cui
# from utils.embedding import make_embedding
from utils.grounding import ground
from utils.paths import find_paths, score_paths, prune_paths, generate_path_and_graph_from_adj
from utils.semmed import extract_semmed_cui, construct_graph, construct_subgraph
from utils.graph import extract_subgraph_cui, extract_cui_and_subgraph_from_ground, extract_subgraph_from_path, generate_graph, generate_adj_data_from_grounded_concepts
from utils.triples import generate_triples_from_adj

input_paths = {
    'hfdata': {
        'train': './data/hfdata/hf_training_new.pickle',
        'dev': './data/hfdata/hf_validation_new.pickle',
        'test': './data/hfdata/hf_testing_new.pickle',
        'code2idx': './data/hfdata/hf_code2idx_new.pickle',
    },
    'semmed': {
        'csv': './data/semmed/database.csv',
    },
    'snomed': {
        'snomedct': './data/snomed/SNOMEDCT.txt',
        'icd2snomed_1to1': './data/snomed/ICD9CM_SNOMED_MAP_1TO1.txt',
        'icd2snomed_1toM': './data/snomed/ICD9CM_SNOMED_MAP_1TOM.txt',
    },
    'transe': {
        'ent': './data/transe/glove.transe.sgd.ent.npy',
        'rel': './data/transe/glove.transe.sgd.rel.npy',
    },
}

output_paths = {
    'semmed': {
        'cui-list': './data/semmed/cui_list.txt',
        'subgraph-cui-list': './data/semmed/subgraph_cui_list.txt',
        'subgraph-cui-list-path': './data/semmed/subgraph_cui_list_path.txt',

        'graph': './data/semmed/semmed.unpruned.graph',
        'subgraph': './data/semmed/semmed.subgraph.graph',

        'txt': './data/semmed/graph.txt',
        'sub-txt': './data/semmed/graph.txt',
        'txt-path': './data/semmed/subgraph_path.txt',
    },
    'hfdata': {
        'converted': {
            'train': './data/hfdata/converted/train.jsonl',
            'dev': './data/hfdata/converted/dev.jsonl',
            'test': './data/hfdata/converted/test.jsonl',
            'vocab': './data/hfdata/converted/vocab.json', # haven't change to cui-list (hfdata) don't know where it will be used
        },
        'grounded': {
            'train': './data/hfdata/grounded/train.grounded.jsonl',
            'dev': './data/hfdata/grounded/dev.grounded.jsonl',
            'test': './data/hfdata/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': './data/hfdata/paths/train.paths.raw.jsonl',
            'raw-dev': './data/hfdata/paths/dev.paths.raw.jsonl',
            'raw-test': './data/hfdata/paths/test.paths.raw.jsonl',
            'scores-train': './data/hfdata/paths/train.paths.scores.jsonl',
            'scores-dev': './data/hfdata/paths/dev.paths.scores.jsonl',
            'scores-test': './data/hfdata/paths/test.paths.scores.jsonl',
            'pruned-train': './data/hfdata/paths/train.paths.pruned.jsonl',
            'pruned-dev': './data/hfdata/paths/dev.paths.pruned.jsonl',
            'pruned-test': './data/hfdata/paths/test.paths.pruned.jsonl',
            'adj-train': './data/hfdata/paths/train.paths.adj.jsonl',
            'adj-dev': './data/hfdata/paths/dev.paths.adj.jsonl',
            'adj-test': './data/hfdata/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': './data/hfdata/graph/train.graph.jsonl',
            'dev': './data/hfdata/graph/dev.graph.jsonl',
            'test': './data/hfdata/graph/test.graph.jsonl',
            'adj-train': './data/hfdata/graph/train.graph.adj.pk',
            'adj-dev': './data/hfdata/graph/dev.graph.adj.pk',
            'adj-test': './data/hfdata/graph/test.graph.adj.pk',
            'nxg-from-adj-train': './data/hfdata/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': './data/hfdata/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': './data/hfdata/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': './data/hfdata/triples/train.triples.pk',
            'dev': './data/hfdata/triples/dev.triples.pk',
            'test': './data/hfdata/triples/test.triples.pk',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common', 'hfdata'], choices=['common', 'hfdata',  'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            # {'func': extract_semmed_cui, 'args': (input_paths['semmed']['csv'], output_paths['semmed']['cui-list'])},
            # {'func': construct_graph, 'args': (input_path0s['semmed']['csv'], output_paths['semmed']['cui-list'], output_paths['semmed']['graph'], output_paths['semmed']['txt'])},
        ],
        'hfdata': [
            # {'func': convert_to_cui, 'args': (input_paths['hfdata']['train'], output_paths['hfdata']['converted']['train'], input_paths['hfdata']['code2idx'],
            #                                   input_paths['snomed']['snomedct'], input_paths['snomed']['icd2snomed_1to1'], input_paths['snomed']['icd2snomed_1toM'])},
            # {'func': convert_to_cui, 'args': (input_paths['hfdata']['dev'], output_paths['hfdata']['converted']['dev'], input_paths['hfdata']['code2idx'],
            #                                   input_paths['snomed']['snomedct'], input_paths['snomed']['icd2snomed_1to1'], input_paths['snomed']['icd2snomed_1toM'])},
            # {'func': convert_to_cui, 'args': (input_paths['hfdata']['test'], output_paths['hfdata']['converted']['test'], input_paths['hfdata']['code2idx'],
            #                                   input_paths['snomed']['snomedct'], input_paths['snomed']['icd2snomed_1to1'], input_paths['snomed']['icd2snomed_1toM'])},
            # {'func': make_cui_list, 'args': (output_paths['hfdata']['converted']['train'], output_paths['hfdata']['converted']['cui-list'])},
            # {'func': ground, 'args': (output_paths['hfdata']['converted']['train'], output_paths['semmed']['cui-list'], output_paths['hfdata']['grounded']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['hfdata']['converted']['dev'], output_paths['semmed']['cui-list'], output_paths['hfdata']['grounded']['dev'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['hfdata']['converted']['test'], output_paths['semmed']['cui-list'], output_paths['hfdata']['grounded']['test'], args.nprocs)},

            {'func': extract_cui_and_subgraph_from_ground, 'args': (output_paths['hfdata']['grounded']['train'], output_paths['hfdata']['grounded']['dev'], output_paths['hfdata']['grounded']['test'],
                                                    output_paths['semmed']['graph'], output_paths['semmed']['cui-list'], output_paths['semmed']['subgraph-cui-list'], output_paths['semmed']['sub-txt'])},
            # {'func': construct_subgraph, 'args': (input_paths['semmed']['csv'], output_paths['semmed']['subgraph-cui-list'], output_paths['semmed']['subgraph'], output_paths['semmed']['sub-txt'])},
            # {'func': ground, 'args': (output_paths['hfdata']['converted']['train'], output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['grounded']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['hfdata']['converted']['dev'], output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['grounded']['dev'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['hfdata']['converted']['test'], output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['grounded']['test'], args.nprocs)},

            # {'func': find_paths, 'args': (output_paths['hfdata']['grounded']['train'], output_paths['semmed']['cui-list'],
            #                               output_paths['semmed']['graph'], output_paths['hfdata']['paths']['raw-train'], args.nprocs, args.seed)},
            # {'func': find_paths, 'args': (output_paths['hfdata']['grounded']['dev'], output_paths['semmed']['cui-list'],
            #                               output_paths['semmed']['graph'], output_paths['hfdata']['paths']['raw-dev'], args.nprocs, args.seed)},
            # {'func': find_paths, 'args': (output_paths['hfdata']['grounded']['test'], output_paths['semmed']['cui-list'],
            #                               output_paths['semmed']['graph'], output_paths['hfdata']['paths']['raw-test'], args.nprocs, args.seed)},
            # {'func': extract_subgraph_from_path, 'args': (output_paths['hfdata']['paths']['raw-train'], output_paths['hfdata']['paths']['raw-dev'], output_paths['hfdata']['paths']['raw-test'],
            #                                               output_paths['semmed']['cui-list'], output_paths['semmed']['subgraph-cui-list-path'], output_paths['semmed']['txt-path'])},
            # {'func': score_paths, 'args': (output_paths['hfdata']['paths']['raw-train'], input_paths['transe']['ent'], input_paths['transe']['rel'],
            #                                output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['paths']['scores-train'], args.nprocs)},
            # {'func': score_paths, 'args': (output_paths['hfdata']['paths']['raw-dev'], input_paths['transe']['ent'], input_paths['transe']['rel'],
            #                                output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['paths']['scores-dev'], args.nprocs)},
            # {'func': score_paths, 'args': (output_paths['hfdata']['paths']['raw-test'], input_paths['transe']['ent'], input_paths['transe']['rel'],
            #                                output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['paths']['scores-test'], args.nprocs)},
            # {'func': prune_paths, 'args': (output_paths['hfdata']['paths']['raw-train'], output_paths['hfdata']['paths']['scores-train'],
            #                                output_paths['hfdata']['paths']['pruned-train'], args.path_prune_threshold)},
            # {'func': prune_paths, 'args': (output_paths['hfdata']['paths']['raw-dev'], output_paths['hfdata']['paths']['scores-dev'],
            #                                output_paths['hfdata']['paths']['pruned-dev'], args.path_prune_threshold)},
            # {'func': prune_paths, 'args': (output_paths['hfdata']['paths']['raw-test'], output_paths['hfdata']['paths']['scores-test'],
            #                                output_paths['hfdata']['paths']['pruned-test'], args.path_prune_threshold)},
            # {'func': generate_graph, 'args': (output_paths['hfdata']['grounded']['train'], output_paths['hfdata']['paths']['pruned-train'],
            #                                   output_paths['semmed']['subgraph-cui-list'], output_paths['semmed']['graph'], output_paths['hfdata']['graph']['train'])},
            # {'func': generate_graph, 'args': (output_paths['hfdata']['grounded']['dev'], output_paths['hfdata']['paths']['pruned-dev'],
            #                                   output_paths['semmed']['subgraph-cui-list'], output_paths['semmed']['graph'], output_paths['hfdata']['graph']['dev'])},
            # {'func': generate_graph, 'args': (output_paths['hfdata']['grounded']['test'], output_paths['hfdata']['paths']['pruned-test'],
            #                                   output_paths['semmed']['subgraph-cui-list'], output_paths['semmed']['graph'], output_paths['hfdata']['graph']['test'])},
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['hfdata']['grounded']['train'], output_paths['semmed']['graph'],
            #                                                             output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['graph']['adj-train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['hfdata']['grounded']['dev'], output_paths['semmed']['graph'],
            #                                                             output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['graph']['adj-dev'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['hfdata']['grounded']['test'], output_paths['semmed']['graph'],
            #                                                             output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['graph']['adj-test'], args.nprocs)},
            # {'func': generate_triples_from_adj, 'args': (output_paths['hfdata']['graph']['adj-train'], output_paths['hfdata']['grounded']['train'],
            #                                              output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['triple']['train'])},
            # {'func': generate_triples_from_adj, 'args': (output_paths['hfdata']['graph']['adj-dev'], output_paths['hfdata']['grounded']['dev'],
            #                                              output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['triple']['dev'])},
            # {'func': generate_triples_from_adj, 'args': (output_paths['hfdata']['graph']['adj-test'], output_paths['hfdata']['grounded']['test'],
            #                                              output_paths['semmed']['subgraph-cui-list'], output_paths['hfdata']['triple']['test'])},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['hfdata']['graph']['adj-train'], output_paths['semmed']['graph'], output_paths['hfdata']['paths']['adj-train'], output_paths['hfdata']['graph']['nxg-from-adj-train'], args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['hfdata']['graph']['adj-dev'], output_paths['semmed']['graph'], output_paths['hfdata']['paths']['adj-dev'], output_paths['hfdata']['graph']['nxg-from-adj-dev'], args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['hfdata']['graph']['adj-test'], output_paths['semmed']['graph'], output_paths['hfdata']['paths']['adj-test'], output_paths['hfdata']['graph']['nxg-from-adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    # print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
