#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

entity_base = "data/target-kg/v01"


def get_torchbiggraph_config():

    config = dict(
        entity_path=entity_base,

        entities={
            'all': {'num_partitions': 1},
        },

        relations=[{
            'name': 'all_edges',
            'lhs': 'all',
            'rhs': 'all',
            'operator': 'translation',
        }],

        dynamic_relations=True,
        edge_paths=[],
        global_emb=False,
        
        eval_fraction=0.01,
    )

    return config
