from codes.utils.utils import SUPERNET_TAG, encode_block_types, encode_pooling_types


def make_tag(user_config, item_config, pooling_config):
    user_tag = f"user_tower_{user_config['num_layers']}_{user_config['embedding_dim']}_{user_config['block_in_dim']}_{user_config['tower_out_dim']}_{user_config['reg']}" \
               f"_{encode_block_types(user_config['first_layer_blocks'])}_{encode_block_types(user_config['all_layer_blocks'])}"

    item_tag = f"item_tower_{item_config['num_layers']}_{item_config['embedding_dim']}_{item_config['block_in_dim']}_{item_config['tower_out_dim']}_{item_config['reg']}" \
               f"_{encode_block_types(item_config['first_layer_blocks'])}_{encode_block_types(item_config['all_layer_blocks'])}"
               
    pooling_tag = f"pooling_{pooling_config['embedding_dim']}_{pooling_config['seq_fields']}_{pooling_config['seq_len']}_"  \
        f"{encode_pooling_types(pooling_config['types'])}"

    # print(f'{SUPERNET_TAG}_{user_tag}_{item_tag}')
    return f'{SUPERNET_TAG}_{user_tag}_{item_tag}_{pooling_tag}'


CONFIG = {
    "num_layers": 3,
    "all_layer_blocks": [
        "MLP-16", "MLP-32", "MLP-64", "MLP-128", "MLP-256", "MLP-512", "MLP-1024",
        "ElementWise-sum", "ElementWise-avg", "ElementWise-min", "ElementWise-max", "ElementWise-innerproduct",
        "SelfAttention-1", "SelfAttention-2", "SelfAttention-3", "SelfAttention-4"
    ],
    "first_layer_blocks": [
        "MLP-16", "MLP-32", "MLP-64", "MLP-128", "MLP-256", "MLP-512", "MLP-1024",
        "SelfAttention-1", "SelfAttention-2", "SelfAttention-3", "SelfAttention-4"
    ],
    "embedding_dim": 16,
    "block_in_dim": 64,
    "tower_out_dim": 64,
    "reg": 1e-5
}



POOLING_BLOCK = {
    "types": [
        'SumPooling', 'AveragePooling', 'SelfAttentivePooling', 'SelfAttentionPooling'
    ],
    'embedding_dim': 16,
    'seq_fields': 0,
    'seq_len': 0
}
