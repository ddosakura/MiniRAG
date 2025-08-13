import asyncio
from typing import Union
from collections import Counter, defaultdict

from ..utils import (
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    list_of_list_to_csv,
    truncate_list_by_token_size,
    split_string_by_multi_markers,
    logger,
    locate_json_string_body_from_string,
    process_combine_contexts,
    clean_str,
    edge_vote_path,
    is_float_regex,
    pack_user_ass_to_openai_messages,
    compute_mdhash_id,
    calculate_similarity,
    cal_path_score_list,
)
from ..prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param,
    text_chunks_db,
    knowledge_graph_inst,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:  # Add check for None edges
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1

            chunk_data = await text_chunks_db.get_by_id(c_id)
            if chunk_data is not None and "content" in chunk_data:  # Add content check
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "relation_counts": relation_counts,
                }

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param,
    knowledge_graph_inst,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param,
    knowledge_graph_inst,
):
    entity_names = set()
    for e in edge_datas:
        entity_names.add(e["src_id"])
        entity_names.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param,
    text_chunks_db,
    knowledge_graph_inst,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units = [t["data"] for t in all_text_units]

    return all_text_units


async def path2chunk(
    scored_edged_reasoning_path, knowledge_graph_inst, pairs_append, query, max_chunks=5
):
    already_node = {}
    for k, v in scored_edged_reasoning_path.items():
        node_chunk_id = None

        for pathtuple, scorelist in v["Path"].items():
            if pathtuple in pairs_append:
                use_edge = pairs_append[pathtuple]
                edge_datas = []
                edge_datas = await asyncio.gather(
                    *[knowledge_graph_inst.get_edge(r[0], r[1]) for r in use_edge]
                )
                text_units = [
                    split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
                    for dp in edge_datas  # chunk ID
                ][0]

            else:
                use_edge = []
                text_units = []

            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(pathtuple[0])]
            )
            for dp in node_datas:
                text_units_node = split_string_by_multi_markers(
                    dp["source_id"], [GRAPH_FIELD_SEP]
                )
                text_units = text_units + text_units_node

            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(ents) for ents in pathtuple[1:]]
            )
            if query is not None:
                for dp in node_datas:
                    text_units_node = split_string_by_multi_markers(
                        dp["source_id"], [GRAPH_FIELD_SEP]
                    )
                    descriptionlist_node = split_string_by_multi_markers(
                        dp["description"], [GRAPH_FIELD_SEP]
                    )
                    if descriptionlist_node[0] not in already_node.keys():
                        already_node[descriptionlist_node[0]] = None

                        if len(text_units_node) == len(descriptionlist_node):
                            if len(text_units_node) > 5:
                                max_ids = int(max(5, len(text_units_node) / 2))
                                should_consider_idx = calculate_similarity(
                                    descriptionlist_node, query, k=max_ids
                                )
                                text_units_node = [
                                    text_units_node[i] for i in should_consider_idx
                                ]
                                already_node[descriptionlist_node[0]] = text_units_node
                    else:
                        text_units_node = already_node[descriptionlist_node[0]]
                    if text_units_node is not None:
                        text_units = text_units + text_units_node

            count_dict = Counter(text_units)
            total_score = scorelist[0] + scorelist[1] + 1
            for key, value in count_dict.items():
                count_dict[key] = value * total_score
            if node_chunk_id is None:
                node_chunk_id = count_dict
            else:
                node_chunk_id = node_chunk_id + count_dict
        v["Path"] = []
        if node_chunk_id is None:
            node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(k)])
            for dp in node_datas:
                text_units_node = split_string_by_multi_markers(
                    dp["source_id"], [GRAPH_FIELD_SEP]
                )
                count_dict = Counter(text_units_node)

            for id in count_dict.most_common(max_chunks):
                v["Path"].append(id[0])
        else:
            for id in count_dict.most_common(max_chunks):
                v["Path"].append(id[0])
    return scored_edged_reasoning_path


def scorednode2chunk(input_dict, values_dict):
    for key, value_list in input_dict.items():
        input_dict[key] = [
            values_dict.get(val, None) for val in value_list if val in values_dict
        ]
        input_dict[key] = [val for val in input_dict[key] if val is not None]


def kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums):
    final_chunk = Counter()
    final_chunk_id = []
    for key, list_of_dicts in ent_from_query_dict.items():
        total_id_scores = Counter()
        id_scores_list = []
        id_scores = {}
        for d in list_of_dicts:
            if d == list_of_dicts[0]:
                score = d["Score"] * 2
            else:
                score = d["Score"]
            path = d["Path"]

            for id in path:
                if id == path[0] and id in chunks_ids:
                    score = score * 10
                if id in id_scores:
                    id_scores[id] += score
                else:
                    id_scores[id] = score
        id_scores_list.append(id_scores)

        for scores in id_scores_list:
            total_id_scores.update(scores)
        final_chunk = final_chunk + total_id_scores  # .most_common(3)

    for i in final_chunk.most_common(chunk_nums):
        final_chunk_id.append(i[0])
    return final_chunk_id