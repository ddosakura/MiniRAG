import asyncio
import json
import json_repair
from collections import Counter

from ..utils import (
    list_of_list_to_csv,
    truncate_list_by_token_size,
    logger,
    locate_json_string_body_from_string,
)
from ..prompt import PROMPTS
from .utils import (
    path2chunk,
    scorednode2chunk,
    kwd2chunk,
    edge_vote_path,
    cal_path_score_list,
)


async def _build_mini_query_context(
    ent_from_query,
    type_keywords,
    originalquery,
    knowledge_graph_inst,
    entities_vdb,
    entity_name_vdb,
    relationships_vdb,
    chunks_vdb,
    text_chunks_db,
    embedder,
    query_param,
):
    imp_ents = []
    nodes_from_query_list = []
    ent_from_query_dict = {}

    for ent in ent_from_query:
        ent_from_query_dict[ent] = []
        results_node = await entity_name_vdb.query(ent, top_k=query_param.top_k)

        nodes_from_query_list.append(results_node)
        ent_from_query_dict[ent] = [e["entity_name"] for e in results_node]

    candidate_reasoning_path = {}

    for results_node_list in nodes_from_query_list:
        candidate_reasoning_path_new = {
            key["entity_name"]: {"Score": key["distance"], "Path": []}
            for key in results_node_list
        }

        candidate_reasoning_path = {
            **candidate_reasoning_path,
            **candidate_reasoning_path_new,
        }
    for key in candidate_reasoning_path.keys():
        candidate_reasoning_path[key][
            "Path"
        ] = await knowledge_graph_inst.get_neighbors_within_k_hops(key, 2)
        imp_ents.append(key)

    short_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) < 1
    }
    sorted_short_path_entries = sorted(
        short_path_entries.items(), key=lambda x: x[1]["Score"], reverse=True
    )
    save_p = max(1, int(len(sorted_short_path_entries) * 0.2))
    top_short_path_entries = sorted_short_path_entries[:save_p]
    top_short_path_dict = {name: entry for name, entry in top_short_path_entries}
    long_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) >= 1
    }
    candidate_reasoning_path = {**long_path_entries, **top_short_path_dict}
    node_datas_from_type = await knowledge_graph_inst.get_node_from_types(
        type_keywords
    )  # entity_type, description,...

    maybe_answer_list = [n["entity_name"] for n in node_datas_from_type]
    imp_ents = imp_ents + maybe_answer_list
    scored_reasoning_path = cal_path_score_list(
        candidate_reasoning_path, maybe_answer_list
    )

    results_edge = await relationships_vdb.query(
        originalquery, top_k=len(ent_from_query) * query_param.top_k
    )
    goodedge = []
    badedge = []
    for item in results_edge:
        if item["src_id"] in imp_ents or item["tgt_id"] in imp_ents:
            goodedge.append(item)
        else:
            badedge.append(item)
    scored_edged_reasoning_path, pairs_append = edge_vote_path(
        scored_reasoning_path, goodedge
    )
    scored_edged_reasoning_path = await path2chunk(
        scored_edged_reasoning_path,
        knowledge_graph_inst,
        pairs_append,
        originalquery,
        max_chunks=3,
    )

    entites_section_list = []
    node_datas = await asyncio.gather(
        *[
            knowledge_graph_inst.get_node(entity_name)
            for entity_name in scored_edged_reasoning_path.keys()
        ]
    )
    node_datas = [
        {**n, "entity_name": k, "Score": scored_edged_reasoning_path[k]["Score"]}
        for k, n in zip(scored_edged_reasoning_path.keys(), node_datas)
    ]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                n["entity_name"],
                n["Score"],
                n.get("description", "UNKNOWN"),
            ]
        )
    entites_section_list = sorted(
        entites_section_list, key=lambda x: x[1], reverse=True
    )
    entites_section_list = truncate_list_by_token_size(
        entites_section_list,
        key=lambda x: x[2],
        max_token_size=query_param.max_token_for_node_context,
    )

    entites_section_list.insert(0, ["entity", "score", "description"])
    entities_context = list_of_list_to_csv(entites_section_list)

    scorednode2chunk(ent_from_query_dict, scored_edged_reasoning_path)

    results = await chunks_vdb.query(originalquery, top_k=int(query_param.top_k / 2))
    chunks_ids = [r["id"] for r in results]
    final_chunk_id = kwd2chunk(
        ent_from_query_dict, chunks_ids, chunk_nums=int(query_param.top_k / 2)
    )

    if not len(results_node):
        return None

    if not len(results_edge):
        return None

    use_text_units = await asyncio.gather(
        *[text_chunks_db.get_by_id(id) for id in final_chunk_id]
    )
    text_units_section_list = [["id", "content"]]

    for i, t in enumerate(use_text_units):
        if t is not None:
            text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def minirag_query(
    query,
    knowledge_graph_inst,
    entities_vdb,
    entity_name_vdb,
    relationships_vdb,
    chunks_vdb,
    text_chunks_db,
    embedder,
    query_param,
    global_config: dict,
) -> str:
    use_model_func = global_config["llm_model_func"]
    kw_prompt_temp = PROMPTS["minirag_query2kwd"]
    TYPE_POOL, TYPE_POOL_w_CASE = await knowledge_graph_inst.get_types()
    kw_prompt = kw_prompt_temp.format(query=query, TYPE_POOL=TYPE_POOL)
    result = await use_model_func(kw_prompt)

    try:
        keywords_data = json_repair.loads(result)

        type_keywords = keywords_data.get("answer_type_keywords", [])
        entities_from_query = keywords_data.get("entities_from_query", [])[:5]

    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]

        # Handle parsing error
        except Exception as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]

    context = await _build_mini_query_context(
        entities_from_query,
        type_keywords,
        query,
        knowledge_graph_inst,
        entities_vdb,
        entity_name_vdb,
        relationships_vdb,
        chunks_vdb,
        text_chunks_db,
        embedder,
        query_param,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    return response