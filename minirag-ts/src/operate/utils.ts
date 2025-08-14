import { Counter } from '../collections';

import {
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
} from '../utils';
import { GRAPH_FIELD_SEP, PROMPTS } from '../prompt';

export function chunking_by_token_size(
  content: string, 
  overlap_token_size: number = 128, 
  max_token_size: number = 1024, 
  tiktoken_model: string = "gpt-4o"
): Array<{tokens: number, content: string, chunk_order_index: number}> {
  const tokens = encode_string_by_tiktoken(content, tiktoken_model);
  const results = [];
  
  for (let index = 0, start = 0; start < tokens.length; index++, start += max_token_size - overlap_token_size) {
    const chunk_tokens = tokens.slice(start, start + max_token_size);
    const chunk_content = decode_tokens_by_tiktoken(chunk_tokens, tiktoken_model);
    
    results.push({
      tokens: Math.min(max_token_size, tokens.length - start),
      content: chunk_content.trim(),
      chunk_order_index: index,
    });
  }
  
  return results;
}

export async function _handle_entity_relation_summary(
  entity_or_relation_name: string,
  description: string,
  global_config: Record<string, any>
): Promise<string> {
  const tiktoken_model_name = global_config["tiktoken_model_name"];
  const summary_max_tokens = global_config["entity_summary_to_max_tokens"];

  const tokens = encode_string_by_tiktoken(description, tiktoken_model_name);
  if (tokens.length < summary_max_tokens) {  // No need for summary
    return description;
  }
  
  return description; // 原代码中没有实现摘要逻辑，保持一致
}

export async function _find_most_related_text_unit_from_entities(
  node_datas: Array<Record<string, any>>,
  query_param: any,
  text_chunks_db: any,
  knowledge_graph_inst: any
): Promise<Array<Record<string, any>>> {
  const text_units = node_datas.map(dp => 
    split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
  );
  
  const edges_promises = node_datas.map(dp => 
    knowledge_graph_inst.get_node_edges(dp["entity_name"])
  );
  const edges = await Promise.all(edges_promises);
  
  const all_one_hop_nodes = new Set<string>();
  for (const this_edges of edges) {
    if (!this_edges) {
      continue;
    }
    for (const e of this_edges) {
      all_one_hop_nodes.add(e[1]);
    }
  }

  const all_one_hop_nodes_array = Array.from(all_one_hop_nodes);
  const all_one_hop_nodes_data_promises = all_one_hop_nodes_array.map(e => 
    knowledge_graph_inst.get_node(e)
  );
  const all_one_hop_nodes_data = await Promise.all(all_one_hop_nodes_data_promises);

  // 添加节点数据的空值检查
  const all_one_hop_text_units_lookup: Record<string, Set<string>> = {};
  for (let i = 0; i < all_one_hop_nodes_array.length; i++) {
    const k = all_one_hop_nodes_array[i];
    const v = all_one_hop_nodes_data[i];
    if (v !== null && "source_id" in v) {
      all_one_hop_text_units_lookup[k] = new Set(
        split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP])
      );
    }
  }

  const all_text_units_lookup: Record<string, any> = {};
  for (let index = 0; index < text_units.length; index++) {
    const this_text_units = text_units[index];
    const this_edges = edges[index];
    
    for (const c_id of this_text_units) {
      if (c_id in all_text_units_lookup) {
        continue;
      }
      
      let relation_counts = 0;
      if (this_edges) {
        for (const e of this_edges) {
          if (
            e[1] in all_one_hop_text_units_lookup && 
            all_one_hop_text_units_lookup[e[1]].has(c_id)
          ) {
            relation_counts += 1;
          }
        }
      }

      const chunk_data = await text_chunks_db.get_by_id(c_id);
      if (chunk_data !== null && "content" in chunk_data) {
        all_text_units_lookup[c_id] = {
          "data": chunk_data,
          "order": index,
          "relation_counts": relation_counts,
        };
      }
    }
  }

  // 过滤掉空值并确保数据有内容
  const all_text_units = Object.entries(all_text_units_lookup)
    .filter(([_, v]) => 
      v !== null && 
      v["data"] !== null && 
      "content" in v["data"]
    )
    .map(([k, v]) => ({
      id: k,
      ...v
    }));

  if (all_text_units.length === 0) {
    logger.warning("No valid text units found");
    return [];
  }

  const sorted_text_units = all_text_units.sort((a, b) => {
    if (a["order"] !== b["order"]) {
      return a["order"] - b["order"];
    }
    return b["relation_counts"] - a["relation_counts"];
  });

  const truncated_text_units = truncate_list_by_token_size(
    sorted_text_units,
    x => x["data"]["content"],
    query_param.max_token_for_text_unit
  );

  return truncated_text_units.map(t => t["data"]);
}

export async function _find_most_related_edges_from_entities(
  node_datas: Array<Record<string, any>>,
  query_param: any,
  knowledge_graph_inst: any
): Promise<Array<Record<string, any>>> {
  const all_related_edges_promises = node_datas.map(dp => 
    knowledge_graph_inst.get_node_edges(dp["entity_name"])
  );
  const all_related_edges = await Promise.all(all_related_edges_promises);
  
  const all_edges = new Set<string>();
  for (const this_edges of all_related_edges) {
    if (this_edges) {
      for (const e of this_edges) {
        all_edges.add(JSON.stringify([...e].sort()));
      }
    }
  }
  
  const all_edges_array = Array.from(all_edges).map(e => JSON.parse(e));
  
  const all_edges_pack_promises = all_edges_array.map(e => 
    knowledge_graph_inst.get_edge(e[0], e[1])
  );
  const all_edges_pack = await Promise.all(all_edges_pack_promises);
  
  const all_edges_degree_promises = all_edges_array.map(e => 
    knowledge_graph_inst.edge_degree(e[0], e[1])
  );
  const all_edges_degree = await Promise.all(all_edges_degree_promises);
  
  const all_edges_data = all_edges_array
    .map((k, i) => ({
      src_tgt: k,
      rank: all_edges_degree[i],
      ...all_edges_pack[i]
    }))
    .filter(v => v !== null);
  
  const sorted_edges_data = all_edges_data.sort((a, b) => {
    if (b["rank"] !== a["rank"]) {
      return b["rank"] - a["rank"];
    }
    return b["weight"] - a["weight"];
  });
  
  const truncated_edges_data = truncate_list_by_token_size(
    sorted_edges_data,
    x => x["description"],
    query_param.max_token_for_global_context
  );
  
  return truncated_edges_data;
}

export async function _find_most_related_entities_from_relationships(
  edge_datas: Array<Record<string, any>>,
  query_param: any,
  knowledge_graph_inst: any
): Promise<Array<Record<string, any>>> {
  const entity_names = new Set<string>();
  for (const e of edge_datas) {
    entity_names.add(e["src_id"]);
    entity_names.add(e["tgt_id"]);
  }
  
  const entity_names_array = Array.from(entity_names);
  
  const node_datas_promises = entity_names_array.map(entity_name => 
    knowledge_graph_inst.get_node(entity_name)
  );
  const node_datas = await Promise.all(node_datas_promises);
  
  const node_degrees_promises = entity_names_array.map(entity_name => 
    knowledge_graph_inst.node_degree(entity_name)
  );
  const node_degrees = await Promise.all(node_degrees_promises);
  
  const enriched_node_datas = entity_names_array.map((k, i) => ({
    ...node_datas[i],
    entity_name: k,
    rank: node_degrees[i]
  }));
  
  const truncated_node_datas = truncate_list_by_token_size(
    enriched_node_datas,
    x => x["description"],
    query_param.max_token_for_local_context
  );
  
  return truncated_node_datas;
}

export async function _find_related_text_unit_from_relationships(
  edge_datas: Array<Record<string, any>>,
  query_param: any,
  text_chunks_db: any,
  knowledge_graph_inst: any
): Promise<Array<Record<string, any>>> {
  const text_units = edge_datas.map(dp => 
    split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
  );
  
  const all_text_units_lookup: Record<string, any> = {};
  
  for (let index = 0; index < text_units.length; index++) {
    const unit_list = text_units[index];
    for (const c_id of unit_list) {
      if (!(c_id in all_text_units_lookup)) {
        all_text_units_lookup[c_id] = {
          data: await text_chunks_db.get_by_id(c_id),
          order: index
        };
      }
    }
  }
  
  if (Object.values(all_text_units_lookup).some(v => v === null)) {
    logger.warning("Text chunks are missing, maybe the storage is damaged");
  }
  
  const all_text_units = Object.entries(all_text_units_lookup)
    .filter(([_, v]) => v !== null)
    .map(([k, v]) => ({
      id: k,
      ...v
    }));
  
  const sorted_text_units = all_text_units.sort((a, b) => a["order"] - b["order"]);
  
  const truncated_text_units = truncate_list_by_token_size(
    sorted_text_units,
    x => x["data"]["content"],
    query_param.max_token_for_text_unit
  );
  
  return truncated_text_units.map(t => t["data"]);
}

export async function path2chunk(
  scored_edged_reasoning_path: Record<string, any>,
  knowledge_graph_inst: any,
  pairs_append: Record<string, any>,
  query: string | null,
  max_chunks: number = 5
): Promise<Record<string, any>> {
  const already_node: Record<string, Array<string> | null> = {};
  
  for (const [k, v] of Object.entries(scored_edged_reasoning_path)) {
    let node_chunk_id: Counter<string> | null = null;
    
    for (const [pathtuple, scorelist] of Object.entries(v["Path"])) {
      let text_units: Array<string> = [];
      
      if (pathtuple in pairs_append) {
        const use_edge = pairs_append[pathtuple];
        const edge_datas_promises = use_edge.map((r: [string, string]) => 
          knowledge_graph_inst.get_edge(r[0], r[1])
        );
        const edge_datas = await Promise.all(edge_datas_promises);
        
        text_units = split_string_by_multi_markers(edge_datas[0]["source_id"], [GRAPH_FIELD_SEP]);
      }
      
      const node_datas_promises = [knowledge_graph_inst.get_node(pathtuple[0])];
      const node_datas = await Promise.all(node_datas_promises);
      
      for (const dp of node_datas) {
        const text_units_node = split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP]);
        text_units = [...text_units, ...text_units_node];
      }
      
      const path_nodes_promises = Array.from(pathtuple).slice(1).map((ents: string) => 
        knowledge_graph_inst.get_node(ents)
      );
      const path_nodes = await Promise.all(path_nodes_promises);
      
      if (query !== null) {
        for (const dp of path_nodes) {
          const text_units_node = split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP]);
          const descriptionlist_node = split_string_by_multi_markers(dp["description"], [GRAPH_FIELD_SEP]);
          
          if (!(descriptionlist_node[0] in already_node)) {
            already_node[descriptionlist_node[0]] = null;
            
            if (text_units_node.length === descriptionlist_node.length) {
              if (text_units_node.length > 5) {
                const max_ids = Math.max(5, Math.floor(text_units_node.length / 2));
                const should_consider_idx = calculate_similarity(descriptionlist_node, query, undefined, undefined, max_ids);
                const filtered_text_units = should_consider_idx.map(i => text_units_node[i]);
                already_node[descriptionlist_node[0]] = filtered_text_units;
              }
            }
          } else {
            const cached_text_units = already_node[descriptionlist_node[0]];
            if (cached_text_units !== null) {
              text_units = [...text_units, ...cached_text_units];
            }
          }
        }
      }
      
      const count_dict = new Counter<string>();
      for (const unit of text_units) {
        count_dict.add(unit);
      }
      
      const total_score = (scorelist as number[])[0] + (scorelist as number[])[1] + 1;
      for (const [key, value] of count_dict.entries()) {
        count_dict.set(key, value * total_score);
      }
      
      if (node_chunk_id === null) {
        node_chunk_id = count_dict;
      } else {
        for (const [key, value] of count_dict.entries()) {
          node_chunk_id.add(key, value);
        }
      }
    }
    
    v["Path"] = [];
    
    if (node_chunk_id === null) {
      const node_datas_promises = [knowledge_graph_inst.get_node(k)];
      const node_datas = await Promise.all(node_datas_promises);
      
      for (const dp of node_datas) {
        const text_units_node = split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP]);
        const count_dict = new Counter<string>();
        for (const unit of text_units_node) {
          count_dict.add(unit);
        }
        
        for (const [id, _] of count_dict.mostCommon(max_chunks)) {
          v["Path"].push(id);
        }
      }
    } else {
      for (const [id, _] of node_chunk_id.mostCommon(max_chunks)) {
        v["Path"].push(id);
      }
    }
  }
  
  return scored_edged_reasoning_path;
}

export function scorednode2chunk(
  input_dict: Record<string, Array<string>>, 
  values_dict: Record<string, any>
): void {
  for (const [key, value_list] of Object.entries(input_dict)) {
    input_dict[key] = value_list
      .filter(val => val in values_dict)
      .map(val => values_dict[val])
      .filter(val => val !== null);
  }
}

export function kwd2chunk(
  ent_from_query_dict: Record<string, Array<Record<string, any>>>, 
  chunks_ids: Array<string>, 
  chunk_nums: number
): Array<string> {
  const final_chunk = new Counter<string>();
  const final_chunk_id: Array<string> = [];
  
  for (const [key, list_of_dicts] of Object.entries(ent_from_query_dict)) {
    const total_id_scores = new Counter<string>();
    const id_scores_list: Array<Record<string, number>> = [];
    const id_scores: Record<string, number> = {};
    
    for (let i = 0; i < list_of_dicts.length; i++) {
      const d = list_of_dicts[i];
      let score = i === 0 ? d["Score"] * 2 : d["Score"];
      const path = d["Path"];
      
      for (let j = 0; j < path.length; j++) {
        const id = path[j];
        if (j === 0 && chunks_ids.includes(id)) {
          score = score * 10;
        }
        
        if (id in id_scores) {
          id_scores[id] += score;
        } else {
          id_scores[id] = score;
        }
      }
    }
    
    id_scores_list.push(id_scores);
    
    for (const scores of id_scores_list) {
      for (const [id, score] of Object.entries(scores)) {
        total_id_scores.add(id, score);
      }
    }
    
    for (const [id, _] of total_id_scores.entries()) {
      final_chunk.add(id, total_id_scores.get(id));
    }
  }
  
  for (const [id, _] of final_chunk.mostCommon(chunk_nums)) {
    final_chunk_id.push(id);
  }
  
  return final_chunk_id;
}
