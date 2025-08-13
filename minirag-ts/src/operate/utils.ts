import { Counter, defaultdict } from 'collections.js';

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
