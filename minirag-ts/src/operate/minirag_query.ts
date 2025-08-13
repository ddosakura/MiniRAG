import { Counter } from 'collections.js';

import {
  list_of_list_to_csv,
  truncate_list_by_token_size,
  logger,
  locate_json_string_body_from_string,
} from '../utils';
import { PROMPTS } from '../prompt';
import {
  path2chunk,
  scorednode2chunk,
  kwd2chunk,
  edge_vote_path,
  cal_path_score_list,
} from './utils';

async function _build_mini_query_context(
  ent_from_query: string[],
  type_keywords: string[],
  originalquery: string,
  knowledge_graph_inst: any,
  entities_vdb: any,
  entity_name_vdb: any,
  relationships_vdb: any,
  chunks_vdb: any,
  text_chunks_db: any,
  embedder: any,
  query_param: any,
): Promise<string | null> {
  const imp_ents: string[] = [];
  const nodes_from_query_list: any[] = [];
  const ent_from_query_dict: Record<string, any[]> = {};

  for (const ent of ent_from_query) {
    ent_from_query_dict[ent] = [];
    const results_node = await entity_name_vdb.query(ent, { top_k: query_param.top_k });
    nodes_from_query_list.push(results_node);
    ent_from_query_dict[ent] = results_node.map(e => e["entity_name"]);
  }

  let candidate_reasoning_path: Record<string, any> = {};

  for (const results_node_list of nodes_from_query_list) {
    const candidate_reasoning_path_new = Object.fromEntries(
      results_node_list.map(key => [
        key["entity_name"], 
        { "Score": key["distance"], "Path": [] }
      ])
    );

    candidate_reasoning_path = {
      ...candidate_reasoning_path,
      ...candidate_reasoning_path_new,
    };
  }
  
  for (const key of Object.keys(candidate_reasoning_path)) {
    candidate_reasoning_path[key]["Path"] = await knowledge_graph_inst.get_neighbors_within_k_hops(key, 2);
    imp_ents.push(key);
  }

  const short_path_entries = Object.entries(candidate_reasoning_path)
    .filter(([_, entry]) => entry["Path"].length < 1);
    
  const sorted_short_path_entries = short_path_entries
    .sort((a, b) => b[1]["Score"] - a[1]["Score"]);
    
  const save_p = Math.max(1, Math.floor(sorted_short_path_entries.length * 0.2));
  const top_short_path_entries = sorted_short_path_entries.slice(0, save_p);
  const top_short_path_dict = Object.fromEntries(top_short_path_entries);
  
  const long_path_entries = Object.entries(candidate_reasoning_path)
    .filter(([_, entry]) => entry["Path"].length >= 1);
  const long_path_dict = Object.fromEntries(long_path_entries);
  
  candidate_reasoning_path = { ...long_path_dict, ...top_short_path_dict };
  
  const node_datas_from_type = await knowledge_graph_inst.get_node_from_types(type_keywords);

  const maybe_answer_list = node_datas_from_type.map(n => n["entity_name"]);
  imp_ents.push(...maybe_answer_list);
  
  const scored_reasoning_path = cal_path_score_list(
    candidate_reasoning_path,
    maybe_answer_list
  );

  const results_edge = await relationships_vdb.query(
    originalquery, 
    { top_k: ent_from_query.length * query_param.top_k }
  );
  
  const goodedge: any[] = [];
  const badedge: any[] = [];
  
  for (const item of results_edge) {
    if (imp_ents.includes(item["src_id"]) || imp_ents.includes(item["tgt_id"])) {
      goodedge.push(item);
    } else {
      badedge.push(item);
    }
  }
  
  const [scored_edged_reasoning_path, pairs_append] = edge_vote_path(
    scored_reasoning_path,
    goodedge
  );
  
  const final_scored_path = await path2chunk(
    scored_edged_reasoning_path,
    knowledge_graph_inst,
    pairs_append,
    originalquery,
    3
  );

  const entites_section_list: any[][] = [];
  
  const entity_names = Object.keys(final_scored_path);
  const node_datas_promises = entity_names.map(entity_name => 
    knowledge_graph_inst.get_node(entity_name)
  );
  const node_datas = await Promise.all(node_datas_promises);
  
  const enriched_node_datas = node_datas.map((n, i) => ({
    ...n,
    entity_name: entity_names[i],
    Score: final_scored_path[entity_names[i]]["Score"]
  }));

  for (const n of enriched_node_datas) {
    entites_section_list.push([
      n["entity_name"],
      n["Score"].toString(),
      n["description"] || "UNKNOWN",
    ]);
  }
  
  const sorted_entities = entites_section_list.sort((a, b) => 
    parseFloat(b[1]) - parseFloat(a[1])
  );
  
  const truncated_entities = truncate_list_by_token_size(
    sorted_entities,
    x => x[2],
    query_param.max_token_for_node_context
  );

  truncated_entities.unshift(["entity", "score", "description"]);
  const entities_context = list_of_list_to_csv(truncated_entities);

  scorednode2chunk(ent_from_query_dict, final_scored_path);

  const results = await chunks_vdb.query(originalquery, { top_k: Math.floor(query_param.top_k / 2) });
  const chunks_ids = results.map(r => r["id"]);
  
  const final_chunk_id = kwd2chunk(
    ent_from_query_dict,
    chunks_ids,
    Math.floor(query_param.top_k / 2)
  );

  if (nodes_from_query_list.length === 0) {
    return null;
  }

  if (results_edge.length === 0) {
    return null;
  }

  const use_text_units_promises = final_chunk_id.map(id => text_chunks_db.get_by_id(id));
  const use_text_units = await Promise.all(use_text_units_promises);
  
  const text_units_section_list = [["id", "content"]];

  use_text_units.forEach((t, i) => {
    if (t !== null) {
      text_units_section_list.push([i.toString(), t["content"]]);
    }
  });
  
  const text_units_context = list_of_list_to_csv(text_units_section_list);

  return `
-----Entities-----
\`\`\`csv
${entities_context}
\`\`\`
-----Sources-----
\`\`\`csv
${text_units_context}
\`\`\`
`;
}

export async function minirag_query(
  query: string,
  knowledge_graph_inst: any,
  entities_vdb: any,
  entity_name_vdb: any,
  relationships_vdb: any,
  chunks_vdb: any,
  text_chunks_db: any,
  embedder: any,
  query_param: any,
  global_config: Record<string, any>
): Promise<string> {
  const use_model_func = global_config["llm_model_func"];
  const kw_prompt_temp = PROMPTS["minirag_query2kwd"];
  
  const [TYPE_POOL, TYPE_POOL_w_CASE] = await knowledge_graph_inst.get_types();
  const kw_prompt = kw_prompt_temp
    .replace("{query}", query)
    .replace("{TYPE_POOL}", TYPE_POOL);
    
  const result = await use_model_func(kw_prompt);

  let type_keywords: string[] = [];
  let entities_from_query: string[] = [];
  
  try {
    const keywords_data = JSON.parse(result);
    type_keywords = keywords_data.answer_type_keywords || [];
    entities_from_query = (keywords_data.entities_from_query || []).slice(0, 5);
  } catch (e) {
    try {
      const cleanedResult = result
        .replace(kw_prompt.slice(0, -1), "")
        .replace("user", "")
        .replace("model", "")
        .trim();
      const jsonPart = "{" + cleanedResult.split("{")[1].split("}")[0] + "}";
      const keywords_data = JSON.parse(jsonPart);
      type_keywords = keywords_data.answer_type_keywords || [];
      entities_from_query = (keywords_data.entities_from_query || []).slice(0, 5);
    } catch (jsonError) {
      console.log(`JSON parsing error: ${jsonError}`);
      return PROMPTS["fail_response"];
    }
  }

  const context = await _build_mini_query_context(
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
  );

  if (query_param.only_need_context) {
    return context || "";
  }
  
  if (context === null) {
    return PROMPTS["fail_response"];
  }

  const sys_prompt_temp = PROMPTS["rag_response"];
  const sys_prompt = sys_prompt_temp
    .replace("{context_data}", context)
    .replace("{response_type}", query_param.response_type);
    
  const response = await use_model_func(
    query,
    sys_prompt,
  );

  return response;
}