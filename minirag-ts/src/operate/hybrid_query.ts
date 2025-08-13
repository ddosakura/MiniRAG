import {
  locate_json_string_body_from_string,
  list_of_list_to_csv,
  truncate_list_by_token_size,
  logger,
  process_combine_contexts,
} from '../utils';
import { PROMPTS } from '../prompt';
import {
  _find_most_related_text_unit_from_entities,
  _find_most_related_edges_from_entities,
  _find_most_related_entities_from_relationships,
  _find_related_text_unit_from_relationships,
  chunking_by_token_size,
} from './utils';

async function _build_local_query_context(
  query: string,
  knowledge_graph_inst: any,
  entities_vdb: any,
  text_chunks_db: any,
  query_param: any,
): Promise<string | null> {
  const results = await entities_vdb.query(query, { top_k: query_param.top_k });

  if (results.length === 0) {
    return null;
  }
  
  const node_datas_promises = results.map(r => knowledge_graph_inst.get_node(r["entity_name"]));
  const node_datas = await Promise.all(node_datas_promises);
  
  if (!node_datas.every(n => n !== null)) {
    logger.warning("Some nodes are missing, maybe the storage is damaged");
  }
  
  const node_degrees_promises = results.map(r => knowledge_graph_inst.node_degree(r["entity_name"]));
  const node_degrees = await Promise.all(node_degrees_promises);
  
  const valid_node_datas = results
    .map((k, i) => ({
      ...node_datas[i],
      entity_name: k["entity_name"],
      rank: node_degrees[i]
    }))
    .filter(n => n !== null);

  const use_text_units = await _find_most_related_text_unit_from_entities(
    valid_node_datas,
    query_param,
    text_chunks_db,
    knowledge_graph_inst
  );
  
  const use_relations = await _find_most_related_edges_from_entities(
    valid_node_datas,
    query_param,
    knowledge_graph_inst
  );
  
  logger.info(
    `Local query uses ${valid_node_datas.length} entites, ${use_relations.length} relations, ${use_text_units.length} text units`
  );
  
  const entites_section_list = [["id", "entity", "type", "description", "rank"]];
  valid_node_datas.forEach((n, i) => {
    entites_section_list.push([
      i.toString(),
      n["entity_name"],
      n["entity_type"] || "UNKNOWN",
      n["description"] || "UNKNOWN",
      n["rank"].toString(),
    ]);
  });
  
  const entities_context = list_of_list_to_csv(entites_section_list);

  const relations_section_list = [
    ["id", "source", "target", "description", "keywords", "weight", "rank"]
  ];
  
  use_relations.forEach((e, i) => {
    relations_section_list.push([
      i.toString(),
      e["src_tgt"][0],
      e["src_tgt"][1],
      e["description"],
      e["keywords"],
      e["weight"].toString(),
      e["rank"].toString(),
    ]);
  });
  
  const relations_context = list_of_list_to_csv(relations_section_list);

  const text_units_section_list = [["id", "content"]];
  use_text_units.forEach((t, i) => {
    text_units_section_list.push([i.toString(), t["content"]]);
  });
  
  const text_units_context = list_of_list_to_csv(text_units_section_list);
  
  return `
-----Entities-----
\`\`\`csv
${entities_context}
\`\`\`
-----Relationships-----
\`\`\`csv
${relations_context}
\`\`\`
-----Sources-----
\`\`\`csv
${text_units_context}
\`\`\`
`;
}

async function _build_global_query_context(
  keywords: string,
  knowledge_graph_inst: any,
  entities_vdb: any,
  relationships_vdb: any,
  text_chunks_db: any,
  query_param: any,
): Promise<string | null> {
  const results = await relationships_vdb.query(keywords, { top_k: query_param.top_k });

  if (results.length === 0) {
    return null;
  }

  const edge_datas_promises = results.map(r => 
    knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"])
  );
  const edge_datas = await Promise.all(edge_datas_promises);

  if (!edge_datas.every(n => n !== null)) {
    logger.warning("Some edges are missing, maybe the storage is damaged");
  }
  
  const edge_degree_promises = results.map(r => 
    knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
  );
  const edge_degree = await Promise.all(edge_degree_promises);
  
  const valid_edge_datas = results
    .map((k, i) => ({
      src_id: k["src_id"],
      tgt_id: k["tgt_id"],
      rank: edge_degree[i],
      ...edge_datas[i]
    }))
    .filter(v => v !== null);
  
  const sorted_edge_datas = valid_edge_datas.sort((a, b) => {
    if (b["rank"] !== a["rank"]) return b["rank"] - a["rank"];
    return b["weight"] - a["weight"];
  });
  
  const truncated_edge_datas = truncate_list_by_token_size(
    sorted_edge_datas,
    (x) => x["description"],
    query_param.max_token_for_global_context
  );

  const use_entities = await _find_most_related_entities_from_relationships(
    truncated_edge_datas,
    query_param,
    knowledge_graph_inst
  );
  
  const use_text_units = await _find_related_text_unit_from_relationships(
    truncated_edge_datas,
    query_param,
    text_chunks_db,
    knowledge_graph_inst
  );
  
  logger.info(
    `Global query uses ${use_entities.length} entites, ${truncated_edge_datas.length} relations, ${use_text_units.length} text units`
  );
  
  const relations_section_list = [
    ["id", "source", "target", "description", "keywords", "weight", "rank"]
  ];
  
  truncated_edge_datas.forEach((e, i) => {
    relations_section_list.push([
      i.toString(),
      e["src_id"],
      e["tgt_id"],
      e["description"],
      e["keywords"],
      e["weight"].toString(),
      e["rank"].toString(),
    ]);
  });
  
  const relations_context = list_of_list_to_csv(relations_section_list);

  const entites_section_list = [["id", "entity", "type", "description", "rank"]];
  use_entities.forEach((n, i) => {
    entites_section_list.push([
      i.toString(),
      n["entity_name"],
      n["entity_type"] || "UNKNOWN",
      n["description"] || "UNKNOWN",
      n["rank"].toString(),
    ]);
  });
  
  const entities_context = list_of_list_to_csv(entites_section_list);

  const text_units_section_list = [["id", "content"]];
  use_text_units.forEach((t, i) => {
    text_units_section_list.push([i.toString(), t["content"]]);
  });
  
  const text_units_context = list_of_list_to_csv(text_units_section_list);

  return `
-----Entities-----
\`\`\`csv
${entities_context}
\`\`\`
-----Relationships-----
\`\`\`csv
${relations_context}
\`\`\`
-----Sources-----
\`\`\`csv
${text_units_context}
\`\`\`
`;
}

async function local_query(
  query: string,
  knowledge_graph_inst: any,
  entities_vdb: any,
  relationships_vdb: any,
  text_chunks_db: any,
  query_param: any,
  global_config: Record<string, any>
): Promise<string> {
  let context = null;
  const use_model_func = global_config["llm_model_func"];

  const kw_prompt_temp = PROMPTS["keywords_extraction"];
  const kw_prompt = kw_prompt_temp.replace("{query}", query);
  const result = await use_model_func(kw_prompt);
  const json_text = locate_json_string_body_from_string(result);

  let keywords = "";
  try {
    const keywords_data = JSON.parse(json_text);
    const keywordsList = keywords_data.low_level_keywords || [];
    keywords = keywordsList.join(", ");
  } catch (e) {
    try {
      const cleanedResult = result
        .replace(kw_prompt.slice(0, -1), "")
        .replace("user", "")
        .replace("model", "")
        .trim();
      const jsonPart = "{" + cleanedResult.split("{")[1].split("}")[0] + "}";
      const keywords_data = JSON.parse(jsonPart);
      const keywordsList = keywords_data.low_level_keywords || [];
      keywords = keywordsList.join(", ");
    } catch (jsonError) {
      console.log(`JSON parsing error: ${jsonError}`);
      return PROMPTS["fail_response"];
    }
  }
  
  if (keywords) {
    context = await _build_local_query_context(
      keywords,
      knowledge_graph_inst,
      entities_vdb,
      text_chunks_db,
      query_param,
    );
  }
  
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
    
  let response = await use_model_func(
    query,
    sys_prompt,
  );
  
  if (response.length > sys_prompt.length) {
    response = response
      .replace(sys_prompt, "")
      .replace("user", "")
      .replace("model", "")
      .replace(query, "")
      .replace("<system>", "")
      .replace("</system>", "")
      .trim();
  }

  return response;
}

async function global_query(
  query: string,
  knowledge_graph_inst: any,
  entities_vdb: any,
  relationships_vdb: any,
  text_chunks_db: any,
  query_param: any,
  global_config: Record<string, any>
): Promise<string> {
  let context = null;
  const use_model_func = global_config["llm_model_func"];

  const kw_prompt_temp = PROMPTS["keywords_extraction"];
  const kw_prompt = kw_prompt_temp.replace("{query}", query);
  const result = await use_model_func(kw_prompt);
  const json_text = locate_json_string_body_from_string(result);

  let keywords = "";
  try {
    const keywords_data = JSON.parse(json_text);
    const keywordsList = keywords_data.high_level_keywords || [];
    keywords = keywordsList.join(", ");
  } catch (e) {
    try {
      const cleanedResult = result
        .replace(kw_prompt.slice(0, -1), "")
        .replace("user", "")
        .replace("model", "")
        .trim();
      const jsonPart = "{" + cleanedResult.split("{")[1].split("}")[0] + "}";
      const keywords_data = JSON.parse(jsonPart);
      const keywordsList = keywords_data.high_level_keywords || [];
      keywords = keywordsList.join(", ");
    } catch (jsonError) {
      console.log(`JSON parsing error: ${jsonError}`);
      return PROMPTS["fail_response"];
    }
  }
  
  if (keywords) {
    context = await _build_global_query_context(
      keywords,
      knowledge_graph_inst,
      entities_vdb,
      relationships_vdb,
      text_chunks_db,
      query_param,
    );
  }

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
    
  let response = await use_model_func(
    query,
    sys_prompt,
  );
  
  if (response.length > sys_prompt.length) {
    response = response
      .replace(sys_prompt, "")
      .replace("user", "")
      .replace("model", "")
      .replace(query, "")
      .replace("<system>", "")
      .replace("</system>", "")
      .trim();
  }

  return response;
}

export async function hybrid_query(
  query: string,
  knowledge_graph_inst: any,
  entities_vdb: any,
  relationships_vdb: any,
  text_chunks_db: any,
  query_param: any,
  global_config: Record<string, any>
): Promise<string> {
  let low_level_context = null;
  let high_level_context = null;
  const use_model_func = global_config["llm_model_func"];

  const kw_prompt_temp = PROMPTS["keywords_extraction"];
  const kw_prompt = kw_prompt_temp.replace("{query}", query);

  const result = await use_model_func(kw_prompt);
  const json_text = locate_json_string_body_from_string(result);
  
  let hl_keywords = "";
  let ll_keywords = "";
  
  try {
    const keywords_data = JSON.parse(json_text);
    const hlKeywordsList = keywords_data.high_level_keywords || [];
    const llKeywordsList = keywords_data.low_level_keywords || [];
    hl_keywords = hlKeywordsList.join(", ");
    ll_keywords = llKeywordsList.join(", ");
  } catch (e) {
    try {
      const cleanedResult = result
        .replace(kw_prompt.slice(0, -1), "")
        .replace("user", "")
        .replace("model", "")
        .trim();
      const jsonPart = "{" + cleanedResult.split("{")[1].split("}")[0] + "}";
      const keywords_data = JSON.parse(jsonPart);
      const hlKeywordsList = keywords_data.high_level_keywords || [];
      const llKeywordsList = keywords_data.low_level_keywords || [];
      hl_keywords = hlKeywordsList.join(", ");
      ll_keywords = llKeywordsList.join(", ");
    } catch (jsonError) {
      console.log(`JSON parsing error: ${jsonError}`);
      return PROMPTS["fail_response"];
    }
  }
  
  if (ll_keywords) {
    low_level_context = await _build_local_query_context(
      ll_keywords,
      knowledge_graph_inst,
      entities_vdb,
      text_chunks_db,
      query_param,
    );
  }

  if (hl_keywords) {
    high_level_context = await _build_global_query_context(
      hl_keywords,
      knowledge_graph_inst,
      entities_vdb,
      relationships_vdb,
      text_chunks_db,
      query_param,
    );
  }

  const context = combine_contexts(high_level_context, low_level_context);

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
    
  let response = await use_model_func(
    query,
    sys_prompt,
  );
  
  if (response.length > sys_prompt.length) {
    response = response
      .replace(sys_prompt, "")
      .replace("user", "")
      .replace("model", "")
      .replace(query, "")
      .replace("<system>", "")
      .replace("</system>", "")
      .trim();
  }
  
  return response;
}

export function combine_contexts(high_level_context: string | null, low_level_context: string | null): string {
  // 从上下文字符串中提取实体、关系和源
  function extract_sections(context: string): [string, string, string] {
    const entities_match = context.match(/-----Entities-----\s*```csv\s*(.*?)\s*```/s);
    const relationships_match = context.match(/-----Relationships-----\s*```csv\s*(.*?)\s*```/s);
    const sources_match = context.match(/-----Sources-----\s*```csv\s*(.*?)\s*```/s);

    const entities = entities_match ? entities_match[1] : "";
    const relationships = relationships_match ? relationships_match[1] : "";
    const sources = sources_match ? sources_match[1] : "";

    return [entities, relationships, sources];
  }

  // 从两个上下文中提取部分
  let hl_entities = "", hl_relationships = "", hl_sources = "";
  let ll_entities = "", ll_relationships = "", ll_sources = "";

  if (high_level_context === null) {
    console.warn("High Level context is None. Return empty High entity/relationship/source");
  } else {
    [hl_entities, hl_relationships, hl_sources] = extract_sections(high_level_context);
  }

  if (low_level_context === null) {
    console.warn("Low Level context is None. Return empty Low entity/relationship/source");
  } else {
    [ll_entities, ll_relationships, ll_sources] = extract_sections(low_level_context);
  }

  // 合并并去重实体
  let combined_entities = process_combine_contexts(hl_entities, ll_entities);
  combined_entities = chunking_by_token_size(combined_entities, 128, 2000, "gpt-4o")[0]?.content || "";
  
  // 合并并去重关系
  let combined_relationships = process_combine_contexts(hl_relationships, ll_relationships);
  combined_relationships = chunking_by_token_size(combined_relationships, 128, 2000, "gpt-4o")[0]?.content || "";
  
  // 合并并去重源
  let combined_sources = process_combine_contexts(hl_sources, ll_sources);
  combined_sources = chunking_by_token_size(combined_sources, 128, 2000, "gpt-4o")[0]?.content || "";
  
  // 格式化合并的上下文
  return `
-----Entities-----
\`\`\`csv
${combined_entities}
\`\`\`
-----Relationships-----
\`\`\`csv
${combined_relationships}
\`\`\`
-----Sources-----
\`\`\`csv
${combined_sources}
\`\`\`
`;
}