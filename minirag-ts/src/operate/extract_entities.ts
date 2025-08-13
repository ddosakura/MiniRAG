import { defaultdict, Counter } from 'collections.js';

import {
  compute_mdhash_id,
  pack_user_ass_to_openai_messages,
  split_string_by_multi_markers,
  clean_str,
  logger,
  encode_string_by_tiktoken,
} from '../utils';
import { GRAPH_FIELD_SEP, PROMPTS } from '../prompt';
import { is_float_regex } from './utils';

async function _handle_entity_relation_summary(
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

async function _handle_single_entity_extraction(
  record_attributes: string[],
  chunk_key: string
): Promise<Record<string, any> | null> {
  if (record_attributes.length < 4 || record_attributes[0] !== '"entity"') {
    return null;
  }
  // add this record as a node in the G
  const entity_name = clean_str(record_attributes[1].toUpperCase());
  if (!entity_name.trim()) {
    return null;
  }
  const entity_type = clean_str(record_attributes[2].toUpperCase());
  const entity_description = clean_str(record_attributes[3]);
  const entity_source_id = chunk_key;
  return {
    entity_name,
    entity_type,
    description: entity_description,
    source_id: entity_source_id,
  };
}

async function _handle_single_relationship_extraction(
  record_attributes: string[],
  chunk_key: string
): Promise<Record<string, any> | null> {
  if (record_attributes.length < 5 || record_attributes[0] !== '"relationship"') {
    return null;
  }
  // add this record as edge
  const source = clean_str(record_attributes[1].toUpperCase());
  const target = clean_str(record_attributes[2].toUpperCase());
  const edge_description = clean_str(record_attributes[3]);

  const edge_keywords = clean_str(record_attributes[4]);
  const edge_source_id = chunk_key;
  const weight = is_float_regex(record_attributes[record_attributes.length - 1]) 
    ? parseFloat(record_attributes[record_attributes.length - 1]) 
    : 1.0;
    
  return {
    src_id: source,
    tgt_id: target,
    weight,
    description: edge_description,
    keywords: edge_keywords,
    source_id: edge_source_id,
  };
}

async function _merge_nodes_then_upsert(
  entity_name: string,
  nodes_data: Record<string, any>[],
  knowledge_graph_inst: any,
  global_config: Record<string, any>
): Promise<Record<string, any>> {
  const already_entitiy_types: string[] = [];
  const already_source_ids: string[] = [];
  const already_description: string[] = [];

  const already_node = await knowledge_graph_inst.get_node(entity_name);
  if (already_node !== null) {
    already_entitiy_types.push(already_node["entity_type"]);
    already_source_ids.push(...split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP]));
    already_description.push(already_node["description"]);
  }

  const entity_type_counter = new Counter();
  for (const dp of nodes_data) {
    entity_type_counter.update([dp["entity_type"]]);
  }
  for (const type of already_entitiy_types) {
    entity_type_counter.update([type]);
  }
  
  const entity_type = [...entity_type_counter.entries()]
    .sort((a, b) => b[1] - a[1])[0][0];

  const description = [...new Set([...nodes_data.map(dp => dp["description"]), ...already_description])]
    .sort()
    .join(GRAPH_FIELD_SEP);
    
  const source_id = [...new Set([...nodes_data.map(dp => dp["source_id"]), ...already_source_ids])]
    .join(GRAPH_FIELD_SEP);

  const node_data = {
    entity_type,
    description,
    source_id,
  };
  
  await knowledge_graph_inst.upsert_node(
    entity_name,
    node_data,
  );
  
  return {
    ...node_data,
    entity_name,
  };
}

async function _merge_edges_then_upsert(
  src_id: string,
  tgt_id: string,
  edges_data: Record<string, any>[],
  knowledge_graph_inst: any,
  global_config: Record<string, any>
): Promise<Record<string, any>> {
  const already_weights: number[] = [];
  const already_source_ids: string[] = [];
  const already_description: string[] = [];
  const already_keywords: string[] = [];

  if (await knowledge_graph_inst.has_edge(src_id, tgt_id)) {
    const already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id);
    already_weights.push(already_edge["weight"]);
    already_source_ids.push(...split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP]));
    already_description.push(already_edge["description"]);
    already_keywords.push(...split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP]));
  }

  const weight = edges_data.map(dp => dp["weight"]).reduce((a, b) => a + b, 0) + 
    already_weights.reduce((a, b) => a + b, 0);
    
  const description = [...new Set([...edges_data.map(dp => dp["description"]), ...already_description])]
    .sort()
    .join(GRAPH_FIELD_SEP);
    
  const keywords = [...new Set([...edges_data.map(dp => dp["keywords"]), ...already_keywords])]
    .sort()
    .join(GRAPH_FIELD_SEP);
    
  const source_id = [...new Set([...edges_data.map(dp => dp["source_id"]), ...already_source_ids])]
    .join(GRAPH_FIELD_SEP);
    
  for (const need_insert_id of [src_id, tgt_id]) {
    if (!(await knowledge_graph_inst.has_node(need_insert_id))) {
      await knowledge_graph_inst.upsert_node(
        need_insert_id,
        {
          "source_id": source_id,
          "description": description,
          "entity_type": '"UNKNOWN"',
        },
      );
    }
  }
  
  await knowledge_graph_inst.upsert_edge(
    src_id,
    tgt_id,
    {
      weight,
      description,
      keywords,
      source_id,
    },
  );

  return {
    src_id,
    tgt_id,
    description,
    keywords,
  };
}

export async function extract_entities(
  chunks: Record<string, any>,
  knowledge_graph_inst: any,
  entity_vdb: any,
  entity_name_vdb: any,
  relationships_vdb: any,
  global_config: Record<string, any>
): Promise<any> {
  const use_llm_func: Function = global_config["llm_model_func"];
  const entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"];

  const ordered_chunks = Object.entries(chunks);
  const entity_extract_prompt = PROMPTS["entity_extraction"];

  const context_base = {
    tuple_delimiter: PROMPTS["DEFAULT_TUPLE_DELIMITER"],
    record_delimiter: PROMPTS["DEFAULT_RECORD_DELIMITER"],
    completion_delimiter: PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    entity_types: PROMPTS["DEFAULT_ENTITY_TYPES"].join(","),
  };
  
  const continue_prompt = PROMPTS["entiti_continue_extraction"];
  const if_loop_prompt = PROMPTS["entiti_if_loop_extraction"];

  let already_processed = 0;
  let already_entities = 0;
  let already_relations = 0;

  async function _process_single_content(chunk_key_dp: [string, Record<string, any>]): Promise<[Record<string, any[]>, Record<string, any[]>]> {
    const chunk_key = chunk_key_dp[0];
    const chunk_dp = chunk_key_dp[1];
    const content = chunk_dp["content"];
    const hint_prompt = entity_extract_prompt.replace("{input_text}", content)
      .replace("{tuple_delimiter}", context_base.tuple_delimiter)
      .replace("{record_delimiter}", context_base.record_delimiter)
      .replace("{completion_delimiter}", context_base.completion_delimiter)
      .replace("{entity_types}", context_base.entity_types);
      
    let final_result = await use_llm_func(hint_prompt);

    let history = pack_user_ass_to_openai_messages(hint_prompt, final_result);
    for (let now_glean_index = 0; now_glean_index < entity_extract_max_gleaning; now_glean_index++) {
      const glean_result = await use_llm_func(continue_prompt, null, history);

      history = [...history, ...pack_user_ass_to_openai_messages(continue_prompt, glean_result)];
      final_result += glean_result;
      if (now_glean_index === entity_extract_max_gleaning - 1) {
        break;
      }

      const if_loop_result: string = await use_llm_func(if_loop_prompt, null, history);
      const cleanResult = if_loop_result.trim().replace(/^["']|["']$/g, '').toLowerCase();
      if (cleanResult !== "yes") {
        break;
      }
    }

    const records = split_string_by_multi_markers(
      final_result,
      [context_base.record_delimiter, context_base.completion_delimiter],
    );

    const maybe_nodes: Record<string, any[]> = defaultdict(() => []);
    const maybe_edges: Record<string, any[]> = defaultdict(() => []);
    
    for (const record of records) {
      const match = record.match(/\((.*)\)/);
      if (match === null) {
        continue;
      }
      const record_content = match[1];
      const record_attributes = split_string_by_multi_markers(
        record_content, 
        [context_base.tuple_delimiter]
      );
      
      const if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key);
      if (if_entities !== null) {
        maybe_nodes[if_entities["entity_name"]].push(if_entities);
        continue;
      }

      const if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key);
      if (if_relation !== null) {
        const key = `${if_relation["src_id"]},${if_relation["tgt_id"]}`;
        maybe_edges[key].push(if_relation);
      }
    }
    
    already_processed += 1;
    already_entities += Object.keys(maybe_nodes).length;
    already_relations += Object.keys(maybe_edges).length;
    
    const now_ticks = PROMPTS["process_tickers"][already_processed % PROMPTS["process_tickers"].length];
    process.stdout.write(
      `${now_ticks} Processed ${already_processed} chunks, ${already_entities} entities(duplicated), ${already_relations} relations(duplicated)\r`
    );
    
    return [maybe_nodes, maybe_edges];
  }

  // 处理所有内容
  const results = await Promise.all(ordered_chunks.map(_process_single_content));
  console.log(); // 清除进度条
  
  const maybe_nodes: Record<string, any[]> = defaultdict(() => []);
  const maybe_edges: Record<string, any[]> = defaultdict(() => []);
  
  for (const [m_nodes, m_edges] of results) {
    for (const [k, v] of Object.entries(m_nodes)) {
      maybe_nodes[k].push(...v);
    }
    for (const [k, v] of Object.entries(m_edges)) {
      const parts = k.split(',');
      const sortedKey = [parts[0], parts[1]].sort().join(',');
      maybe_edges[sortedKey].push(...v);
    }
  }
  
  const all_entities_data = await Promise.all(
    Object.entries(maybe_nodes).map(([k, v]) => 
      _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
    )
  );
  
  const all_relationships_data = await Promise.all(
    Object.entries(maybe_edges).map(([k, v]) => {
      const [src_id, tgt_id] = k.split(',');
      return _merge_edges_then_upsert(src_id, tgt_id, v, knowledge_graph_inst, global_config);
    })
  );
  
  if (all_entities_data.length === 0) {
    logger.warning("Didn't extract any entities, maybe your LLM is not working");
    return null;
  }
  
  if (all_relationships_data.length === 0) {
    logger.warning("Didn't extract any relationships, maybe your LLM is not working");
    return null;
  }

  if (entity_vdb !== null) {
    const data_for_vdb = Object.fromEntries(
      all_entities_data.map(dp => [
        compute_mdhash_id(dp["entity_name"], "ent-"),
        {
          "content": dp["entity_name"] + dp["description"],
          "entity_name": dp["entity_name"],
        }
      ])
    );
    await entity_vdb.upsert(data_for_vdb);
  }

  if (entity_name_vdb !== null) {
    const data_for_vdb = Object.fromEntries(
      all_entities_data.map(dp => [
        compute_mdhash_id(dp["entity_name"], "Ename-"),
        {
          "content": dp["entity_name"],
          "entity_name": dp["entity_name"],
        }
      ])
    );
    await entity_name_vdb.upsert(data_for_vdb);
  }

  if (relationships_vdb !== null) {
    const data_for_vdb = Object.fromEntries(
      all_relationships_data.map(dp => [
        compute_mdhash_id(dp["src_id"] + dp["tgt_id"], "rel-"),
        {
          "src_id": dp["src_id"],
          "tgt_id": dp["tgt_id"],
          "content": dp["keywords"] + " " + dp["src_id"] + " " + dp["tgt_id"] + " " + dp["description"],
        }
      ])
    );
    await relationships_vdb.upsert(data_for_vdb);
  }

  return knowledge_graph_inst;
}