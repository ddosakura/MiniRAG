import * as jsonRepair from 'json-repair';
import { Counter } from './collections';
import { 
  BaseGraphStorage, 
  BaseKVStorage, 
  BaseVectorStorage, 
  QueryParam, 
  TextChunkSchema 
} from './base';
import { 
  calculate_similarity,
  cal_path_score_list,
  clean_str,
  decode_tokens_by_tiktoken,
  edge_vote_path,
  encode_string_by_tiktoken,
  is_float_regex,
  list_of_list_to_csv,
  locate_json_string_body_from_string,
  logger,
  pack_user_ass_to_openai_messages,
  process_combine_contexts,
  split_string_by_multi_markers,
  truncate_list_by_token_size,
  compute_mdhash_id
} from './utils';
import { GRAPH_FIELD_SEP, PROMPTS } from './prompt';
import * as np from 'numpy-ts';

/**
 * 按令牌大小分块文本
 * @param content 内容
 * @param overlap_token_size 重叠令牌大小
 * @param max_token_size 最大令牌大小
 * @param tiktoken_model tiktoken模型
 * @returns 分块结果
 */
export function chunking_by_token_size(
  content: string, 
  overlap_token_size: number = 128, 
  max_token_size: number = 1024, 
  tiktoken_model: string = "gpt-4o"
): Array<{tokens: number, content: string, chunk_order_index: number}> {
  const tokens = encode_string_by_tiktoken(content, tiktoken_model);
  const results = [];
  
  for (let index = 0; index < tokens.length; index += (max_token_size - overlap_token_size)) {
    const chunk_tokens = tokens.slice(index, index + max_token_size);
    const chunk_content = decode_tokens_by_tiktoken(chunk_tokens, tiktoken_model);
    
    results.push({
      tokens: Math.min(max_token_size, tokens.length - index),
      content: chunk_content.trim(),
      chunk_order_index: Math.floor(index / (max_token_size - overlap_token_size))
    });
  }
  
  return results;
}

/**
 * 处理实体或关系摘要
 * @param entity_or_relation_name 实体或关系名称
 * @param description 描述
 * @param global_config 全局配置
 * @returns 摘要
 */
async function _handle_entity_relation_summary(
  entity_or_relation_name: string,
  description: string,
  global_config: Record<string, any>
): Promise<string> {
  const tiktoken_model_name = global_config.tiktoken_model_name;
  const summary_max_tokens = global_config.entity_summary_to_max_tokens;
  
  const tokens = encode_string_by_tiktoken(description, tiktoken_model_name);
  if (tokens.length < summary_max_tokens) {
    // 无需摘要
    return description;
  }
  
  // 这里应该实现摘要逻辑，但原代码中没有实现，直接返回原描述
  return description;
}

/**
 * 处理单个实体提取
 * @param record_attributes 记录属性
 * @param chunk_key 块键
 * @returns 实体数据
 */
async function _handle_single_entity_extraction(
  record_attributes: string[],
  chunk_key: string
): Promise<Record<string, any> | null> {
  if (record_attributes.length < 4 || record_attributes[0] !== '"entity"') {
    return null;
  }
  
  // 将此记录添加为G中的节点
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
    source_id: entity_source_id
  };
}

/**
 * 处理单个关系提取
 * @param record_attributes 记录属性
 * @param chunk_key 块键
 * @returns 关系数据
 */
async function _handle_single_relationship_extraction(
  record_attributes: string[],
  chunk_key: string
): Promise<Record<string, any> | null> {
  if (record_attributes.length < 5 || record_attributes[0] !== '"relationship"') {
    return null;
  }
  
  // 添加此记录作为边
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
    source_id: edge_source_id
  };
}

/**
 * 合并节点然后更新
 * @param entity_name 实体名称
 * @param nodes_data 节点数据
 * @param knowledge_graph_inst 知识图谱实例
 * @param global_config 全局配置
 * @returns 节点数据
 */
async function _merge_nodes_then_upsert(
  entity_name: string,
  nodes_data: Array<Record<string, any>>,
  knowledge_graph_inst: BaseGraphStorage,
  global_config: Record<string, any>
): Promise<Record<string, any>> {
  const already_entity_types: string[] = [];
  const already_source_ids: string[] = [];
  const already_description: string[] = [];
  
  const already_node = await knowledge_graph_inst.get_node(entity_name);
  if (already_node !== null) {
    already_entity_types.push(already_node.entity_type);
    already_source_ids.push(...split_string_by_multi_markers(already_node.source_id, [GRAPH_FIELD_SEP]));
    already_description.push(already_node.description);
  }
  
  // 计算实体类型频率并选择最常见的
  const entity_types = [...nodes_data.map(dp => dp.entity_type), ...already_entity_types];
  const typeCounter = new Counter<string>();
  entity_types.forEach(type => typeCounter.add(type));
  const entity_type = typeCounter.mostCommon(1)[0][0];
  
  // 合并描述和源ID
  const description = [...new Set([...nodes_data.map(dp => dp.description), ...already_description])].sort().join(GRAPH_FIELD_SEP);
  const source_id = [...new Set([...nodes_data.map(dp => dp.source_id), ...already_source_ids])].join(GRAPH_FIELD_SEP);
  
  // 更新节点数据
  const node_data = {
    entity_type,
    description,
    source_id
  };
  
  await knowledge_graph_inst.upsert_node(entity_name, node_data);
  
  return {
    ...node_data,
    entity_name
  };
}

/**
 * 合并边然后更新
 * @param src_id 源ID
 * @param tgt_id 目标ID
 * @param edges_data 边数据
 * @param knowledge_graph_inst 知识图谱实例
 * @param global_config 全局配置
 * @returns 边数据
 */
async function _merge_edges_then_upsert(
  src_id: string,
  tgt_id: string,
  edges_data: Array<Record<string, any>>,
  knowledge_graph_inst: BaseGraphStorage,
  global_config: Record<string, any>
): Promise<Record<string, any>> {
  const already_weights: number[] = [];
  const already_source_ids: string[] = [];
  const already_description: string[] = [];
  const already_keywords: string[] = [];
  
  if (await knowledge_graph_inst.has_edge(src_id, tgt_id)) {
    const already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id);
    if (already_edge) {
      already_weights.push(already_edge.weight);
      already_source_ids.push(...split_string_by_multi_markers(already_edge.source_id, [GRAPH_FIELD_SEP]));
      already_description.push(already_edge.description);
      already_keywords.push(...split_string_by_multi_markers(already_edge.keywords, [GRAPH_FIELD_SEP]));
    }
  }
  
  // 合并权重、描述、关键词和源ID
  const weight = edges_data.reduce((sum, dp) => sum + dp.weight, 0) + already_weights.reduce((sum, w) => sum + w, 0);
  const description = [...new Set([...edges_data.map(dp => dp.description), ...already_description])].sort().join(GRAPH_FIELD_SEP);
  const keywords = [...new Set([...edges_data.map(dp => dp.keywords), ...already_keywords])].sort().join(GRAPH_FIELD_SEP);
  const source_id = [...new Set([...edges_data.map(dp => dp.source_id), ...already_source_ids])].join(GRAPH_FIELD_SEP);
  
  // 确保节点存在
  for (const need_insert_id of [src_id, tgt_id]) {
    if (!(await knowledge_graph_inst.has_node(need_insert_id))) {
      await knowledge_graph_inst.upsert_node(
        need_insert_id,
        {
          source_id,
          description,
          entity_type: '"UNKNOWN"'
        }
      );
    }
  }
  
  // 更新边
  await knowledge_graph_inst.upsert_edge(
    src_id,
    tgt_id,
    {
      weight,
      description,
      keywords,
      source_id
    }
  );
  
  return {
    src_id,
    tgt_id,
    description,
    keywords
  };
}

/**
 * 提取实体
 * @param chunks 块
 * @param knowledge_graph_inst 知识图谱实例
 * @param entity_vdb 实体向量数据库
 * @param entity_name_vdb 实体名称向量数据库
 * @param relationships_vdb 关系向量数据库
 * @param global_config 全局配置
 * @returns 知识图谱实例
 */
export async function extract_entities(
  chunks: Record<string, TextChunkSchema>,
  knowledge_graph_inst: BaseGraphStorage,
  entity_vdb: BaseVectorStorage,
  entity_name_vdb: BaseVectorStorage,
  relationships_vdb: BaseVectorStorage,
  global_config: Record<string, any>
): Promise<BaseGraphStorage | null> {
  const use_llm_func: (...args: any[]) => Promise<any> = global_config.llm_model_func;
  const entity_extract_max_gleaning = global_config.entity_extract_max_gleaning;
  
  const ordered_chunks = Object.entries(chunks);
  const entity_extract_prompt = PROMPTS.entity_extraction;
  
  const context_base = {
    tuple_delimiter: PROMPTS.DEFAULT_TUPLE_DELIMITER,
    record_delimiter: PROMPTS.DEFAULT_RECORD_DELIMITER,
    completion_delimiter: PROMPTS.DEFAULT_COMPLETION_DELIMITER,
    entity_types: PROMPTS.DEFAULT_ENTITY_TYPES.join(',')
  };
  
  const continue_prompt = PROMPTS.entiti_continue_extraction;
  const if_loop_prompt = PROMPTS.entiti_if_loop_extraction;
  
  let already_processed = 0;
  let already_entities = 0;
  let already_relations = 0;
  
  /**
   * 处理单个内容
   * @param chunk_key_dp 块键和数据对
   * @returns 节点和边
   */
  async function _process_single_content(chunk_key_dp: [string, TextChunkSchema]): Promise<[Record<string, any[]>, Record<string, any[]>]> {
    const chunk_key = chunk_key_dp[0];
    const chunk_dp = chunk_key_dp[1];
    const content = chunk_dp.content;
    
    const hint_prompt = entity_extract_prompt.replace('{input_text}', content)
      .replace('{tuple_delimiter}', context_base.tuple_delimiter)
      .replace('{record_delimiter}', context_base.record_delimiter)
      .replace('{completion_delimiter}', context_base.completion_delimiter)
      .replace('{entity_types}', context_base.entity_types);
    
    let final_result = await use_llm_func(hint_prompt);
    let history = pack_user_ass_to_openai_messages(hint_prompt, final_result);
    
    for (let now_glean_index = 0; now_glean_index < entity_extract_max_gleaning; now_glean_index++) {
      const glean_result = await use_llm_func(continue_prompt, { history_messages: history });
      
      history = [...history, ...pack_user_ass_to_openai_messages(continue_prompt, glean_result)];
      final_result += glean_result;
      
      if (now_glean_index === entity_extract_max_gleaning - 1) {
        break;
      }
      
      let if_loop_result: string = await use_llm_func(if_loop_prompt, { history_messages: history });
      if_loop_result = if_loop_result.trim().replace(/['"]/g, '').toLowerCase();
      
      if (if_loop_result !== "yes") {
        break;
      }
    }
    
    const records = split_string_by_multi_markers(
      final_result,
      [context_base.record_delimiter, context_base.completion_delimiter]
    );
    
    const maybe_nodes: Record<string, any[]> = {};
    const maybe_edges: Record<string, any[]> = {};
    
    for (const record of records) {
      const match = record.match(/\((.*)\)/);
      if (!match) continue;
      
      const record_content = match[1];
      const record_attributes = split_string_by_multi_markers(record_content, [context_base.tuple_delimiter]);
      
      const if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key);
      if (if_entities !== null) {
        if (!maybe_nodes[if_entities.entity_name]) {
          maybe_nodes[if_entities.entity_name] = [];
        }
        maybe_nodes[if_entities.entity_name].push(if_entities);
        continue;
      }
      
      const if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key);
      if (if_relation !== null) {
        const key = `${if_relation.src_id},${if_relation.tgt_id}`;
        if (!maybe_edges[key]) {
          maybe_edges[key] = [];
        }
        maybe_edges[key].push(if_relation);
      }
    }
    
    already_processed++;
    already_entities += Object.keys(maybe_nodes).length;
    already_relations += Object.keys(maybe_edges).length;
    
    const now_ticks = PROMPTS.process_tickers[already_processed % PROMPTS.process_tickers.length];
    process.stdout.write(`${now_ticks} Processed ${already_processed} chunks, ${already_entities} entities(duplicated), ${already_relations} relations(duplicated)\r`);
    
    return [maybe_nodes, maybe_edges];
  }
  
  // 并行处理所有块
  const results = await Promise.all(ordered_chunks.map(_process_single_content));
  console.log(); // 清除进度条
  
  // 合并结果
  const maybe_nodes: Record<string, any[]> = {};
  const maybe_edges: Record<string, any[]> = {};
  
  for (const [m_nodes, m_edges] of results) {
    for (const [k, v] of Object.entries(m_nodes)) {
      if (!maybe_nodes[k]) {
        maybe_nodes[k] = [];
      }
      maybe_nodes[k].push(...v);
    }
    
    for (const [k, v] of Object.entries(m_edges)) {
      const [src, tgt] = k.split(',');
      const sorted_key = [src, tgt].sort().join(',');
      
      if (!maybe_edges[sorted_key]) {
        maybe_edges[sorted_key] = [];
      }
      maybe_edges[sorted_key].push(...v);
    }
  }
  
  // 合并节点和边
  const all_entities_data = await Promise.all(
    Object.entries(maybe_nodes).map(([k, v]) => _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config))
  );
  
  const all_relationships_data = await Promise.all(
    Object.entries(maybe_edges).map(([k, v]) => {
      const [src, tgt] = k.split(',');
      return _merge_edges_then_upsert(src, tgt, v, knowledge_graph_inst, global_config);
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
  
  // 更新向量数据库
  if (entity_vdb !== null) {
    const data_for_vdb: Record<string, any> = {};
    
    for (const dp of all_entities_data) {
      const id = compute_mdhash_id(dp.entity_name, "ent-");
      data_for_vdb[id] = {
        content: dp.entity_name + dp.description,
        entity_name: dp.entity_name
      };
    }
    
    await entity_vdb.upsert(data_for_vdb);
  }
  
  if (entity_name_vdb !== null) {
    const data_for_vdb: Record<string, any> = {};
    
    for (const dp of all_entities_data) {
      const id = compute_mdhash_id(dp.entity_name, "Ename-");
      data_for_vdb[id] = {
        content: dp.entity_name,
        entity_name: dp.entity_name
      };
    }
    
    await entity_name_vdb.upsert(data_for_vdb);
  }
  
  if (relationships_vdb !== null) {
    const data_for_vdb: Record<string, any> = {};
    
    for (const dp of all_relationships_data) {
      const id = compute_mdhash_id(dp.src_id + dp.tgt_id, "rel-");
      data_for_vdb[id] = {
        src_id: dp.src_id,
        tgt_id: dp.tgt_id,
        content: dp.keywords + " " + dp.src_id + " " + dp.tgt_id + " " + dp.description
      };
    }
    
    await relationships_vdb.upsert(data_for_vdb);
  }
  
  return knowledge_graph_inst;
}

/**
 * 本地查询
 * @param query 查询
 * @param knowledge_graph_inst 知识图谱实例
 * @param entities_vdb 实体向量数据库
 * @param relationships_vdb 关系向量数据库
 * @param text_chunks_db 文本块数据库
 * @param query_param 查询参数
 * @param global_config 全局配置
 * @returns 查询结果
 */
export async function local_query(
  query: string,
  knowledge_graph_inst: BaseGraphStorage,
  entities_vdb: BaseVectorStorage,
  relationships_vdb: BaseVectorStorage,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  query_param: QueryParam,
  global_config: Record<string, any>
): Promise<string> {
  let context = null;
  const use_model_func = global_config.llm_model_func;
  
  const kw_prompt_temp = PROMPTS.keywords_extraction;
  const kw_prompt = kw_prompt_temp.replace('{query}', query);
  const result = await use_model_func(kw_prompt);
  const json_text = locate_json_string_body_from_string(result);
  
  let keywords = "";
  
  try {
    const keywords_data = JSON.parse(json_text || "{}");
    const keywordsList = keywords_data.low_level_keywords || [];
    keywords = keywordsList.join(", ");
  } catch (e) {
    try {
      // 尝试修复JSON
      const cleanResult = result
        .replace(kw_prompt, "")
        .replace("user", "")
        .replace("model", "")
        .trim();
      
      const jsonPart = "{" + cleanResult.split("{")[1].split("}")[0] + "}";
      const keywords_data = JSON.parse(jsonPart);
      const keywordsList = keywords_data.low_level_keywords || [];
      keywords = keywordsList.join(", ");
    } catch (e) {
      console.error(`JSON parsing error: ${e}`);
      return PROMPTS.fail_response;
    }
  }
  
  if (keywords) {
    context = await _build_local_query_context(
      keywords,
      knowledge_graph_inst,
      entities_vdb,
      text_chunks_db,
      query_param
    );
  }
  
  if (query_param.only_need_context) {
    return context || "";
  }
  
  if (context === null) {
    return PROMPTS.fail_response;
  }
  
  const sys_prompt_temp = PROMPTS.rag_response;
  const sys_prompt = sys_prompt_temp
    .replace('{context_data}', context)
    .replace('{response_type}', query_param.response_type);
  
  let response = await use_model_func(query, { system_prompt: sys_prompt });
  
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

/**
 * 构建本地查询上下文
 * @param query 查询
 * @param knowledge_graph_inst 知识图谱实例
 * @param entities_vdb 实体向量数据库
 * @param text_chunks_db 文本块数据库
 * @param query_param 查询参数
 * @returns 查询上下文
 */
async function _build_local_query_context(
  query: string,
  knowledge_graph_inst: BaseGraphStorage,
  entities_vdb: BaseVectorStorage,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  query_param: QueryParam
): Promise<string | null> {
  const results = await entities_vdb.query(query, query_param.top_k);
  
  if (results.length === 0) {
    return null;
  }
  
  const node_datas = await Promise.all(
    results.map(r => knowledge_graph_inst.get_node(r.entity_name))
  );
  
  if (!node_datas.every(n => n !== null)) {
    logger.warning("Some nodes are missing, maybe the storage is damaged");
  }
  
  const node_degrees = await Promise.all(
    results.map(r => knowledge_graph_inst.node_degree(r.entity_name))
  );
  
  const validNodeData = node_datas
    .map((n, i) => n !== null ? {
      ...n,
      entity_name: results[i].entity_name,
      rank: node_degrees[i]
    } : null)
    .filter(n => n !== null) as Record<string, any>[];
  
  const use_text_units = await _find_most_related_text_unit_from_entities(
    validNodeData,
    query_param,
    text_chunks_db,
    knowledge_graph_inst
  );
  
  const use_relations = await _find_most_related_edges_from_entities(
    validNodeData,
    query_param,
    knowledge_graph_inst
  );
  
  logger.info(
    `Local query uses ${validNodeData.length} entities, ${use_relations.length} relations, ${use_text_units.length} text units`
  );
  
  // 构建实体部分
  const entites_section_list: any[][] = [["id", "entity", "type", "description", "rank"]];
  
  for (let i = 0; i < validNodeData.length; i++) {
    const n = validNodeData[i];
    entites_section_list.push([
      i,
      n.entity_name,
      n.entity_type || "UNKNOWN",
      n.description || "UNKNOWN",
      n.rank
    ]);
  }
  
  const entities_context = list_of_list_to_csv(entites_section_list);
  
  // 构建关系部分
  const relations_section_list: any[][] = [
    ["id", "source", "target", "description", "keywords", "weight", "rank"]
  ];
  
  for (let i = 0; i < use_relations.length; i++) {
    const e = use_relations[i];
    relations_section_list.push([
      i,
      e.src_tgt[0],
      e.src_tgt[1],
      e.description,
      e.keywords,
      e.weight,
      e.rank
    ]);
  }
  
  const relations_context = list_of_list_to_csv(relations_section_list);
  
  // 构建文本单元部分
  const text_units_section_list: any[][] = [["id", "content"]];
  
  for (let i = 0; i < use_text_units.length; i++) {
    const t = use_text_units[i];
    text_units_section_list.push([i, t.content]);
  }
  
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

/**
 * 查找与实体最相关的文本单元
 * @param node_datas 节点数据
 * @param query_param 查询参数
 * @param text_chunks_db 文本块数据库
 * @param knowledge_graph_inst 知识图谱实例
 * @returns 文本单元
 */
async function _find_most_related_text_unit_from_entities(
  node_datas: Array<Record<string, any>>,
  query_param: QueryParam,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  knowledge_graph_inst: BaseGraphStorage
): Promise<Array<TextChunkSchema>> {
  const text_units = node_datas.map(dp => 
    split_string_by_multi_markers(dp.source_id, [GRAPH_FIELD_SEP])
  );
  
  const edges = await Promise.all(
    node_datas.map(dp => knowledge_graph_inst.get_node_edges(dp.entity_name))
  );
  
  const all_one_hop_nodes = new Set<string>();
  
  for (const this_edges of edges) {
    if (!this_edges) continue;
    
    for (const e of this_edges) {
      all_one_hop_nodes.add(e[1]);
    }
  }
  
  const all_one_hop_nodes_array = Array.from(all_one_hop_nodes);
  const all_one_hop_nodes_data = await Promise.all(
    all_one_hop_nodes_array.map(e => knowledge_graph_inst.get_node(e))
  );
  
  // 添加节点数据的空检查
  const all_one_hop_text_units_lookup: Record<string, Set<string>> = {};
  
  for (let i = 0; i < all_one_hop_nodes_array.length; i++) {
    const k = all_one_hop_nodes_array[i];
    const v = all_one_hop_nodes_data[i];
    
    if (v !== null && v.source_id) {
      all_one_hop_text_units_lookup[k] = new Set(
        split_string_by_multi_markers(v.source_id, [GRAPH_FIELD_SEP])
      );
    }
  }
  
  const all_text_units_lookup: Record<string, any> = {};
  
  for (let index = 0; index < text_units.length; index++) {
    const this_text_units = text_units[index];
    const this_edges = edges[index];
    
    for (const c_id of this_text_units) {
      if (c_id in all_text_units_lookup) continue;
      
      let relation_counts = 0;
      
      if (this_edges) {
        for (const e of this_edges) {
          if (
            e[1] in all_one_hop_text_units_lookup &&
            all_one_hop_text_units_lookup[e[1]].has(c_id)
          ) {
            relation_counts++;
          }
        }
      }
      
      const chunk_data = await text_chunks_db.get_by_id(c_id);
      if (chunk_data !== null && chunk_data.content) {
        all_text_units_lookup[c_id] = {
          data: chunk_data,
          order: index,
          relation_counts
        };
      }
    }
  }
  
  // 过滤掉空值并确保数据有内容
  const all_text_units = Object.entries(all_text_units_lookup)
    .map(([k, v]) => ({
      id: k,
      ...v
    }))
    .filter(v => v.data !== null && v.data.content);
  
  if (all_text_units.length === 0) {
    logger.warning("No valid text units found");
    return [];
  }
  
  // 按顺序和关系计数排序
  all_text_units.sort((a, b) => {
    if (a.order !== b.order) {
      return a.order - b.order;
    }
    return b.relation_counts - a.relation_counts;
  });
  
  // 截断列表以符合令牌大小限制
  const truncated_units = truncate_list_by_token_size(
    all_text_units,
    item => item.data.content,
    query_param.max_token_for_text_unit
  );
  
  return truncated_units.map(t => t.data);
}

/**
 * 查找与实体最相关的边
 * @param node_datas 节点数据
 * @param query_param 查询参数
 * @param knowledge_graph_inst 知识图谱实例
 * @returns 边数据
 */
async function _find_most_related_edges_from_entities(
  node_datas: Array<Record<string, any>>,
  query_param: QueryParam,
  knowledge_graph_inst: BaseGraphStorage
): Promise<Array<Record<string, any>>> {
  const all_related_edges = await Promise.all(
    node_datas.map(dp => knowledge_graph_inst.get_node_edges(dp.entity_name))
  );
  
  const all_edges = new Set<string>();
  
  for (const this_edges of all_related_edges) {
    if (!this_edges) continue;
    
    for (const e of this_edges) {
      const sorted_edge = [e[0], e[1]].sort().join(',');
      all_edges.add(sorted_edge);
    }
  }
  
  const all_edges_array = Array.from(all_edges).map(e => {
    const [src, tgt] = e.split(',');
    return [src, tgt];
  });
  
  const all_edges_pack = await Promise.all(
    all_edges_array.map(e => knowledge_graph_inst.get_edge(e[0], e[1]))
  );
  
  const all_edges_degree = await Promise.all(
    all_edges_array.map(e => knowledge_graph_inst.edge_degree(e[0], e[1]))
  );
  
  const all_edges_data = all_edges_array
    .map((k, i) => {
      const v = all_edges_pack[i];
      const d = all_edges_degree[i];
      
      if (v === null) return null;
      
      return {
        src_tgt: k,
        rank: d,
        ...v
      };
    })
    .filter(item => item !== null) as Array<Record<string, any>>;
  
  // 按等级和权重排序
  all_edges_data.sort((a, b) => {
    if (a.rank !== b.rank) {
      return b.rank - a.rank;
    }
    return b.weight - a.weight;
  });
  
  // 截断列表以符合令牌大小限制
  return truncate_list_by_token_size(
    all_edges_data,
    item => item.description,
    query_param.max_token_for_global_context
  );
}

/**
 * 全局查询
 * @param query 查询
 * @param knowledge_graph_inst 知识图谱实例
 * @param entities_vdb 实体向量数据库
 * @param relationships_vdb 关系向量数据库
 * @param text_chunks_db 文本块数据库
 * @param query_param 查询参数
 * @param global_config 全局配置
 * @returns 查询结果
 */
export async function global_query(
  query: string,
  knowledge_graph_inst: BaseGraphStorage,
  entities_vdb: BaseVectorStorage,
  relationships_vdb: BaseVectorStorage,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  query_param: QueryParam,
  global_config: Record<string, any>
): Promise<string> {
  let context = null;
  const use_model_func = global_config.llm_model_func;
  
  const kw_prompt_temp = PROMPTS.keywords_extraction;
  const kw_prompt = kw_prompt_temp.replace('{query}', query);
  const result = await use_model_func(kw_prompt);
  const json_text = locate_json_string_body_from_string(result);
  
  let keywords = "";
  
  try {
    const keywords_data = JSON.parse(json_text || "{}");
    const keywordsList = keywords_data.high_level_keywords || [];
    keywords = keywordsList.join(", ");
  } catch (e) {
    try {
      // 尝试修复JSON
      const cleanResult = result
        .replace(kw_prompt, "")
        .replace("user", "")
        .replace("model", "")
        .trim();
      
      const jsonPart = "{" + cleanResult.split("{")[1].split("}")[0] + "}";
      const keywords_data = JSON.parse(jsonPart);
      const keywordsList = keywords_data.high_level_keywords || [];
      keywords = keywordsList.join(", ");
    } catch (e) {
      console.error(`JSON parsing error: ${e}`);
      return PROMPTS.fail_response;
    }
  }
  
  if (keywords) {
    context = await _build_global_query_context(
      keywords,
      knowledge_graph_inst,
      entities_vdb,
      relationships_vdb,
      text_chunks_db,
      query_param
    );
  }
  
  if (query_param.only_need_context) {
    return context || "";
  }
  
  if (context === null) {
    return PROMPTS.fail_response;
  }
  
  const sys_prompt_temp = PROMPTS.rag_response;
  const sys_prompt = sys_prompt_temp
    .replace('{context_data}', context)
    .replace('{response_type}', query_param.response_type);
  
  let response = await use_model_func(query, { system_prompt: sys_prompt });
  
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

/**
 * 构建全局查询上下文
 * @param keywords 关键词
 * @param knowledge_graph_inst 知识图谱实例
 * @param entities_vdb 实体向量数据库
 * @param relationships_vdb 关系向量数据库
 * @param text_chunks_db 文本块数据库
 * @param query_param 查询参数
 * @returns 查询上下文
 */
async function _build_global_query_context(
  keywords: string,
  knowledge_graph_inst: BaseGraphStorage,
  entities_vdb: BaseVectorStorage,
  relationships_vdb: BaseVectorStorage,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  query_param: QueryParam
): Promise<string | null> {
  const results = await relationships_vdb.query(keywords, query_param.top_k);
  
  if (results.length === 0) {
    return null;
  }
  
  const edge_datas = await Promise.all(
    results.map(r => knowledge_graph_inst.get_edge(r.src_id, r.tgt_id))
  );
  
  if (!edge_datas.every(n => n !== null)) {
    logger.warning("Some edges are missing, maybe the storage is damaged");
  }
  
  const edge_degree = await Promise.all(
    results.map(r => knowledge_graph_inst.edge_degree(r.src_id, r.tgt_id))
  );
  
  const valid_edge_datas = results
    .map((k, i) => {
      const v = edge_datas[i];
      const d = edge_degree[i];
      
      if (v === null) return null;
      
      return {
        src_id: k.src_id,
        tgt_id: k.tgt_id,
        rank: d,
        ...v
      };
    })
    .filter(item => item !== null) as Array<Record<string, any>>;
  
  // 按等级和权重排序
  valid_edge_datas.sort((a, b) => {
    if (a.rank !== b.rank) {
      return b.rank - a.rank;
    }
    return b.weight - a.weight;
  });
  
  // 截断列表以符合令牌大小限制
  const truncated_edge_datas = truncate_list_by_token_size(
    valid_edge_datas,
    item => item.description,
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
    `Global query uses ${use_entities.length} entities, ${truncated_edge_datas.length} relations, ${use_text_units.length} text units`
  );
  
  // 构建关系部分
  const relations_section_list: any[][] = [
    ["id", "source", "target", "description", "keywords", "weight", "rank"]
  ];
  
  for (let i = 0; i < truncated_edge_datas.length; i++) {
    const e = truncated_edge_datas[i];
    relations_section_list.push([
      i,
      e.src_id,
      e.tgt_id,
      e.description,
      e.keywords,
      e.weight,
      e.rank
    ]);
  }
  
  const relations_context = list_of_list_to_csv(relations_section_list);
  
  // 构建实体部分
  const entites_section_list: any[][] = [["id", "entity", "type", "description", "rank"]];
  
  for (let i = 0; i < use_entities.length; i++) {
    const n = use_entities[i];
    entites_section_list.push([
      i,
      n.entity_name,
      n.entity_type || "UNKNOWN",
      n.description || "UNKNOWN",
      n.rank
    ]);
  }
  
  const entities_context = list_of_list_to_csv(entites_section_list);
  
  // 构建文本单元部分
  const text_units_section_list: any[][] = [["id", "content"]];
  
  for (let i = 0; i < use_text_units.length; i++) {
    const t = use_text_units[i];
    text_units_section_list.push([i, t.content]);
  }
  
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

/**
 * 从关系中查找最相关的实体
 * @param edge_datas 边数据
 * @param query_param 查询参数
 * @param knowledge_graph_inst 知识图谱实例
 * @returns 实体数据
 */
async function _find_most_related_entities_from_relationships(
  edge_datas: Array<Record<string, any>>,
  query_param: QueryParam,
  knowledge_graph_inst: BaseGraphStorage
): Promise<Array<Record<string, any>>> {
  const entity_names = new Set<string>();
  
  for (const e of edge_datas) {
    entity_names.add(e.src_id);
    entity_names.add(e.tgt_id);
  }
  
  const entity_names_array = Array.from(entity_names);
  const node_datas = await Promise.all(
    entity_names_array.map(entity_name => knowledge_graph_inst.get_node(entity_name))
  );
  
  const node_degrees = await Promise.all(
    entity_names_array.map(entity_name => knowledge_graph_inst.node_degree(entity_name))
  );
  
  const valid_node_datas = entity_names_array
    .map((k, i) => {
      const n = node_datas[i];
      const d = node_degrees[i];
      
      if (n === null) return null;
      
      return {
        ...n,
        entity_name: k,
        rank: d
      };
    })
    .filter(item => item !== null) as Array<Record<string, any>>;
  
  // 截断列表以符合令牌大小限制
  return truncate_list_by_token_size(
    valid_node_datas,
    item => item.description,
    query_param.max_token_for_local_context
  );
}

/**
 * 从关系中查找相关的文本单元
 * @param edge_datas 边数据
 * @param query_param 查询参数
 * @param text_chunks_db 文本块数据库
 * @param knowledge_graph_inst 知识图谱实例
 * @returns 文本单元
 */
async function _find_related_text_unit_from_relationships(
  edge_datas: Array<Record<string, any>>,
  query_param: QueryParam,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  knowledge_graph_inst: BaseGraphStorage
): Promise<Array<TextChunkSchema>> {
  const text_units = edge_datas.map(dp => 
    split_string_by_multi_markers(dp.source_id, [GRAPH_FIELD_SEP])
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
    .map(([k, v]) => ({
      id: k,
      ...v
    }))
    .filter(v => v.data !== null);
  
  // 按顺序排序
  all_text_units.sort((a, b) => a.order - b.order);
  
  // 截断列表以符合令牌大小限制
  const truncated_units = truncate_list_by_token_size(
    all_text_units,
    item => item.data.content,
    query_param.max_token_for_text_unit
  );
  
  return truncated_units.map(t => t.data);
}

/**
 * 混合查询
 * @param query 查询
 * @param knowledge_graph_inst 知识图谱实例
 * @param entities_vdb 实体向量数据库
 * @param relationships_vdb 关系向量数据库
 * @param text_chunks_db 文本块数据库
 * @param query_param 查询参数
 * @param global_config 全局配置
 * @returns 查询结果
 */
export async function hybrid_query(
  query: string,
  knowledge_graph_inst: BaseGraphStorage,
  entities_vdb: BaseVectorStorage,
  relationships_vdb: BaseVectorStorage,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  query_param: QueryParam,
  global_config: Record<string, any>
): Promise<string> {
  let low_level_context = null;
  let high_level_context = null;
  const use_model_func = global_config.llm_model_func;
  
  const kw_prompt_temp = PROMPTS.keywords_extraction;
  const kw_prompt = kw_prompt_temp.replace('{query}', query);
  
  const result = await use_model_func(kw_prompt);
  const json_text = locate_json_string_body_from_string(result);
  
  let hl_keywords = "";
  let ll_keywords = "";
  
  try {
    const keywords_data = JSON.parse(json_text || "{}");
    const hl_keywords_list = keywords_data.high_level_keywords || [];
    const ll_keywords_list = keywords_data.low_level_keywords || [];
    hl_keywords = hl_keywords_list.join(", ");
    ll_keywords = ll_keywords_list.join(", ");
  } catch (e) {
    try {
      // 尝试修复JSON
      const cleanResult = result
        .replace(kw_prompt, "")
        .replace("user", "")
        .replace("model", "")
        .trim();
      
      const jsonPart = "{" + cleanResult.split("{")[1].split("}")[0] + "}";
      const keywords_data = JSON.parse(jsonPart);
      const hl_keywords_list = keywords_data.high_level_keywords || [];
      const ll_keywords_list = keywords_data.low_level_keywords || [];
      hl_keywords = hl_keywords_list.join(", ");
      ll_keywords = ll_keywords_list.join(", ");
    } catch (e) {
      console.error(`JSON parsing error: ${e}`);
      return PROMPTS.fail_response;
    }
  }
  
  if (ll_keywords) {
    low_level_context = await _build_local_query_context(
      ll_keywords,
      knowledge_graph_inst,
      entities_vdb,
      text_chunks_db,
      query_param
    );
  }
  
  if (hl_keywords) {
    high_level_context = await _build_global_query_context(
      hl_keywords,
      knowledge_graph_inst,
      entities_vdb,
      relationships_vdb,
      text_chunks_db,
      query_param
    );
  }
  
  const context = combine_contexts(high_level_context, low_level_context);
  
  if (query_param.only_need_context) {
    return context || "";
  }
  
  if (context === null) {
    return PROMPTS.fail_response;
  }
  
  const sys_prompt_temp = PROMPTS.rag_response;
  const sys_prompt = sys_prompt_temp
    .replace('{context_data}', context)
    .replace('{response_type}', query_param.response_type);
  
  let response = await use_model_func(query, { system_prompt: sys_prompt });
  
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

/**
 * 合并上下文
 * @param high_level_context 高级上下文
 * @param low_level_context 低级上下文
 * @returns 合并后的上下文
 */
function combine_contexts(high_level_context: string | null, low_level_context: string | null): string | null {
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
  const combined_entities = process_combine_contexts(hl_entities, ll_entities);
  const processed_entities = chunking_by_token_size(combined_entities, 0, 2000)[0]?.content || combined_entities;
  
  // 合并并去重关系
  const combined_relationships = process_combine_contexts(hl_relationships, ll_relationships);
  const processed_relationships = chunking_by_token_size(combined_relationships, 0, 2000)[0]?.content || combined_relationships;
  
  // 合并并去重源
  const combined_sources = process_combine_contexts(hl_sources, ll_sources);
  const processed_sources = chunking_by_token_size(combined_sources, 0, 2000)[0]?.content || combined_sources;
  
  // 格式化合并后的上下文
  return `
-----Entities-----
\`\`\`csv
${processed_entities}
\`\`\`
-----Relationships-----
\`\`\`csv
${processed_relationships}
\`\`\`
-----Sources-----
\`\`\`csv
${processed_sources}
\`\`\`
`;
}

/**
 * 朴素查询
 * @param query 查询
 * @param chunks_vdb 块向量数据库
 * @param text_chunks_db 文本块数据库
 * @param query_param 查询参数
 * @param global_config 全局配置
 * @returns 查询结果
 */
export async function naive_query(
  query: string,
  chunks_vdb: BaseVectorStorage,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  query_param: QueryParam,
  global_config: Record<string, any>
): Promise<string> {
  const use_model_func = global_config.llm_model_func;
  const results = await chunks_vdb.query(query, query_param.top_k);
  
  if (results.length === 0) {
    return PROMPTS.fail_response;
  }
  
  const chunks_ids = results.map(r => r.id);
  const chunks = await text_chunks_db.get_by_ids(chunks_ids);
  
  const maybe_trun_chunks = truncate_list_by_token_size(
    chunks.filter(c => c !== null) as TextChunkSchema[],
    c => c.content,
    query_param.max_token_for_text_unit
  );
  
  logger.info(`Truncate ${chunks.length} to ${maybe_trun_chunks.length} chunks`);
  
  const section = maybe_trun_chunks.map(c => c.content).join("\n--New Chunk--\n");
  
  if (query_param.only_need_context) {
    return section;
  }
  
  const sys_prompt_temp = PROMPTS.naive_rag_response;
  const sys_prompt = sys_prompt_temp
    .replace('{content_data}', section)
    .replace('{response_type}', query_param.response_type);
  
  let response = await use_model_func(query, { system_prompt: sys_prompt });
  
  if (response.length > sys_prompt.length) {
    response = response
      .substring(sys_prompt.length)
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

/**
 * 路径到块
 * @param scored_edged_reasoning_path 评分边推理路径
 * @param knowledge_graph_inst 知识图谱实例
 * @param pairs_append 附加对
 * @param query 查询
 * @param max_chunks 最大块数
 * @returns 评分边推理路径
 */
export async function path2chunk(
  scored_edged_reasoning_path: Record<string, any>,
  knowledge_graph_inst: BaseGraphStorage,
  pairs_append: Record<string, any[]>,
  query: string | null,
  max_chunks: number = 5
): Promise<Record<string, any>> {
  const already_node: Record<string, string[] | null> = {};
  
  for (const [k, v] of Object.entries(scored_edged_reasoning_path)) {
    let node_chunk_id: Record<string, number> | null = null;
    
    for (const [pathtuple, scorelist] of Object.entries(v.Path)) {
      let text_units: string[] = [];
      
      if (pathtuple in pairs_append) {
        const use_edge = pairs_append[pathtuple];
        const edge_datas = await Promise.all(
          use_edge.map(r => knowledge_graph_inst.get_edge(r[0], r[1]))
        );
        
        if (edge_datas.length > 0 && edge_datas[0] !== null) {
          text_units = split_string_by_multi_markers(edge_datas[0].source_id, [GRAPH_FIELD_SEP]);
        }
      }
      
      // 解析pathtuple字符串为数组
      const pathArray = JSON.parse(pathtuple);
      
      // 获取第一个节点的数据
      const node_datas = await Promise.all([
        knowledge_graph_inst.get_node(pathArray[0])
      ]);
      
      for (const dp of node_datas) {
        if (dp !== null) {
          const text_units_node = split_string_by_multi_markers(dp.source_id, [GRAPH_FIELD_SEP]);
          text_units = [...text_units, ...text_units_node];
        }
      }
      
      // 获取其余节点的数据
      const other_node_datas = await Promise.all(
        pathArray.slice(1).map(ent => knowledge_graph_inst.get_node(ent))
      );
      
      if (query !== null) {
        for (const dp of other_node_datas) {
          if (dp === null) continue;
          
          const text_units_node = split_string_by_multi_markers(dp.source_id, [GRAPH_FIELD_SEP]);
          const descriptionlist_node = split_string_by_multi_markers(dp.description, [GRAPH_FIELD_SEP]);
          
          if (!(descriptionlist_node[0] in already_node)) {
            already_node[descriptionlist_node[0]] = null;
            
            if (text_units_node.length === descriptionlist_node.length) {
              if (text_units_node.length > 5) {
                const max_ids = Math.max(5,
import * as jsonRepair from 'json-repair';
import { Counter } from 'collections';
import { 
  BaseGraphStorage, 
  BaseKVStorage, 
  BaseVectorStorage, 
  QueryParam, 
  TextChunkSchema 
} from './base';
import { 
  calculate_similarity,
  cal_path_score_list,
  clean_str,
  decode_tokens_by_tiktoken,
  edge_vote_path,
  encode_string_by_tiktoken,
  is_float_regex,
  list_of_list_to_csv,
  locate_json_string_body_from_string,
  logger,
  pack_user_ass_to_openai_messages,
  process_combine_contexts,
  split_string_by_multi_markers,
  truncate_list_by_token_size,
  compute_mdhash_id
} from './utils';
import { GRAPH_FIELD_SEP, PROMPTS } from './prompt';
import * as np from 'numpy-ts';

/**
 * 按令牌大小分块文本
 * @param content 内容
 * @param overlap_token_size 重叠令牌大小
 * @param max_token_size 最大令牌大小
 * @param tiktoken_model tiktoken模型
 * @returns 分块结果
 */
export function chunking_by_token_size(
  content: string, 
  overlap_token_size: number = 128, 
  max_token_size: number = 1024, 
  tiktoken_model: string = "gpt-4o"
): Array<{tokens: number, content: string, chunk_order_index: number}> {
  const tokens = encode_string_by_tiktoken(content, tiktoken_model);
  const results = [];
  
  for (let index = 0; index < tokens.length; index += (max_token_size - overlap_token_size)) {
    const chunk_tokens = tokens.slice(index, index + max_token_size);
    const chunk_content = decode_tokens_by_tiktoken(chunk_tokens, tiktoken_model);
    
    results.push({
      tokens: Math.min(max_token_size, tokens.length - index),
      content: chunk_content.trim(),
      chunk_order_index: Math.floor(index / (max_token_size - overlap_token_size))
    });
  }
  
  return results;
}

/**
 * 处理实体或关系摘要
 * @param entity_or_relation_name 实体或关系名称
 * @param description 描述
 * @param global_config 全局配置
 * @returns 摘要
 */
async function _handle_entity_relation_summary(
  entity_or_relation_name: string,
  description: string,
  global_config: Record<string, any>
): Promise<string> {
  const tiktoken_model_name = global_config.tiktoken_model_name;
  const summary_max_tokens = global_config.entity_summary_to_max_tokens;
  
  const tokens = encode_string_by_tiktoken(description, tiktoken_model_name);
  if (tokens.length < summary_max_tokens) {
    // 无需摘要
    return description;
  }
  
  // 这里应该实现摘要逻辑，但原代码中没有实现，直接返回原描述
  return description;
}

/**
 * 处理单个实体提取
 * @param record_attributes 记录属性
 * @param chunk_key 块键
 * @returns 实体数据
 */
async function _handle_single_entity_extraction(
  record_attributes: string[],
  chunk_key: string
): Promise<Record<string, any> | null> {
  if (record_attributes.length < 4 || record_attributes[0] !== '"entity"') {
    return null;
  }
  
  // 将此记录添加为G中的节点
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
    source_id: entity_source_id
  };
}

/**
 * 处理单个关系提取
 * @param record_attributes 记录属性
 * @param chunk_key 块键
 * @returns 关系数据
 */
async function _handle_single_relationship_extraction(
  record_attributes: string[],
  chunk_key: string
): Promise<Record<string, any> | null> {
  if (record_attributes.length < 5 || record_attributes[0] !== '"relationship"') {
    return null;
  }
  
  // 添加此记录作为边
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
    source_id: edge_source_id
  };
}

/**
 * 合并节点然后更新
 * @param entity_name 实体名称
 * @param nodes_data 节点数据
 * @param knowledge_graph_inst 知识图谱实例
 * @param global_config 全局配置
 * @returns 节点数据
 */
async function _merge_nodes_then_upsert(
  entity_name: string,
  nodes_data: Array<Record<string, any>>,
  knowledge_graph_inst: BaseGraphStorage,
  global_config: Record<string, any>
): Promise<Record<string, any>> {
  const already_entity_types: string[] = [];
  const already_source_ids: string[] = [];
  const already_description: string[] = [];
  
  const already_node = await knowledge_graph_inst.get_node(entity_name);
  if (already_node !== null) {
    already_entity_types.push(already_node.entity_type);
    already_source_ids.push(...split_string_by_multi_markers(already_node.source_id, [GRAPH_FIELD_SEP]));
    already_description.push(already_node.description);
  }
  
  // 计算实体类型频率并选择最常见的
  const entity_types = [...nodes_data.map(dp => dp.entity_type), ...already_entity_types];
  const typeCounter = new Counter<string>();
  entity_types.forEach(type => typeCounter.add(type));
  const entity_type = typeCounter.mostCommon(1)[0][0];
  
  // 合并描述和源ID
  const description = [...new Set([...nodes_data.map(dp => dp.description), ...already_description])].sort().join(GRAPH_FIELD_SEP);
  const source_id = [...new Set([...nodes_data.map(dp => dp.source_id), ...already_source_ids])].join(GRAPH_FIELD_SEP);
  
  // 更新节点数据
  const node_data = {
    entity_type,
    description,
    source_id
  };
  
  await knowledge_graph_inst.upsert_node(entity_name, node_data);
  
  return {
    ...node_data,
    entity_name
  };
}

/**
 * 合并边然后更新
 * @param src_id 源ID
 * @param tgt_id 目标ID
 * @param edges_data 边数据
 * @param knowledge_graph_inst 知识图谱实例
 * @param global_config 全局配置
 * @returns 边数据
 */
async function _merge_edges_then_upsert(
  src_id: string,
  tgt_id: string,
  edges_data: Array<Record<string, any>>,
  knowledge_graph_inst: BaseGraphStorage,
  global_config: Record<string, any>
): Promise<Record<string, any>> {
  const already_weights: number[] = [];
  const already_source_ids: string[] = [];
  const already_description: string[] = [];
  const already_keywords: string[] = [];
  
  if (await knowledge_graph_inst.has_edge(src_id, tgt_id)) {
    const already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id);
    if (already_edge) {
      already_weights.push(already_edge.weight);
      already_source_ids.push(...split_string_by_multi_markers(already_edge.source_id, [GRAPH_FIELD_SEP]));
      already_description.push(already_edge.description);
      already_keywords.push(...split_string_by_multi_markers(already_edge.keywords, [GRAPH_FIELD_SEP]));
    }
  }
  
  // 合并权重、描述、关键词和源ID
  const weight = edges_data.reduce((sum, dp) => sum + dp.weight, 0) + already_weights.reduce((sum, w) => sum + w, 0);
  const description = [...new Set([...edges_data.map(dp => dp.description), ...already_description])].sort().join(GRAPH_FIELD_SEP);
  const keywords = [...new Set([...edges_data.map(dp => dp.keywords), ...already_keywords])].sort().join(GRAPH_FIELD_SEP);
  const source_id = [...new Set([...edges_data.map(dp => dp.source_id), ...already_source_ids])].join(GRAPH_FIELD_SEP);
  
  // 确保节点存在
  for (const need_insert_id of [src_id, tgt_id]) {
    if (!(await knowledge_graph_inst.has_node(need_insert_id))) {
      await knowledge_graph_inst.upsert_node(
        need_insert_id,
        {
          source_id,
          description,
          entity_type: '"UNKNOWN"'
        }
      );
    }
  }
  
  // 更新边
  await knowledge_graph_inst.upsert_edge(
    src_id,
    tgt_id,
    {
      weight,
      description,
      keywords,
      source_id
    }
  );
  
  return {
    src_id,
    tgt_id,
    description,
    keywords
  };
}

/**
 * 提取实体
 * @param chunks 块
 * @param knowledge_graph_inst 知识图谱实例
 * @param entity_vdb 实体向量数据库
 * @param entity_name_vdb 实体名称向量数据库
 * @param relationships_vdb 关系向量数据库
 * @param global_config 全局配置
 * @returns 知识图谱实例
 */
export async function extract_entities(
  chunks: Record<string, TextChunkSchema>,
  knowledge_graph_inst: BaseGraphStorage,
  entity_vdb: BaseVectorStorage,
  entity_name_vdb: BaseVectorStorage,
  relationships_vdb: BaseVectorStorage,
  global_config: Record<string, any>
): Promise<BaseGraphStorage | null> {
  const use_llm_func: (...args: any[]) => Promise<any> = global_config.llm_model_func;
  const entity_extract_max_gleaning = global_config.entity_extract_max_gleaning;
  
  const ordered_chunks = Object.entries(chunks);
  const entity_extract_prompt = PROMPTS.entity_extraction;
  
  const context_base = {
    tuple_delimiter: PROMPTS.DEFAULT_TUPLE_DELIMITER,
    record_delimiter: PROMPTS.DEFAULT_RECORD_DELIMITER,
    completion_delimiter: PROMPTS.DEFAULT_COMPLETION_DELIMITER,
    entity_types: PROMPTS.DEFAULT_ENTITY_TYPES.join(',')
  };
  
  const continue_prompt = PROMPTS.entiti_continue_extraction;
  const if_loop_prompt = PROMPTS.entiti_if_loop_extraction;
  
  let already_processed = 0;
  let already_entities = 0;
  let already_relations = 0;
  
  /**
   * 处理单个内容
   * @param chunk_key_dp 块键和数据对
   * @returns 节点和边
   */
  async function _process_single_content(chunk_key_dp: [string, TextChunkSchema]): Promise<[Record<string, any[]>, Record<string, any[]>]> {
    const chunk_key = chunk_key_dp[0];
    const chunk_dp = chunk_key_dp[1];
    const content = chunk_dp.content;
    
    const hint_prompt = entity_extract_prompt.replace('{input_text}', content)
      .replace('{tuple_delimiter}', context_base.tuple_delimiter)
      .replace('{record_delimiter}', context_base.record_delimiter)
      .replace('{completion_delimiter}', context_base.completion_delimiter)
      .replace('{entity_types}', context_base.entity_types);
    
    let final_result = await use_llm_func(hint_prompt);
    let history = pack_user_ass_to_openai_messages(hint_prompt, final_result);
    
    for (let now_glean_index = 0; now_glean_index < entity_extract_max_gleaning; now_glean_index++) {
      const glean_result = await use_llm_func(continue_prompt, { history_messages: history });
      
      history = [...history, ...pack_user_ass_to_openai_messages(continue_prompt, glean_result)];
      final_result += glean_result;
      
      if (now_glean_index === entity_extract_max_gleaning - 1) {
        break;
      }
      
      let if_loop_result: string = await use_llm_func(if_loop_prompt, { history_messages: history });
      if_loop_result = if_loop_result.trim().replace(/['"]/g, '').toLowerCase();
      
      if (if_loop_result !== "yes") {
        break;
      }
    }
    
    const records = split_string_by_multi_markers(
      final_result,
      [context_base.record_delimiter, context_base.completion_delimiter]
    );
    
    const maybe_nodes: Record<string, any[]> = {};
    const maybe_edges: Record<string, any[]> = {};
    
    for (const record of records) {
      const match = record.match(/\((.*)\)/);
      if (!match) continue;
      
      const record_content = match[1];
      const record_attributes = split_string_by_multi_markers(record_content, [context_base.tuple_delimiter]);
      
      const if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key);
      if (if_entities !== null) {
        if (!maybe_nodes[if_entities.entity_name]) {
          maybe_nodes[if_entities.entity_name] = [];
        }
        maybe_nodes[if_entities.entity_name].push(if_entities);
        continue;
      }
      
      const if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key);
      if (if_relation !== null) {
        const key = `${if_relation.src_id},${if_relation.tgt_id}`;
        if (!maybe_edges[key]) {
          maybe_edges[key] = [];
        }
        maybe_edges[key].push(if_relation);
      }
    }
    
    already_processed++;
    already_entities += Object.keys(maybe_nodes).length;
    already_relations += Object.keys(maybe_edges).length;
    
    const now_ticks = PROMPTS.process_tickers[already_processed % PROMPTS.process_tickers.length];
    process.stdout.write(`${now_ticks} Processed ${already_processed} chunks, ${already_entities} entities(duplicated), ${already_relations} relations(duplicated)\r`);
    
    return [maybe_nodes, maybe_edges];
  }
  
  // 并行处理所有块
  const results = await Promise.all(ordered_chunks.map(_process_single_content));
  console.log(); // 清除进度条
  
  // 合并结果
  const maybe_nodes: Record<string, any[]> = {};
  const maybe_edges: Record<string, any[]> = {};
  
  for (const [m_nodes, m_edges] of results) {
    for (const [k, v] of Object.entries(m_nodes)) {
      if (!maybe_nodes[k]) {
        maybe_nodes[k] = [];
      }
      maybe_nodes[k].push(...v);
    }
    
    for (const [k, v] of Object.entries(m_edges)) {
      const [src, tgt] = k.split(',');
      const sorted_key = [src, tgt].sort().join(',');
      
      if (!maybe_edges[sorted_key]) {
        maybe_edges[sorted_key] = [];
      }
      maybe_edges[sorted_key].push(...v);
    }
  }
  
  // 合并节点和边
  const all_entities_data = await Promise.all(
    Object.entries(maybe_nodes).map(([k, v]) => _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config))
  );
  
  const all_relationships_data = await Promise.all(
    Object.entries(maybe_edges).map(([k, v]) => {
      const [src, tgt] = k.split(',');
      return _merge_edges_then_upsert(src, tgt, v, knowledge_graph_inst, global_config);
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
  
  // 更新向量数据库
  if (entity_vdb !== null) {
    const data_for_vdb: Record<string, any> = {};
    
    for (const dp of all_entities_data) {
      const id = compute_mdhash_id(dp.entity_name, "ent-");
      data_for_vdb[id] = {
        content: dp.entity_name + dp.description,
        entity_name: dp.entity_name
      };
    }
    
    await entity_vdb.upsert(data_for_vdb);
  }
  
  if (entity_name_vdb !== null) {
    const data_for_vdb: Record<string, any> = {};
    
    for (const dp of all_entities_data) {
      const id = compute_mdhash_id(dp.entity_name, "Ename-");
      data_for_vdb[id] = {
        content: dp.entity_name,
        entity_name: dp.entity_name
      };
    }
    
    await entity_name_vdb.upsert(data_for_vdb);
  }
  
  if (relationships_vdb !== null) {
    const data_for_vdb: Record<string, any> = {};
    
    for (const dp of all_relationships_data) {
      const id = compute_mdhash_id(dp.src_id + dp.tgt_id, "rel-");
      data_for_vdb[id] = {
        src_id: dp.src_id,
        tgt_id: dp.tgt_id,
        content: dp.keywords + " " + dp.src_id + " " + dp.tgt_id + " " + dp.description
      };
    }
    
    await relationships_vdb.upsert(data_for_vdb);
  }
  
  return knowledge_graph_inst;
}

/**
 * 本地查询
 * @param query 查询
 * @param knowledge_graph_inst 知识图谱实例
 * @param entities_vdb 实体向量数据库
 * @param relationships_vdb 关系向量数据库
 * @param text_chunks_db 文本块数据库
 * @param query_param 查询参数
 * @param global_config 全局配置
 * @returns 查询结果
 */
export async function local_query(
  query: string,
  knowledge_graph_inst: BaseGraphStorage,
  entities_vdb: BaseVectorStorage,
  relationships_vdb: BaseVectorStorage,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  query_param: QueryParam,
  global_config: Record<string, any>
): Promise<string> {
  let context = null;
  const use_model_func = global_config.llm_model_func;
  
  const kw_prompt_temp = PROMPTS.keywords_extraction;
  const kw_prompt = kw_prompt_temp.replace('{query}', query);
  const result = await use_model_func(kw_prompt);
  const json_text = locate_json_string_body_from_string(result);
  
  let keywords = "";
  
  try {
    const keywords_data = JSON.parse(json_text || "{}");
    const keywordsList = keywords_data.low_level_keywords || [];
    keywords = keywordsList.join(", ");
  } catch (e) {
    try {
      // 尝试修复JSON
      const cleanResult = result
        .replace(kw_prompt, "")
        .replace("user", "")
        .replace("model", "")
        .trim();
      
      const jsonPart = "{" + cleanResult.split("{")[1].split("}")[0] + "}";
      const keywords_data = JSON.parse(jsonPart);
      const keywordsList = keywords_data.low_level_keywords || [];
      keywords = keywordsList.join(", ");
    } catch (e) {
      console.error(`JSON parsing error: ${e}`);
      return PROMPTS.fail_response;
    }
  }
  
  if (keywords) {
    context = await _build_local_query_context(
      keywords,
      knowledge_graph_inst,
      entities_vdb,
      text_chunks_db,
      query_param
    );
  }
  
  if (query_param.only_need_context) {
    return context || "";
  }
  
  if (context === null) {
    return PROMPTS.fail_response;
  }
  
  const sys_prompt_temp = PROMPTS.rag_response;
  const sys_prompt = sys_prompt_temp
    .replace('{context_data}', context)
    .replace('{response_type}', query_param.response_type);
  
  let response = await use_model_func(query, { system_prompt: sys_prompt });
  
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

/**
 * 构建本地查询上下文
 * @param query 查询
 * @param knowledge_graph_inst 知识图谱实例
 * @param entities_vdb 实体向量数据库
 * @param text_chunks_db 文本块数据库
 * @param query_param 查询参数
 * @returns 查询上下文
 */
async function _build_local_query_context(
  query: string,
  knowledge_graph_inst: BaseGraphStorage,
  entities_vdb: BaseVectorStorage,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  query_param: QueryParam
): Promise<string | null> {
  const results = await entities_vdb.query(query, query_param.top_k);
  
  if (results.length === 0) {
    return null;
  }
  
  const node_datas = await Promise.all(
    results.map(r => knowledge_graph_inst.get_node(r.entity_name))
  );
  
  if (!node_datas.every(n => n !== null)) {
    logger.warning("Some nodes are missing, maybe the storage is damaged");
  }
  
  const node_degrees = await Promise.all(
    results.map(r => knowledge_graph_inst.node_degree(r.entity_name))
  );
  
  const validNodeData = node_datas
    .map((n, i) => n !== null ? {
      ...n,
      entity_name: results[i].entity_name,
      rank: node_degrees[i]
    } : null)
    .filter(n => n !== null) as Record<string, any>[];
  
  const use_text_units = await _find_most_related_text_unit_from_entities(
    validNodeData,
    query_param,
    text_chunks_db,
    knowledge_graph_inst
  );
  
  const use_relations = await _find_most_related_edges_from_entities(
    validNodeData,
    query_param,
    knowledge_graph_inst
  );
  
  logger.info(
    `Local query uses ${validNodeData.length} entities, ${use_relations.length} relations, ${use_text_units.length} text units`
  );
  
  // 构建实体部分
  const entites_section_list: any[][] = [["id", "entity", "type", "description", "rank"]];
  
  for (let i = 0; i < validNodeData.length; i++) {
    const n = validNodeData[i];
    entites_section_list.push([
      i,
      n.entity_name,
      n.entity_type || "UNKNOWN",
      n.description || "UNKNOWN",
      n.rank
    ]);
  }
  
  const entities_context = list_of_list_to_csv(entites_section_list);
  
  // 构建关系部分
  const relations_section_list: any[][] = [
    ["id", "source", "target", "description", "keywords", "weight", "rank"]
  ];
  
  for (let i = 0; i < use_relations.length; i++) {
    const e = use_relations[i];
    relations_section_list.push([
      i,
      e.src_tgt[0],
      e.src_tgt[1],
      e.description,
      e.keywords,
      e.weight,
      e.rank
    ]);
  }
  
  const relations_context = list_of_list_to_csv(relations_section_list);
  
  // 构建文本单元部分
  const text_units_section_list: any[][] = [["id", "content"]];
  
  for (let i = 0; i < use_text_units.length; i++) {
    const t = use_text_units[i];
    text_units_section_list.push([i, t.content]);
  }
  
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

/**
 * 查找与实体最相关的文本单元
 * @param node_datas 节点数据
 * @param query_param 查询参数
 * @param text_chunks_db 文本块数据库
 * @param knowledge_graph_inst 知识图谱实例
 * @returns 文本单元
 */
async function _find_most_related_text_unit_from_entities(
  node_datas: Array<Record<string, any>>,
  query_param: QueryParam,
  text_chunks_db: BaseKVStorage<TextChunkSchema>,
  knowledge_graph_inst: BaseGraphStorage
): Promise<Array<TextChunkSchema>> {
  const text_units = node_datas.map(dp => 
    split_string_by_multi_markers(dp.source_id, [GRAPH_FIELD_SEP])
  );
  
  const edges = await Promise.all(
    node_datas.map(dp => knowledge_graph_inst.get_node_edges(dp.entity_name))
  );
  
  const all_one_hop_nodes = new Set<string>();
  
  for (const this_edges of edges) {
    if (!this_edges) continue;
    
    for (const e of this_edges) {
      all_one_hop_nodes.add(e[1]);
    }
  }
  
  const all_one_hop_nodes_array = Array.from(all_one_hop_nodes);
  const all_one_hop_nodes_data = await Promise.all(
    all_one_hop_nodes_array.map(e => knowledge_graph_inst.get_node(e))
  );
  
  // 添加节点数据的空检查
  const all_one_hop_text_units_lookup: Record<string, Set<string>> = {};
  
  for (let i = 0; i < all_one_hop_nodes_array.length; i++) {
    const k = all_one_hop_nodes_array[i];
    const v = all_one_hop_nodes_data[i];
    
    if (v !== null && v.source_id) {
      all_one_hop_text_units_lookup[k] = new Set(
        split_string_by_multi_markers(v.source_id, [GRAPH_FIELD_SEP])
      );
    }
  }
  
  const all_text_units_lookup: Record<string, any> = {};
  
  for (let index = 0; index < text_units.length; index++) {
    const this_text_units = text_units[index];
    const this_edges = edges[index];
    
    for (const c_id of this_text_units) {
      if (c_id in all_text_units_lookup) continue;
      
      let relation_counts = 0;
      
      if (this_edges) {
        for (const e of this_edges) {
          if (
            e[1] in all_one_hop_text_units_lookup
