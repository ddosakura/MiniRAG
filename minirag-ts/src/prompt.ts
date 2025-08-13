// 图字段分隔符
export const GRAPH_FIELD_SEP = "|||";

// 提示模板
export const PROMPTS = {
  // 默认分隔符
  DEFAULT_TUPLE_DELIMITER: ",",
  DEFAULT_RECORD_DELIMITER: "\n",
  DEFAULT_COMPLETION_DELIMITER: "###",
  DEFAULT_ENTITY_TYPES: ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT", "PRODUCT", "CONCEPT"],
  
  // 处理标记
  process_tickers: ["-", "\\", "|", "/"],
  
  // 实体提取提示
  entity_extraction: `请从以下文本中提取实体和关系。
  
实体格式: ("entity", "实体名称", "实体类型", "实体描述")
关系格式: ("relationship", "源实体", "目标实体", "关系描述", "关系关键词", 权重)

实体类型可以是: {entity_types}

请使用 {tuple_delimiter} 作为元组分隔符，{record_delimiter} 作为记录分隔符。

输入文本:
{input_text}

提取结果:
`,

  // 实体继续提取提示
  entiti_continue_extraction: "请继续从文本中提取更多实体和关系。",
  
  // 实体循环提取提示
  entiti_if_loop_extraction: "是否还有更多实体和关系可以提取？请回答 'yes' 或 'no'。",
  
  // 关键词提取提示
  keywords_extraction: `请从以下查询中提取高级和低级关键词。
  
高级关键词应该是抽象概念、主题或类别。
低级关键词应该是具体的实体、名称或术语。

查询: {query}

请以JSON格式返回结果:
{
  "high_level_keywords": ["关键词1", "关键词2", ...],
  "low_level_keywords": ["关键词1", "关键词2", ...]
}`,

  // MiniRAG查询到关键词提示
  minirag_query2kwd: `请从以下查询中提取实体和可能的答案类型关键词。

查询: {query}

可能的实体类型: {TYPE_POOL}

请以JSON格式返回结果:
{
  "entities_from_query": ["实体1", "实体2", ...],
  "answer_type_keywords": ["类型1", "类型2", ...]
}`,

  // RAG响应提示
  rag_response: `你是一个知识助手。请使用以下上下文信息来回答用户的问题。如果上下文中没有足够的信息，请说明你不知道答案。

上下文信息:
{context_data}

请以{response_type}格式回答。`,

  // 朴素RAG响应提示
  naive_rag_response: `你是一个知识助手。请使用以下上下文信息来回答用户的问题。如果上下文中没有足够的信息，请说明你不知道答案。

上下文信息:
{content_data}

请以{response_type}格式回答。`,

  // 失败响应
  fail_response: "抱歉，我无法回答这个问题。请尝试重新表述或提供更多信息。"
};