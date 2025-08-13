import {
  truncate_list_by_token_size,
  logger,
} from '../utils';
import { PROMPTS } from '../prompt';

export async function naive_query(
  query: string,
  chunks_vdb: any,
  text_chunks_db: any,
  query_param: any,
  global_config: Record<string, any>
): Promise<string> {
  const use_model_func = global_config["llm_model_func"];
  const results = await chunks_vdb.query(query, { top_k: query_param.top_k });
  
  if (results.length === 0) {
    return PROMPTS["fail_response"];
  }
  
  const chunks_ids = results.map(r => r["id"]);
  const chunks = await text_chunks_db.get_by_ids(chunks_ids);

  const maybe_trun_chunks = truncate_list_by_token_size(
    chunks,
    x => x["content"],
    query_param.max_token_for_text_unit
  );
  
  logger.info(`Truncate ${chunks.length} to ${maybe_trun_chunks.length} chunks`);
  const section = maybe_trun_chunks.map(c => c["content"]).join("--New Chunk--\n");
  
  if (query_param.only_need_context) {
    return section;
  }
  
  const sys_prompt_temp = PROMPTS["naive_rag_response"];
  const sys_prompt = sys_prompt_temp
    .replace("{content_data}", section)
    .replace("{response_type}", query_param.response_type);
    
  let response = await use_model_func(
    query,
    sys_prompt,
  );

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