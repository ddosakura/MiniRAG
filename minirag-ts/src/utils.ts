import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import * as html from 'html-entities';
import * as csv from 'csv-parse/sync';
import * as csvStringify from 'csv-stringify/sync';
import { createLogger, format, transports, Logger } from 'winston';
import * as np from 'numpy-ts';
import * as tiktoken from 'tiktoken';

// 全局变量
let ENCODER: any = null;

// 创建日志记录器
export const logger = createLogger({
  level: 'debug',
  format: format.combine(
    format.timestamp(),
    format.printf(({ timestamp, level, message }) => {
      return `${timestamp} - minirag - ${level} - ${message}`;
    })
  ),
  transports: []
});

export function set_logger(log_file: string): void {
  // 移除所有现有的传输器
  logger.clear();
  
  // 添加文件传输器
  logger.add(new transports.File({
    filename: log_file,
    level: 'debug',
    format: format.combine(
      format.timestamp(),
      format.printf(({ timestamp, level, message }) => {
        return `${timestamp} - minirag - ${level} - ${message}`;
      })
    )
  }));
}

export class EmbeddingFunc {
  embedding_dim: number;
  max_token_size: number;
  func: (...args: any[]) => Promise<np.NDArray>;

  constructor(embedding_dim: number, max_token_size: number, func: (...args: any[]) => Promise<np.NDArray>) {
    this.embedding_dim = embedding_dim;
    this.max_token_size = max_token_size;
    this.func = func;
  }

  async call(...args: any[]): Promise<np.NDArray> {
    return await this.func(...args);
  }
}

export function compute_mdhash_id(content: string, prefix: string = ""): string {
  return prefix + crypto.createHash('md5').update(content).digest('hex');
}

export function compute_args_hash(...args: any[]): string;
export function compute_args_hash(...args: any[]): string {
  const argsStr = args.map(arg => String(arg)).join('');
  return crypto.createHash('md5').update(argsStr).digest('hex');
}

export function clean_text(text: string): string {
  return text.trim().replace(/\x00/g, '');
}

export function get_content_summary(content: string, max_length: number = 100): string {
  content = content.trim();
  return content.length <= max_length ? content : content.substring(0, max_length) + "...";
}

export function locate_json_string_body_from_string(content: string): string | null {
  const maybeJsonStr = content.match(/{.*}/s);
  return maybeJsonStr ? maybeJsonStr[0] : null;
}

export function convert_response_to_json(response: string): Record<string, any> {
  const jsonStr = locate_json_string_body_from_string(response);
  if (!jsonStr) {
    throw new Error(`Unable to parse JSON from response: ${response}`);
  }
  
  try {
    return JSON.parse(jsonStr);
  } catch (e) {
    logger.error(`Failed to parse JSON: ${jsonStr}`);
    throw e;
  }
}

export function limit_async_func_call<T extends (...args: any[]) => Promise<any>>(
  max_size: number, 
  waitingTime: number = 0.0001
): (func: T) => T {
  return function(func: T): T {
    let currentSize = 0;
    
    const waitFunc = async function(...args: any[]): Promise<any> {
      while (currentSize >= max_size) {
        await new Promise(resolve => setTimeout(resolve, waitingTime * 1000));
      }
      currentSize++;
      try {
        const result = await func(...args);
        return result;
      } finally {
        currentSize--;
      }
    };
    
    return waitFunc as unknown as T;
  };
}

export function wrap_embedding_func_with_attrs(
  embedding_dim: number,
  max_token_size: number
): (func: (...args: any[]) => Promise<np.NDArray>) => EmbeddingFunc {
  return function(func: (...args: any[]) => Promise<np.NDArray>): EmbeddingFunc {
    return new EmbeddingFunc(embedding_dim, max_token_size, func);
  };
}

export function load_json(fileName: string): any {
  if (!fs.existsSync(fileName)) {
    return null;
  }
  const content = fs.readFileSync(fileName, 'utf-8');
  return JSON.parse(content);
}

export function write_json(jsonObj: any, fileName: string): void {
  fs.writeFileSync(fileName, JSON.stringify(jsonObj, null, 2), 'utf-8');
}

export function encode_string_by_tiktoken(content: string, model_name: string = "gpt-4o"): number[] {
  if (ENCODER === null) {
    ENCODER = tiktoken.encoding_for_model(model_name);
  }
  return ENCODER.encode(content);
}

export function decode_tokens_by_tiktoken(tokens: number[], model_name: string = "gpt-4o"): string {
  if (ENCODER === null) {
    ENCODER = tiktoken.encoding_for_model(model_name);
  }
  return ENCODER.decode(tokens);
}

export function pack_user_ass_to_openai_messages(...args: string[]): Array<{role: string, content: string}> {
  const roles = ["user", "assistant"];
  return args.map((content, i) => ({
    role: roles[i % 2],
    content
  }));
}

export function split_string_by_multi_markers(content: string, markers: string[]): string[] {
  if (!markers || markers.length === 0) {
    return [content];
  }
  
  const escapedMarkers = markers.map(marker => marker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const regex = new RegExp(escapedMarkers.join('|'), 'g');
  const results = content.split(regex);
  
  return results.map(r => r.trim()).filter(r => r);
}

export function clean_str(input: any): string | any {
  if (typeof input !== 'string') {
    return input;
  }
  
  const unescaped = html.decode(input.trim());
  return unescaped.replace(/[\x00-\x1f\x7f-\x9f]/g, '');
}

export function is_float_regex(value: string): boolean {
  return /^[-+]?[0-9]*\.?[0-9]+$/.test(value);
}

export function truncate_list_by_token_size<T>(
  list_data: T[], 
  key: (item: T) => string, 
  max_token_size: number
): T[] {
  if (max_token_size <= 0) {
    return [];
  }
  
  let tokens = 0;
  for (let i = 0; i < list_data.length; i++) {
    tokens += encode_string_by_tiktoken(key(list_data[i])).length;
    if (tokens > max_token_size) {
      return list_data.slice(0, i);
    }
  }
  
  return list_data;
}

export function list_of_list_to_csv(data: string[][]): string {
  return csvStringify.stringify(data);
}

export function csv_string_to_list(csv_string: string): string[][] {
  return csv.parse(csv_string);
}

export function save_data_to_file(data: any, fileName: string): void {
  fs.writeFileSync(fileName, JSON.stringify(data, null, 4), 'utf-8');
}

export function xml_to_json(xml_file: string): any {
  try {
    // 注意：这里需要使用适当的XML解析库，例如fast-xml-parser
    // 由于TypeScript版本中可能使用不同的库，这里只提供一个简化的实现
    const xmlContent = fs.readFileSync(xml_file, 'utf-8');
    
    // 这里应该使用XML解析库解析XML内容
    // 由于实现细节依赖于具体的库，这里返回一个空对象
    console.log(`Root element: [XML解析需要具体实现]`);
    console.log(`Root attributes: [XML解析需要具体实现]`);
    
    return {
      nodes: [],
      edges: []
    };
  } catch (e) {
    console.error(`Error parsing XML file: ${e}`);
    return null;
  }
}

export function safe_unicode_decode(content: Buffer): string {
  // 正则表达式查找所有形式为\uXXXX的Unicode转义序列
  const unicodeEscapePattern = /\\u([0-9a-fA-F]{4})/g;
  
  // 将Unicode转义替换为实际字符的函数
  const replaceUnicodeEscape = (match: string, p1: string): string => {
    return String.fromCharCode(parseInt(p1, 16));
  };
  
  // 执行替换
  const decodedContent = content.toString('utf-8').replace(unicodeEscapePattern, replaceUnicodeEscape);
  
  return decodedContent;
}

export function process_combine_contexts(hl: string, ll: string): string {
  let header = null;
  const list_hl = csv_string_to_list(hl.trim());
  const list_ll = csv_string_to_list(ll.trim());
  
  if (list_hl.length > 0) {
    header = list_hl[0];
    list_hl.shift();
  }
  
  if (list_ll.length > 0) {
    header = list_ll[0];
    list_ll.shift();
  }
  
  if (!header) {
    return "";
  }
  
  const hl_items = list_hl.length > 0 
    ? list_hl.filter(item => item.length > 0).map(item => item.slice(1).join(',')) 
    : [];
    
  const ll_items = list_ll.length > 0 
    ? list_ll.filter(item => item.length > 0).map(item => item.slice(1).join(',')) 
    : [];
  
  const combined_sources_set = new Set([...hl_items, ...ll_items].filter(Boolean));
  const combined_sources = [header.join(',\t')];
  
  Array.from(combined_sources_set).forEach((item, i) => {
    combined_sources.push(`${i + 1},\t${item}`);
  });
  
  return combined_sources.join('\n');
}

export function is_continuous_subsequence(subseq: any[], seq: any[]): boolean {
  function find_all_indexes(tup: any[], value: any): number[] {
    const indexes: number[] = [];
    let start = 0;
    
    while (true) {
      const index = tup.indexOf(value, start);
      if (index === -1) break;
      
      indexes.push(index);
      start = index + 1;
    }
    
    return indexes;
  }
  
  const index_list = find_all_indexes(seq, subseq[0]);
  
  for (const idx of index_list) {
    if (idx !== seq.length - 1) {
      if (seq[idx + 1] === subseq[subseq.length - 1]) {
        return true;
      }
    }
  }
  
  return false;
}

export function merge_tuples(list1: any[][], list2: any[][]): any[][] {
  const result: any[][] = [];
  
  for (const tup of list1) {
    const last_element = tup[tup.length - 1];
    
    if (tup.slice(0, -1).includes(last_element)) {
      result.push(tup);
    } else {
      const matching_tuples = list2.filter(t => t[0] === last_element);
      let already_match_flag = 0;
      
      for (const match of matching_tuples) {
        const matchh = [match[1], match[0]];
        
        if (is_continuous_subsequence(match, tup) || is_continuous_subsequence(matchh, tup)) {
          continue;
        }
        
        already_match_flag = 1;
        const merged_tuple = [...tup, ...match.slice(1)];
        result.push(merged_tuple);
      }
      
      if (!already_match_flag) {
        result.push(tup);
      }
    }
  }
  
  return result;
}

export function count_elements_in_tuple(tuple_elements: any[], list_elements: any[]): number {
  const sorted_list = [...list_elements].sort();
  const sorted_tuple = [...tuple_elements].sort();
  let count = 0;
  let list_index = 0;
  
  for (const elem of sorted_tuple) {
    while (list_index < sorted_list.length && sorted_list[list_index] < elem) {
      list_index++;
    }
    
    if (list_index < sorted_list.length && sorted_list[list_index] === elem) {
      count++;
      list_index++;
    }
  }
  
  return count;
}

export function cal_path_score_list(
  candidate_reasoning_path: Record<string, any>,
  maybe_answer_list: any[]
): Record<string, any> {
  const scored_reasoning_path: Record<string, any> = {};
  
  for (const [k, v] of Object.entries(candidate_reasoning_path)) {
    const score = v.Score;
    const paths = v.Path;
    const scores: Record<string, number[]> = {};
    
    for (const p of paths) {
      scores[p] = [count_elements_in_tuple(p, maybe_answer_list)];
    }
    
    scored_reasoning_path[k] = { Score: score, Path: scores };
  }
  
  return scored_reasoning_path;
}

export function edge_vote_path(
  path_dict: Record<string, any>,
  edge_list: Array<{ src_id: string; tgt_id: string }>
): [Record<string, any>, Record<string, any[]>] {
  const return_dict = JSON.parse(JSON.stringify(path_dict));
  const EDGELIST: [string, string][] = [];
  const pairs_append: Record<string, any[]> = {};
  
  for (const i of edge_list) {
    EDGELIST.push([i.src_id, i.tgt_id]);
  }
  
  for (const [i_key, i_value] of Object.entries(return_dict)) {
    for (const [j_key, j_value] of Object.entries(i_value.Path)) {
      if (j_value) {
        let count = 0;
        const j_key_array = JSON.parse(j_key); // 假设j_key是一个JSON字符串表示的数组
        
        for (const pairs of EDGELIST) {
          if (is_continuous_subsequence(pairs, j_key_array)) {
            count++;
            
            if (!pairs_append[j_key]) {
              pairs_append[j_key] = [pairs];
            } else {
              pairs_append[j_key].push(pairs);
            }
          }
        }
        
        // 更新分数
        j_value.push(count);
      }
    }
  }
  
  return [return_dict, pairs_append];
}

// 缓存函数
export function cosine_similarity(v1: number[], v2: number[]): number {
  const dot_product = v1.reduce((sum, val, i) => sum + val * v2[i], 0);
  const norm1 = Math.sqrt(v1.reduce((sum, val) => sum + val * val, 0));
  const norm2 = Math.sqrt(v2.reduce((sum, val) => sum + val * val, 0));
  return dot_product / (norm1 * norm2);
}

export function quantize_embedding(
  embedding: number[] | np.NDArray,
  bits: number = 8
): [Uint8Array, number, number] {
  const arr = Array.isArray(embedding) ? embedding : Array.from(embedding as any);
  const min_val = Math.min(...arr);
  const max_val = Math.max(...arr);
  const scale = (Math.pow(2, bits) - 1) / (max_val - min_val);
  
  const quantized = new Uint8Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    quantized[i] = Math.round((arr[i] - min_val) * scale);
  }
  
  return [quantized, min_val, max_val];
}

export function dequantize_embedding(
  quantized: Uint8Array,
  min_val: number,
  max_val: number,
  bits: number = 8
): Float32Array {
  const scale = (max_val - min_val) / (Math.pow(2, bits) - 1);
  const dequantized = new Float32Array(quantized.length);
  
  for (let i = 0; i < quantized.length; i++) {
    dequantized[i] = quantized[i] * scale + min_val;
  }
  
  return dequantized;
}

export function calculate_similarity(
  sentences: string[],
  target: string,
  method: string = "levenshtein",
  n: number = 1,
  k: number = 1
): number[] {
  const target_tokens = target.toLowerCase().split(/\s+/);
  const similarities_with_index: [number, number][] = [];
  
  if (method === "jaccard") {
    for (let i = 0; i < sentences.length; i++) {
      const sentence_tokens = sentences[i].toLowerCase().split(/\s+/);
      const intersection = new Set(sentence_tokens.filter(token => target_tokens.includes(token)));
      const union = new Set([...sentence_tokens, ...target_tokens]);
      const jaccard_score = union.size > 0 ? intersection.size / union.size : 0;
      similarities_with_index.push([i, jaccard_score]);
    }
  } else if (method === "levenshtein") {
    for (let i = 0; i < sentences.length; i++) {
      const sentence_tokens = sentences[i].toLowerCase().split(/\s+/);
      // 这里需要实现编辑距离算法，简化版本
      const distance = levenshtein_distance(target_tokens, sentence_tokens);
      const similarity = 1 - (distance / Math.max(target_tokens.length, sentence_tokens.length));
      similarities_with_index.push([i, similarity]);
    }
  } else if (method === "overlap") {
    for (let i = 0; i < sentences.length; i++) {
      const sentence_tokens = new Set(sentences[i].toLowerCase().split(/\s+/));
      const target_tokens_set = new Set(target_tokens);
      const overlap = new Set([...sentence_tokens].filter(x => target_tokens_set.has(x)));
      const score = sentence_tokens.size > 0 ? 
        overlap.size / Math.min(sentence_tokens.size, target_tokens_set.size) : 0;
      similarities_with_index.push([i, score]);
    }
  } else {
    throw new Error(`Unsupported method: ${method}. Choose from 'jaccard', 'levenshtein', or 'overlap'.`);
  }
  
  // 按相似度降序排序
  similarities_with_index.sort((a, b) => b[1] - a[1]);
  
  // 返回前k个最相似句子的索引
  return similarities_with_index.slice(0, k).map(item => item[0]);
}

// 辅助函数：计算编辑距离
function levenshtein_distance(a: string[], b: string[]): number {
  const m = a.length;
  const n = b.length;
  
  // 创建距离矩阵
  const d: number[][] = Array(m + 1).fill(0).map(() => Array(n + 1).fill(0));
  
  // 初始化第一行和第一列
  for (let i = 0; i <= m; i++) d[i][0] = i;
  for (let j = 0; j <= n; j++) d[0][j] = j;
  
  // 填充距离矩阵
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(
        d[i - 1][j] + 1,      // 删除
        d[i][j - 1] + 1,      // 插入
        d[i - 1][j - 1] + cost // 替换
      );
    }
  }
  
  return d[m][n];
}