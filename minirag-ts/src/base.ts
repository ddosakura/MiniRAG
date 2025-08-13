import { EmbeddingFunc } from './utils';
import * as np from './types/numpy-js';

/**
 * 文本块模式
 */
export interface TextChunkSchema {
  tokens: number;
  content: string;
  full_doc_id: string;
  chunk_order_index: number;
}

/**
 * 查询参数
 */
export class QueryParam {
  mode: "light" | "naive" | "mini" = "mini";
  only_need_context: boolean = false;
  only_need_prompt: boolean = false;
  response_type: string = "Multiple Paragraphs";
  stream: boolean = false;
  // 要检索的 top-k 项目数量；对应于"local"模式下的实体和"global"模式下的关系。
  top_k: number = parseInt(process.env.TOP_K || "60");
  // 要检索的文档块数量。
  // top_n: number = 10;
  // 原始块的令牌数。
  max_token_for_text_unit: number = 4000;
  // 关系描述的令牌数
  max_token_for_global_context: number = 4000;
  // 实体描述的令牌数
  max_token_for_local_context: number = 4000;
  // Mini模式的节点上下文令牌数，如果太长，SLM可能无法生成任何响应
  max_token_for_node_context: number = 500;

  hl_keywords: string[] = [];
  ll_keywords: string[] = [];
  // 对话历史支持
  conversation_history: Array<{role: string, content: string}> = []; // 格式：[{"role": "user/assistant", "content": "message"}]
  history_turns: number = 3; // 要考虑的完整对话轮次（用户-助手对）数量

  constructor(params?: Partial<QueryParam>) {
    if (params) {
      Object.assign(this, params);
    }
  }
}

/**
 * 存储命名空间
 */
export abstract class StorageNameSpace {
  namespace: string;
  global_config: Record<string, any>;

  constructor(namespace: string, global_config: Record<string, any>) {
    this.namespace = namespace;
    this.global_config = global_config;
  }

  /**
   * 索引完成后提交存储操作
   */
  index_done_callback(): Promise<void> {
    // 默认实现为空
    return Promise.resolve();
  }

  /**
   * 查询完成后提交存储操作
   */
  query_done_callback(): Promise<void> {
    // 默认实现为空
    return Promise.resolve();
  }
}

/**
 * 基础向量存储
 */
export abstract class BaseVectorStorage extends StorageNameSpace {
  embedding_func: EmbeddingFunc;
  meta_fields: Set<string>;

  constructor(namespace: string, global_config: Record<string, any>, embedding_func: EmbeddingFunc, meta_fields: Set<string> = new Set()) {
    super(namespace, global_config);
    this.embedding_func = embedding_func;
    this.meta_fields = meta_fields;
  }

  /**
   * 查询向量数据库
   * @param query 查询字符串
   * @param top_k 返回结果数量
   */
  abstract query(query: string, top_k: number): Promise<Array<Record<string, any>>>;

  /**
   * 更新或插入数据
   * @param data 数据字典，使用值中的'content'字段进行嵌入，使用键作为ID。
   * 如果embedding_func为null，则使用值中的'embedding'字段
   */
  abstract upsert(data: Record<string, Record<string, any>>): Promise<void>;
}

/**
 * 基础键值存储
 */
export abstract class BaseKVStorage<T> extends StorageNameSpace {
  embedding_func: EmbeddingFunc;

  constructor(namespace: string, global_config: Record<string, any>, embedding_func: EmbeddingFunc) {
    super(namespace, global_config);
    this.embedding_func = embedding_func;
  }

  /**
   * 获取所有键
   */
  abstract all_keys(): Promise<string[]>;

  /**
   * 通过ID获取值
   * @param id 键ID
   */
  abstract get_by_id(id: string): Promise<T | null>;

  /**
   * 通过多个ID获取值
   * @param ids ID列表
   * @param fields 要获取的字段集合
   */
  abstract get_by_ids(ids: string[], fields?: Set<string> | null): Promise<Array<T | null>>;

  /**
   * 过滤键列表，返回不存在的键
   * @param data 键列表
   */
  abstract filter_keys(data: string[]): Promise<Set<string>>;

  /**
   * 更新或插入数据
   * @param data 数据字典
   */
  abstract upsert(data: Record<string, T>): Promise<void>;

  /**
   * 删除存储
   */
  abstract drop(): Promise<void>;
}

/**
 * 基础图存储
 */
export abstract class BaseGraphStorage extends StorageNameSpace {
  embedding_func: EmbeddingFunc | null;

  constructor(namespace: string, global_config: Record<string, any>, embedding_func: EmbeddingFunc | null = null) {
    super(namespace, global_config);
    this.embedding_func = embedding_func;
  }

  /**
   * 获取类型
   */
  abstract get_types(): Promise<[string[], string[]]>;

  /**
   * 检查节点是否存在
   * @param node_id 节点ID
   */
  abstract has_node(node_id: string): Promise<boolean>;

  /**
   * 检查边是否存在
   * @param source_node_id 源节点ID
   * @param target_node_id 目标节点ID
   */
  abstract has_edge(source_node_id: string, target_node_id: string): Promise<boolean>;

  /**
   * 获取节点度数
   * @param node_id 节点ID
   */
  abstract node_degree(node_id: string): Promise<number>;

  /**
   * 获取边度数
   * @param src_id 源节点ID
   * @param tgt_id 目标节点ID
   */
  abstract edge_degree(src_id: string, tgt_id: string): Promise<number>;

  /**
   * 获取节点
   * @param node_id 节点ID
   */
  abstract get_node(node_id: string): Promise<Record<string, any> | null>;

  /**
   * 获取边
   * @param source_node_id 源节点ID
   * @param target_node_id 目标节点ID
   */
  abstract get_edge(source_node_id: string, target_node_id: string): Promise<Record<string, any> | null>;

  /**
   * 获取节点的边
   * @param source_node_id 源节点ID
   */
  abstract get_node_edges(source_node_id: string): Promise<Array<[string, string]> | null>;

  /**
   * 更新或插入节点
   * @param node_id 节点ID
   * @param node_data 节点数据
   */
  abstract upsert_node(node_id: string, node_data: Record<string, string>): Promise<void>;

  /**
   * 更新或插入边
   * @param source_node_id 源节点ID
   * @param target_node_id 目标节点ID
   * @param edge_data 边数据
   */
  abstract upsert_edge(source_node_id: string, target_node_id: string, edge_data: Record<string, string>): Promise<void>;

  /**
   * 删除节点
   * @param node_id 节点ID
   */
  abstract delete_node(node_id: string): Promise<void>;

  /**
   * 嵌入节点
   * @param algorithm 算法
   */
  embed_nodes(algorithm: string): Promise<[np.NDArray, string[]]> {
    throw new Error("Node embedding is not used in minirag.");
  }
}

/**
 * 文档处理状态枚举
 */
export enum DocStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  PROCESSED = "processed",
  FAILED = "failed"
}

/**
 * 文档处理状态数据结构
 */
export class DocProcessingStatus {
  content: string;                   // 文档的原始内容
  content_summary: string;           // 文档内容的前100个字符，用于预览
  content_length: number;            // 文档总长度
  status: DocStatus;                 // 当前处理状态
  created_at: string;                // 文档创建时的ISO格式时间戳
  updated_at: string;                // 文档最后更新的ISO格式时间戳
  chunks_count?: number;             // 分割后的块数，用于处理
  error?: string;                    // 失败时的错误消息
  metadata: Record<string, any>;     // 附加元数据

  constructor(
    content: string,
    content_summary: string,
    content_length: number,
    status: DocStatus,
    created_at: string,
    updated_at: string,
    chunks_count?: number,
    error?: string,
    metadata: Record<string, any> = {}
  ) {
    this.content = content;
    this.content_summary = content_summary;
    this.content_length = content_length;
    this.status = status;
    this.created_at = created_at;
    this.updated_at = updated_at;
    this.chunks_count = chunks_count;
    this.error = error;
    this.metadata = metadata;
  }
}

/**
 * 文档状态存储基类
 */
export abstract class DocStatusStorage extends BaseKVStorage<DocProcessingStatus> {
  /**
   * 获取每种状态的文档计数
   */
  abstract get_status_counts(): Promise<Record<string, number>>;

  /**
   * 获取所有失败的文档
   */
  abstract get_failed_docs(): Promise<Record<string, DocProcessingStatus>>;

  /**
   * 获取所有待处理的文档
   */
  abstract get_pending_docs(): Promise<Record<string, DocProcessingStatus>>;
}
