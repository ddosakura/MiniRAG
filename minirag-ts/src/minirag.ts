import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';
import { format } from 'date-fns';
import { 
  BaseGraphStorage, 
  BaseKVStorage, 
  BaseVectorStorage, 
  DocProcessingStatus, 
  DocStatus, 
  DocStatusStorage, 
  QueryParam, 
  StorageNameSpace 
} from './base';
import { 
  EmbeddingFunc, 
  clean_text, 
  compute_mdhash_id, 
  convert_response_to_json, 
  get_content_summary, 
  limit_async_func_call, 
  logger, 
  set_logger 
} from './utils';

// 导入操作函数
import {
  chunking_by_token_size,
  extract_entities,
  hybrid_query,
  minirag_query,
  naive_query,
} from './operate';

// 存储类映射
const STORAGES: Record<string, string> = {
  "NetworkXStorage": "./kg/networkx_impl",
  "JsonKVStorage": "./kg/json_kv_impl",
  "NanoVectorDBStorage": "./kg/nano_vector_db_impl",
  "JsonDocStatusStorage": "./kg/jsondocstatus_impl",
  "Neo4JStorage": "./kg/neo4j_impl",
  "OracleKVStorage": "./kg/oracle_impl",
  "OracleGraphStorage": "./kg/oracle_impl",
  "OracleVectorDBStorage": "./kg/oracle_impl",
  "MilvusVectorDBStorge": "./kg/milvus_impl",
  "MongoKVStorage": "./kg/mongo_impl",
  "MongoGraphStorage": "./kg/mongo_impl",
  "RedisKVStorage": "./kg/redis_impl",
  "ChromaVectorDBStorage": "./kg/chroma_impl",
  "TiDBKVStorage": "./kg/tidb_impl",
  "TiDBVectorDBStorage": "./kg/tidb_impl",
  "TiDBGraphStorage": "./kg/tidb_impl",
  "PGKVStorage": "./kg/postgres_impl",
  "PGVectorStorage": "./kg/postgres_impl",
  "AGEStorage": "./kg/age_impl",
  "PGGraphStorage": "./kg/postgres_impl",
  "GremlinStorage": "./kg/gremlin_impl",
  "PGDocStatusStorage": "./kg/postgres_impl",
  "WeaviateVectorStorage": "./kg/weaviate_impl",
  "WeaviateKVStorage": "./kg/weaviate_impl",
  "WeaviateGraphStorage": "./kg/weaviate_impl",
  "run_sync": "./kg/weaviate_impl",
};

// 加载环境变量
dotenv.config({ path: ".env", override: false });

/**
 * 懒加载外部导入
 * @param module_name 模块名称
 * @param class_name 类名
 * @returns 类构造函数
 */
function lazy_external_import(module_name: string, class_name: string): (...args: any[]) => any {
  return function(...args: any[]): any {
    // 动态导入模块
    const module = require(module_name);
    const cls = module[class_name];
    return new cls(...args);
  };
}

/**
 * 确保始终有一个可用的事件循环
 * @returns 当前或新创建的事件循环
 */
function always_get_an_event_loop(): any {
  // 在Node.js中，我们不需要显式创建事件循环
  // 这个函数在TypeScript中主要是为了保持API兼容性
  return {
    run_until_complete: async (promise: Promise<any>) => {
      return await promise;
    }
  };
}

/**
 * MiniRAG类
 */
export class MiniRAG {
  // 基本配置
  working_dir: string;
  kv_storage: string;
  vector_storage: string;
  graph_storage: string;
  current_log_level: string;
  log_level: string;

  // 文本分块配置
  chunk_token_size: number;
  chunk_overlap_token_size: number;
  tiktoken_model_name: string;

  // 实体提取配置
  entity_extract_max_gleaning: number;
  entity_summary_to_max_tokens: number;

  // 节点嵌入配置
  node_embedding_algorithm: string;
  node2vec_params: Record<string, any>;

  // 嵌入函数配置
  embedding_func: EmbeddingFunc;
  embedding_batch_num: number;
  embedding_func_max_async: number;

  // LLM配置
  llm_model_func: (...args: any[]) => Promise<any>;
  llm_model_name: string;
  llm_model_max_token_size: number;
  llm_model_max_async: number;
  llm_model_kwargs: Record<string, any>;

  // 存储配置
  vector_db_storage_cls_kwargs: Record<string, any>;
  enable_llm_cache: boolean;

  // 扩展配置
  addon_params: Record<string, any>;
  convert_response_to_json_func: (response: string) => Record<string, any>;

  // 文档状态存储类型
  doc_status_storage: string;

  // 自定义分块函数
  chunking_func: (...args: any[]) => any;
  chunking_func_kwargs: Record<string, any>;

  // 最大并行插入数
  max_parallel_insert: number;

  // 存储类实例
  key_string_value_json_storage_cls: any;
  vector_db_storage_cls: any;
  graph_storage_cls: any;
  json_doc_status_storage: any;
  llm_response_cache: any;
  full_docs: any;
  text_chunks: any;
  chunk_entity_relation_graph: any;
  entities_vdb: any;
  entity_name_vdb: any;
  relationships_vdb: any;
  chunks_vdb: any;
  doc_status_storage_cls: any;
  doc_status: any;

  /**
   * 构造函数
   */
  constructor(config: Partial<MiniRAG> = {}) {
    // 设置默认值
    this.working_dir = config.working_dir || `./minirag_cache_${format(new Date(), 'yyyy-MM-dd-HH:mm:ss')}`;
    this.kv_storage = config.kv_storage || "JsonKVStorage";
    this.vector_storage = config.vector_storage || "NanoVectorDBStorage";
    this.graph_storage = config.graph_storage || "NetworkXStorage";
    this.current_log_level = logger.level;
    this.log_level = config.log_level || this.current_log_level;

    // 文本分块配置
    this.chunk_token_size = config.chunk_token_size || 1200;
    this.chunk_overlap_token_size = config.chunk_overlap_token_size || 100;
    this.tiktoken_model_name = config.tiktoken_model_name || "gpt-4o-mini";

    // 实体提取配置
    this.entity_extract_max_gleaning = config.entity_extract_max_gleaning || 1;
    this.entity_summary_to_max_tokens = config.entity_summary_to_max_tokens || 500;

    // 节点嵌入配置
    this.node_embedding_algorithm = config.node_embedding_algorithm || "node2vec";
    this.node2vec_params = config.node2vec_params || {
      dimensions: 1536,
      num_walks: 10,
      walk_length: 40,
      window_size: 2,
      iterations: 3,
      random_seed: 3,
    };

    // 嵌入函数配置
    this.embedding_func = config.embedding_func || null;
    this.embedding_batch_num = config.embedding_batch_num || 32;
    this.embedding_func_max_async = config.embedding_func_max_async || 16;

    // LLM配置
    this.llm_model_func = config.llm_model_func || null;
    this.llm_model_name = config.llm_model_name || "meta-llama/Llama-3.2-1B-Instruct";
    this.llm_model_max_token_size = config.llm_model_max_token_size || 32768;
    this.llm_model_max_async = config.llm_model_max_async || 16;
    this.llm_model_kwargs = config.llm_model_kwargs || {};

    // 存储配置
    this.vector_db_storage_cls_kwargs = config.vector_db_storage_cls_kwargs || {};
    this.enable_llm_cache = config.enable_llm_cache !== undefined ? config.enable_llm_cache : true;

    // 扩展配置
    this.addon_params = config.addon_params || {};
    this.convert_response_to_json_func = config.convert_response_to_json_func || convert_response_to_json;

    // 文档状态存储类型
    this.doc_status_storage = config.doc_status_storage || "JsonDocStatusStorage";

    // 自定义分块函数
    this.chunking_func = config.chunking_func || chunking_by_token_size;
    this.chunking_func_kwargs = config.chunking_func_kwargs || {};

    // 最大并行插入数
    this.max_parallel_insert = config.max_parallel_insert || parseInt(process.env.MAX_PARALLEL_INSERT || "2");

    // 初始化
    this._post_init();
  }

  /**
   * 初始化后处理
   */
  private _post_init(): void {
    // 设置日志
    const log_file = path.join(this.working_dir, "minirag.log");
    set_logger(log_file);
    logger.level = this.log_level;

    logger.info(`Logger initialized for working directory: ${this.working_dir}`);
    if (!fs.existsSync(this.working_dir)) {
      logger.info(`Creating working directory ${this.working_dir}`);
      fs.mkdirSync(this.working_dir, { recursive: true });
    }

    // 显示配置
    const global_config = { ...this };
    const _print_config = Object.entries(global_config)
      .map(([k, v]) => `${k} = ${v}`)
      .join(",\n  ");
    logger.debug(`MiniRAG init with param:\n  ${_print_config}\n`);

    // 获取存储类
    this.key_string_value_json_storage_cls = this._get_storage_class(this.kv_storage);
    this.vector_db_storage_cls = this._get_storage_class(this.vector_storage);
    this.graph_storage_cls = this._get_storage_class(this.graph_storage);

    // 绑定全局配置
    const bindGlobalConfig = (fn: any) => {
      return (...args: any[]) => {
        return fn(...args, { global_config });
      };
    };

    this.key_string_value_json_storage_cls = bindGlobalConfig(this.key_string_value_json_storage_cls);
    this.vector_db_storage_cls = bindGlobalConfig(this.vector_db_storage_cls);
    this.graph_storage_cls = bindGlobalConfig(this.graph_storage_cls);

    // 初始化文档状态存储
    this.json_doc_status_storage = this.key_string_value_json_storage_cls(
      "json_doc_status_storage",
      null
    );

    // 创建工作目录
    if (!fs.existsSync(this.working_dir)) {
      logger.info(`Creating working directory ${this.working_dir}`);
      fs.mkdirSync(this.working_dir, { recursive: true });
    }

    // 初始化LLM响应缓存
    this.llm_response_cache = this.enable_llm_cache
      ? this.key_string_value_json_storage_cls(
          "llm_response_cache",
          { ...this },
          null
        )
      : null;

    // 限制嵌入函数的并发调用
    if (this.embedding_func) {
      this.embedding_func = limit_async_func_call(this.embedding_func_max_async)(
        this.embedding_func
      );
    }

    // 初始化全文档存储
    this.full_docs = this.key_string_value_json_storage_cls(
      "full_docs",
      { ...this },
      this.embedding_func
    );

    // 初始化文本块存储
    this.text_chunks = this.key_string_value_json_storage_cls(
      "text_chunks",
      { ...this },
      this.embedding_func
    );

    // 初始化块实体关系图
    this.chunk_entity_relation_graph = this.graph_storage_cls(
      "chunk_entity_relation",
      { ...this },
      this.embedding_func
    );

    // 初始化实体向量数据库
    this.entities_vdb = this.vector_db_storage_cls(
      "entities",
      { ...this },
      this.embedding_func,
      new Set(["entity_name"])
    );

    // 初始化实体名称向量数据库
    this.entity_name_vdb = this.vector_db_storage_cls(
      "entities_name",
      { ...this },
      this.embedding_func,
      new Set(["entity_name"])
    );

    // 初始化关系向量数据库
    this.relationships_vdb = this.vector_db_storage_cls(
      "relationships",
      { ...this },
      this.embedding_func,
      new Set(["src_id", "tgt_id"])
    );

    // 初始化块向量数据库
    this.chunks_vdb = this.vector_db_storage_cls(
      "chunks",
      { ...this },
      this.embedding_func
    );

    // 限制LLM模型函数的并发调用
    if (this.llm_model_func) {
      this.llm_model_func = limit_async_func_call(this.llm_model_max_async)(
        async (...args: any[]) => {
          return await this.llm_model_func(
            ...args,
            { hashing_kv: this.llm_response_cache, ...this.llm_model_kwargs }
          );
        }
      );
    }

    // 初始化文档状态存储
    this.doc_status_storage_cls = this._get_storage_class(this.doc_status_storage);
    this.doc_status = this.doc_status_storage_cls(
      "doc_status",
      { ...this },
      null
    );
  }

  /**
   * 获取存储类
   * @param storage_name 存储名称
   * @returns 存储类
   */
  private _get_storage_class(storage_name: string): any {
    const import_path = STORAGES[storage_name];
    return lazy_external_import(import_path, storage_name);
  }

  /**
   * 设置存储客户端
   * @param db_client 数据库客户端
   */
  set_storage_client(db_client: any): void {
    // 目前仅在Oracle数据库上测试过
    const storages = [
      this.vector_db_storage_cls,
      this.graph_storage_cls,
      this.doc_status,
      this.full_docs,
      this.text_chunks,
      this.llm_response_cache,
      this.key_string_value_json_storage_cls,
      this.chunks_vdb,
      this.relationships_vdb,
      this.entities_vdb,
      this.graph_storage_cls,
      this.chunk_entity_relation_graph,
      this.llm_response_cache,
    ];

    // 设置客户端
    for (const storage of storages) {
      if (storage) {
        storage.db = db_client;
      }
    }
  }

  /**
   * 插入文档
   * @param string_or_strings 字符串或字符串数组
   * @returns 插入结果
   */
  insert(string_or_strings: string | string[]): Promise<void> {
    const loop = always_get_an_event_loop();
    return loop.run_until_complete(this.ainsert(string_or_strings));
  }

  /**
   * 异步插入文档
   * @param input 输入字符串或字符串数组
   * @param split_by_character 按字符分割
   * @param split_by_character_only 仅按字符分割
   * @param ids ID或ID数组
   * @returns 插入结果
   */
  async ainsert(
    input: string | string[],
    split_by_character?: string | null,
    split_by_character_only: boolean = false,
    ids?: string | string[] | null
  ): Promise<void> {
    if (typeof input === 'string') {
      input = [input];
    }
    if (typeof ids === 'string') {
      ids = [ids];
    }

    await this.apipeline_enqueue_documents(input, ids);
    await this.apipeline_process_enqueue_documents(
      split_by_character,
      split_by_character_only
    );

    // 执行额外的实体提取
    const processedDocs = await this.doc_status.get_docs_by_status(DocStatus.PROCESSED);
    const inserting_chunks: Record<string, any> = {};
    
    for (const [doc_id, status_doc] of Object.entries(processedDocs)) {
      const chunks = this.chunking_func(
        status_doc.content,
        this.chunk_overlap_token_size,
        this.chunk_token_size,
        this.tiktoken_model_name
      );
      
      for (const dp of chunks) {
        const chunk_id = compute_mdhash_id(dp.content, "chunk-");
        inserting_chunks[chunk_id] = {
          ...dp,
          full_doc_id: doc_id
        };
      }
    }

    if (Object.keys(inserting_chunks).length > 0) {
      logger.info("Performing entity extraction on newly processed chunks");
      await extract_entities(
        inserting_chunks,
        this.chunk_entity_relation_graph,
        this.entities_vdb,
        this.entity_name_vdb,
        this.relationships_vdb,
        { ...this }
      );
    }

    await this._insert_done();
  }

  /**
   * 文档入队管道
   * @param input 输入字符串或字符串数组
   * @param ids ID数组
   */
  async apipeline_enqueue_documents(
    input: string | string[],
    ids?: string[] | null
  ): Promise<void> {
    if (typeof input === 'string') {
      input = [input];
    }
    if (typeof ids === 'string') {
      ids = [ids];
    }

    let contents: Record<string, string>;
    
    if (ids) {
      if (ids.length !== input.length) {
        throw new Error("Number of IDs must match the number of documents");
      }
      if (new Set(ids).size !== ids.length) {
        throw new Error("IDs must be unique");
      }
      contents = Object.fromEntries(ids.map((id, i) => [id, input[i]]));
    } else {
      const uniqueInput = Array.from(new Set(input.map(doc => clean_text(doc))));
      contents = Object.fromEntries(
        uniqueInput.map(doc => [compute_mdhash_id(doc, "doc-"), doc])
      );
    }

    // 确保内容唯一性
    const contentToId: Record<string, string> = {};
    for (const [id, content] of Object.entries(contents)) {
      contentToId[content] = id;
    }
    
    const unique_contents: Record<string, string> = {};
    for (const [content, id] of Object.entries(contentToId)) {
      unique_contents[id] = content;
    }

    // 创建新文档对象
    const new_docs: Record<string, any> = {};
    for (const [id, content] of Object.entries(unique_contents)) {
      new_docs[id] = {
        content,
        content_summary: get_content_summary(content),
        content_length: content.length,
        status: DocStatus.PENDING,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
    }

    // 过滤已存在的文档
    const all_new_doc_ids = new Set(Object.keys(new_docs));
    const unique_new_doc_ids = await this.doc_status.filter_keys(Array.from(all_new_doc_ids));
    
    const filtered_new_docs: Record<string, any> = {};
    for (const doc_id of unique_new_doc_ids) {
      if (doc_id in new_docs) {
        filtered_new_docs[doc_id] = new_docs[doc_id];
      }
    }

    if (Object.keys(filtered_new_docs).length === 0) {
      logger.info("No new unique documents were found.");
      return;
    }

    await this.doc_status.upsert(filtered_new_docs);
    logger.info(`Stored ${Object.keys(filtered_new_docs).length} new unique documents`);
  }

  /**
   * 处理入队文档
   * @param split_by_character 按字符分割
   * @param split_by_character_only 仅按字符分割
   */
  async apipeline_process_enqueue_documents(
    split_by_character?: string | null,
    split_by_character_only: boolean = false
  ): Promise<void> {
    // 获取待处理文档
    const [processing_docs, failed_docs, pending_docs] = await Promise.all([
      this.doc_status.get_docs_by_status(DocStatus.PROCESSING),
      this.doc_status.get_docs_by_status(DocStatus.FAILED),
      this.doc_status.get_docs_by_status(DocStatus.PENDING),
    ]);

    // 合并所有需要处理的文档
    const to_process_docs: Record<string, any> = {
      ...processing_docs,
      ...failed_docs,
      ...pending_docs,
    };

    if (Object.keys(to_process_docs).length === 0) {
      logger.info("No documents to process");
      return;
    }

    // 将文档分批处理
    const docs_entries = Object.entries(to_process_docs);
    const docs_batches = [];
    for (let i = 0; i < docs_entries.length; i += this.max_parallel_insert) {
      docs_batches.push(docs_entries.slice(i, i + this.max_parallel_insert));
    }
    
    logger.info(`Number of batches to process: ${docs_batches.length}`);

    // 处理每个批次
    for (let batch_idx = 0; batch_idx < docs_batches.length; batch_idx++) {
      for (const [doc_id, status_doc] of docs_batches[batch_idx]) {
        // 将文档分块
        const chunks: Record<string, any> = {};
        const chunksArray = this.chunking_func(
          status_doc.content,
          this.chunk_overlap_token_size,
          this.chunk_token_size,
          this.tiktoken_model_name
        );
        
        for (const dp of chunksArray) {
          const chunk_id = compute_mdhash_id(dp.content, "chunk-");
          chunks[chunk_id] = {
            ...dp,
            full_doc_id: doc_id,
          };
        }

        // 更新存储
        await Promise.all([
          this.chunks_vdb.upsert(chunks),
          this.full_docs.upsert({ [doc_id]: { content: status_doc.content } }),
          this.text_chunks.upsert(chunks),
        ]);

        // 更新文档状态
        await this.doc_status.upsert({
          [doc_id]: {
            status: DocStatus.PROCESSED,
            chunks_count: Object.keys(chunks).length,
            content: status_doc.content,
            content_summary: status_doc.content_summary,
            content_length: status_doc.content_length,
            created_at: status_doc.created_at,
            updated_at: new Date().toISOString(),
          }
        });
      }
    }
    
    logger.info("Document processing pipeline completed");
  }

  /**
   * 插入完成后的回调
   */
  async _insert_done(): Promise<void> {
    const tasks = [];
    const storages = [
      this.full_docs,
      this.text_chunks,
      this.llm_response_cache,
      this.entities_vdb,
      this.entity_name_vdb,
      this.relationships_vdb,
      this.chunks_vdb,
      this.chunk_entity_relation_graph,
    ];

    for (const storage_inst of storages) {
      if (storage_inst) {
        tasks.push((storage_inst as StorageNameSpace).index_done_callback());
      }
    }
    
    await Promise.all(tasks);
  }

  /**
   * 查询
   * @param query 查询字符串
   * @param param 查询参数
   * @returns 查询结果
   */
  query(query: string, param: QueryParam = new QueryParam()): Promise<any> {
    const loop = always_get_an_event_loop();
    return loop.run_until_complete(this.aquery(query, param));
  }

  /**
   * 异步查询
   * @param query 查询字符串
   * @param param 查询参数
   * @returns 查询结果
   */
  async aquery(query: string, param: QueryParam = new QueryParam()): Promise<any> {
    let response;
    
    if (param.mode === "light") {
      response = await hybrid_query(
        query,
        this.chunk_entity_relation_graph,
        this.entities_vdb,
        this.relationships_vdb,
        this.text_chunks,
        param,
        { ...this }
      );
    } else if (param.mode === "mini") {
      response = await minirag_query(
        query,
        this.chunk_entity_relation_graph,
        this.entities_vdb,
        this.entity_name_vdb,
        this.relationships_vdb,
        this.chunks_vdb,
        this.text_chunks,
        this.embedding_func,
        param,
        { ...this }
      );
    } else if (param.mode === "naive") {
      response = await naive_query(
        query,
        this.chunks_vdb,
        this.text_chunks,
        param,
        { ...this }
      );
    } else {
      throw new Error(`Unknown mode ${param.mode}`);
    }
    
    await this._query_done();
    return response;
  }

  /**
   * 查询完成后的回调
   */
  async _query_done(): Promise<void> {
    const tasks = [];
    const storages = [this.llm_response_cache];
    
    for (const storage_inst of storages) {
      if (storage_inst) {
        tasks.push((storage_inst as StorageNameSpace).index_done_callback());
      }
    }
    
    await Promise.all(tasks);
  }

  /**
   * 按实体删除
   * @param entity_name 实体名称
   * @returns 删除结果
   */
  delete_by_entity(entity_name: string): Promise<void> {
    const loop = always_get_an_event_loop();
    return loop.run_until_complete(this.adelete_by_entity(entity_name));
  }

  /**
   * 异步按实体删除
   * @param entity_name 实体名称
   */
  async adelete_by_entity(entity_name: string): Promise<void> {
    entity_name = `"${entity_name.toUpperCase()}"`;

    try {
      await this.entities_vdb.delete_entity(entity_name);
      await this.relationships_vdb.delete_relation(entity_name);
      await this.chunk_entity_relation_graph.delete_node(entity_name);

      logger.info(
        `Entity '${entity_name}' and its relationships have been deleted.`
      );
      await this._delete_by_entity_done();
    } catch (e) {
      logger.error(`Error while deleting entity '${entity_name}': ${e}`);
    }
  }

  /**
   * 按实体删除完成后的回调
   */
  async _delete_by_entity_done(): Promise<void> {
    const tasks = [];
    const storages = [
      this.entities_vdb,
      this.relationships_vdb,
      this.chunk_entity_relation_graph,
    ];
    
    for (const storage_inst of storages) {
      if (storage_inst) {
        tasks.push((storage_inst as StorageNameSpace).index_done_callback());
      }
    }
    
    await Promise.all(tasks);
  }
}

// 导出QueryParam类
export { QueryParam } from './base';