import { z } from 'zod';

/**
 * 这是一个用于定义自定义语言模型的Model类
 */
export class Model {
  /**
   * 一个从语言模型生成响应的可调用函数。该函数必须返回一个字符串
   */
  gen_func: (args: any) => Promise<string>;
  
  /**
   * 传递给可调用函数的参数。例如：api密钥、模型名称等
   */
  kwargs: Record<string, any>;

  constructor(gen_func: (args: any) => Promise<string>, kwargs: Record<string, any>) {
    this.gen_func = gen_func;
    this.kwargs = kwargs;
  }
}

/**
 * 在多个语言模型之间分配负载。对于规避某些API提供商的低速率限制特别有用，尤其是如果您使用的是免费层级。
 * 也可用于在不同模型或提供商之间进行分割。
 */
export class MultiModel {
  private _models: Model[];
  private _current_model: number;

  /**
   * @param models 要使用的语言模型列表
   */
  constructor(models: Model[]) {
    this._models = models;
    this._current_model = 0;
  }

  private _next_model(): Model {
    this._current_model = (this._current_model + 1) % this._models.length;
    return this._models[this._current_model];
  }

  async llm_model_func(
    prompt: string, 
    system_prompt: string | null = null, 
    history_messages: any[] = [], 
    ...kwargs: any[]
  ): Promise<string> {
    const options: Record<string, any> = Object.assign({}, ...kwargs);
    delete options.model;  // 防止覆盖自定义模型名称
    delete options.keyword_extraction;
    delete options.mode;
    
    const next_model = this._next_model();
    const args = {
      prompt,
      system_prompt,
      history_messages,
      ...options,
      ...next_model.kwargs,
    };

    return await next_model.gen_func(args);
  }
}

// 如果作为主模块运行
if (require.main === module) {
  const main = async () => {
    const { gpt_4o_mini_complete } = await import('./llm/openai');
    const result = await gpt_4o_mini_complete("How are you?");
    console.log(result);
  };

  main().catch(console.error);
}