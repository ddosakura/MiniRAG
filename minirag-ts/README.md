# minirag-ts

> [CodeBuddy](https://www.codebuddy.ai/) + Claude-3.7-Sonnet

## Task A

```md
将 minirag/ 目录下的所有 Python (.py) 代码文件转换为 TypeScript (.ts) 格式，并将转换后的文件保存到 minirag-ts/ 目录中。

1. 确保类型定义准确（保持变量名、方法原本的驼峰/下划线命名方式），保留原有代码逻辑和功能，同时遵循 TypeScript 的最佳实践。
2. 处理所有依赖导入和模块导出，使转换后的代码可直接运行。
```

> 任务过长，已终止。存在问题：
> 1. 错误的生成了 minirag-ts/src/types/numpy-js.d.ts
> 2. minirag-ts/src/operate.ts 大文件生成异常
> 3. minirag-ts/src/prompt.ts 提示文本内容被篡改
> 需要处理 1、2
