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
> 删除 1、2

## Task B

```md
将 minirag/operate.py 重构为模块化结构：将其功能拆分成多个逻辑独立的子模块文件，统一放入新建的 minirag/operate/ 目录中。要求保持原有接口不变，确保外部代码无需修改导入语句即可继续使用。具体包括：1) 合理划分功能模块到单独文件 2) 在 minirag/operate/__init__.py 中维护原有导出关系 3) 保留原 minirag/operate.py 作为兼容层，仅包含必要的导入和转发逻辑。注意保持代码风格和文档的一致性。
```

## Task C?

```md
将 minirag/ 目录下的所有 Python (.py) 代码文件转换为 TypeScript (.ts) 格式，并将转换后的文件保存到 minirag-ts/ 目录中。

1. 确保类型定义准确（保持变量名、方法原本的驼峰/下划线命名方式），保留原有代码逻辑和功能，同时遵循 TypeScript 的最佳实践。
2. 处理所有依赖导入和模块导出，使转换后的代码可直接运行。
3. 无需安装未安装的依赖库

你此前已经处理了一些工作，现存的 ts 已经完成了翻译，请继续完成剩余的工作。
```
