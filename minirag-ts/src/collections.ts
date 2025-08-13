/**
 * Counter 类，类似于 Python 的 collections.Counter
 * 用于计数元素出现的次数
 */
export class Counter<T extends string | number> {
  private counts: Map<T, number>;

  constructor(iterable?: Iterable<T>) {
    this.counts = new Map<T, number>();
    
    if (iterable) {
      for (const item of iterable) {
        this.add(item);
      }
    }
  }

  /**
   * 添加一个元素，增加其计数
   * @param item 要添加的元素
   * @param count 计数增加值，默认为1
   */
  add(item: T, count: number = 1): void {
    const currentCount = this.counts.get(item) || 0;
    this.counts.set(item, currentCount + count);
  }

  /**
   * 获取元素的计数
   * @param item 要查询的元素
   * @returns 元素的计数，如果不存在则返回0
   */
  get(item: T): number {
    return this.counts.get(item) || 0;
  }

  /**
   * 返回计数最多的n个元素及其计数
   * @param n 要返回的元素数量
   * @returns 元素和计数的数组，按计数降序排列
   */
  mostCommon(n: number = 1): [T, number][] {
    return Array.from(this.counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, n);
  }

  /**
   * 获取所有元素
   * @returns 所有元素的数组
   */
  elements(): T[] {
    const result: T[] = [];
    for (const [item, count] of this.counts.entries()) {
      for (let i = 0; i < count; i++) {
        result.push(item);
      }
    }
    return result;
  }

  /**
   * 获取计数的总和
   * @returns 所有计数的总和
   */
  total(): number {
    let sum = 0;
    for (const count of this.counts.values()) {
      sum += count;
    }
    return sum;
  }
}