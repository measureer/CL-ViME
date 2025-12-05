import time
from functools import wraps

def mytimer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用原函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算运行时间
        print(f"{func.__name__}运行时间: {elapsed_time:.4f}秒")  # 输出运行时间
        return result
    return wrapper





# 示例用法
@mytimer
def example_function():
    time.sleep(1)  # 模拟耗时操作

if __name__ == '__main__':
    # 调用示例函数
    example_function(1000000)
