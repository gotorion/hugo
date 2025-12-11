+++
title = "Triton分析之一：compiler"
date = "2025-11-29"
author = "李俊辉"
tags = ["Triton", "compiler"]
categories = ["技术"]
+++


## JIT实现

在使用Triton编写kernel的时候，首先需要`@triton.jit`修饰我们的kernel，这个修饰符本质上是把函数变成了一个`JITFunction`。

```python
@overload
def jit(fn: T) -> JITFunction[T]:
    ...

class JITFunction(KernelInterface[T]):
    ...

class KernelInterface(Generic[T]):
    run: T

    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
```

Python中用`obj[key]`索引的时候，其实是调用class的`__getitem__`方法。启动kernel的时候用`kernel[grid](...)`语法传入grid，通过调用`__getitem__`方法，返回一个lambda函数，在lambda函数调用`run`方法来启动。

`JITFunction`类几个主要的方法如下：

1. `run`,执行kernel时候调用的方法；
2. `warmup`，调用run方法，用于提前编译kernel；
3. `cache_key`，这是一个只读方法，用于获取缓存kernel的key；
4. `add_pre_run_hook`&`_call_hook`，自定义一些钩子；

主要的方法是`run`，这里大概做了几件事情:

```python
def run(...):
    # 获取kernel执行的device和stream
    # 执行一些hook函数
    # 计算cache key
    # 如果kernel没有被编译缓存，编译kernel，这里会比较耗时；
    if kernel is None:
        src = self.ASTSource(self, signature, constexprs, attrs)
            kernel = self.compile(src, target=target, options=options.__dict__)
            kernel_cache[key] = kernel
            self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=False)
    # 如果run不是被warmup调用的，计算grid并启动kernel计算，否则把之前编译好的kernel返回给warmup；
```

可以看到在compile的时候，Triton源码先是被解析成AST，然后调用compile方法去编译。

## compile
