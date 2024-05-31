# 🫑 emplace\_back 减少内存拷贝和移动|C++11

对于STL容器，C++11后引入了emplace\_back接口。 emplace\_back是就地构造，不用构造后再次复制到容器中。因此效率更高。 考虑这样的语句：

```cpp
vector<string> testVec;
testVec.push_back(string(16, 'a'));
```

上述语句足够简单易懂，将一个string对象添加到testVec中。底层实现：

* 首先，string(16, ‘a’)会创建一个string类型的临时对象，这涉及到一次string构造过程。
* 其次，vector内会创建一个新的string对象，这是第二次构造。
* 最后在push\_back结束时，最开始的临时对象会被析构。加在一起，这两行代码会涉及到两次 string构造和一次析构。

`c++11`可以用`emplace_back`代替`push_back`，`emplace_back`可以直接在`vector`中构建一个对象，而非创建一个临时对象，再放进`vector`，再销毁。`emplace_back`可以省略一次构建和一次析构，从而达到优化的目的.
