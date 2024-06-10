# 🍏 对象生存期和资源管理|RAII设计思想

### 引出RAII

<mark style="color:red;">与Python不同，C++ 没有自动回收垃圾，这是在程序运行时释放堆内存和其他资源的 一个内部进程。</mark> C++ 程序负责将所有已获取的资源返回到操作系统。 未能释放未使用的 资源称为“泄漏”。 在进程退出之前，泄漏的资源无法用于其他程序。 特别是内存泄漏是 C / C++编程中 bug 的常见原因。

<mark style="color:red;">新式 C++ 通过声明栈上的对象，尽可能避免使用堆内存。 当某个资源对于栈来说太 大时，则它应由对象拥有。 当该对象初始化时，它会获取它拥有的资源。 然后，该对象 负责在其析构函数中释放资源。 在栈上声明拥有资源的对象本身。 对象拥有资源的 则也称为“资源获取即初始化”(RAII)。</mark>&#x20;

当拥有资源的栈对象超出范围时，会自动调用其析构函数。 这样，<mark style="color:red;">C++ 中的垃圾回收 与对象生存期密切相关，是确定性的。</mark> 资源始终在程序中的已知点发布，你可以控制该 点。 仅类似 C++ 中的确定析构函数可公平处理内存和非内存资源。

### RAII 的好处

1. **自动资源管理**：资源的获取和释放绑定在对象的构造与析构中，减少了手动管理资源的负担。
2. **异常安全**：在异常发生时，析构函数会自动调用，确保资源得到正确释放。
3. **简化代码**：减少显式的资源管理代码，使代码更简洁、更容易维护。
4. **提升代码可读性**：资源管理逻辑清晰地嵌入到对象的生命周期中，提高代码的可读性。

### 一些自定义的RAII类

#### 1. 内存管理

虽然智能指针已经很好的解决了内存管理的问题，但在一些特定的场景中，我们可能需要自定义内存管理类。

**示例：自定义内存块管理类**

```cpp
#include <iostream>
#include <cstring>

class MemoryBlock {
public:
    MemoryBlock(size_t size) {
        data_ = new char[size]; // 分配内存
        size_ = size;
        std::cout << "MemoryBlock of size " << size << " allocated." << std::endl;
    }

    ~MemoryBlock() {
        delete[] data_; // 释放内存
        std::cout << "MemoryBlock of size " << size_ << " deallocated." << std::endl;
    }

    // 禁止拷贝
    MemoryBlock(const MemoryBlock&) = delete;
    MemoryBlock& operator=(const MemoryBlock&) = delete;

    // 允许移动
    MemoryBlock(MemoryBlock&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    MemoryBlock& operator=(MemoryBlock&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    char* data() { return data_; }
    size_t size() const { return size_; }

private:
    char* data_;
    size_t size_;
};

int main() {
    MemoryBlock block1(1024); // 分配1KB的内存块
    MemoryBlock block2(2048); // 分配2KB的内存块

    // 将block1的资源转移到block3
    MemoryBlock block3(std::move(block1));

    return 0;
}
```

#### 2. 线程锁管理

在多线程编程中，使用RAII管理锁可以确保在所有代码路径上都能正确释放锁，避免死锁的发生。

**示例：自定义锁管理类**

```cpp
#include <iostream>
#include <mutex>

class LockGuard {
public:
    explicit LockGuard(std::mutex& m) : mutex_(m) {
        mutex_.lock(); // 获取锁
        std::cout << "Lock acquired." << std::endl;
    }

    ~LockGuard() {
        mutex_.unlock(); // 释放锁
        std::cout << "Lock released." << std::endl;
    }

    // 禁止拷贝和赋值
    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;

private:
    std::mutex& mutex_;
};

int main() {
    std::mutex mtx;

    {
        LockGuard lock(mtx); // 自动获取和释放锁
        std::cout << "Critical section." << std::endl;
    } // 离开作用域时自动释放锁

    return 0;
}
```

#### 3. 文件描述符管理

管理文件描述符或套接字等资源，确保在所有代码路径上都能正确关闭资源。

**示例：自定义文件描述符管理类**

```cpp
#include <iostream>
#include <unistd.h> // for close()

class FileDescriptor {
public:
    explicit FileDescriptor(int fd) : fd_(fd) {
        std::cout << "FileDescriptor " << fd << " acquired." << std::endl;
    }

    ~FileDescriptor() {
        if (fd_ != -1) {
            close(fd_); // 关闭文件描述符
            std::cout << "FileDescriptor " << fd_ << " released." << std::endl;
        }
    }

    // 禁止拷贝
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    // 允许移动
    FileDescriptor(FileDescriptor&& other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }

    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            if (fd_ != -1) {
                close(fd_);
            }
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    int get() const { return fd_; }

private:
    int fd_;
};

int main() {
    int fd = open("example.txt", O_RDONLY);
    if (fd == -1) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    FileDescriptor file(fd); // 自动管理文件描述符

    // 使用文件描述符进行操作

    return 0;
}
```

#### 4. 网络资源管理

在网络编程中，管理套接字的生命周期，确保在所有代码路径上都能正确关闭套接字。

**示例：自定义套接字管理类**

```cpp
#include <iostream>
#include <sys/socket.h>
#include <unistd.h> // for close()

class Socket {
public:
    explicit Socket(int sock) : sock_(sock) {
        std::cout << "Socket " << sock << " acquired." << std::endl;
    }

    ~Socket() {
        if (sock_ != -1) {
            close(sock_); // 关闭套接字
            std::cout << "Socket " << sock_ << " released." << std::endl;
        }
    }

    // 禁止拷贝
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    // 允许移动
    Socket(Socket&& other) noexcept : sock_(other.sock_) {
        other.sock_ = -1;
    }

    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) {
            if (sock_ != -1) {
                close(sock_);
            }
            sock_ = other.sock_;
            other.sock_ = -1;
        }
        return *this;
    }

    int get() const { return sock_; }

private:
    int sock_;
};

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        std::cerr << "Failed to create socket." << std::endl;
        return 1;
    }

    Socket socket(sock); // 自动管理套接字

    // 使用套接字进行操作

    return 0;
}
```

#### 5. 临时文件管理

在一些应用程序中，临时文件需要在程序结束时自动删除，使用RAII可以方便地管理这种资源。

**示例：自定义临时文件管理类**

```cpp
#include <iostream>
#include <cstdio>
#include <stdexcept>

class TempFile {
public:
    TempFile(const std::string& filename) : filename_(filename) {
        file_ = std::fopen(filename.c_str(), "w+");
        if (!file_) {
            throw std::runtime_error("Failed to open temporary file");
        }
        std::cout << "Temporary file " << filename << " created." << std::endl;
    }

    ~TempFile() {
        if (file_) {
            std::fclose(file_);
            std::remove(filename_.c_str());
            std::cout << "Temporary file " << filename_ << " deleted." << std::endl;
        }
    }

    // 禁止拷贝
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    // 允许移动
    TempFile(TempFile&& other) noexcept : file_(other.file_), filename_(std::move(other.filename_)) {
        other.file_ = nullptr;
    }

    TempFile& operator=(TempFile&& other) noexcept {
        if (this != &other) {
            if (file_) {
                std::fclose(file_);
                std::remove(filename_.c_str());
            }
            file_ = other.file_;
            filename_ = std::move(other.filename_);
            other.file_ = nullptr;
        }
        return *this;
    }

    FILE* get() const { return file_; }

private:
    FILE* file_;
    std::string filename_;
};

int main() {
    try {
        TempFile tempFile("temp.txt"); // 自动管理临时文件

        // 使用 tempFile.get() 进行文件操作
        std::fprintf(tempFile.get(), "Temporary data\n");

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

通过自定义RAII类，可以大大简化资源管理的代码，提高程序的健壮性和可维护性。

### reference

* [https://learn.microsoft.com/en-us/cpp/cpp/?view=msvc-170](https://learn.microsoft.com/en-us/cpp/cpp/?view=msvc-170)
