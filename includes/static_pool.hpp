#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <boost/lockfree/queue.hpp>

#include "util.hpp"

namespace thread_pool {

class static_pool final
{ // Thread-safe & container-free thread pool.
public:
    class task_container_base {
    public:
        virtual ~task_container_base() {};
        virtual void operator()() = 0;
    };

    template<typename F>
    class task_container : public task_container_base {
    public:
        task_container(F* f) : f_(f) {};
        ~task_container() {delete f_; };
        void operator()() override { (*f_)(); };
        F* f_;
    };

    struct pool_src {
        boost::lockfree::queue<task_container_base*> queue;
        std::atomic<int> to_finish{0};
        std::atomic<bool> shutdown{false};

        pool_src(std::size_t capacity) : queue(capacity) {}
    };

    explicit static_pool(   
            std::size_t = std::thread::hardware_concurrency(), std::size_t = 1000000);

    ~static_pool();

    template<typename Func, typename... Args>
    auto enqueue(Func &&f, Args &&... args) 
    -> std::future<typename std::result_of<Func(Args...)>::type>;

    static bool wait(const std::shared_ptr<pool_src> &ptr);

private:
    const std::size_t size{0};
    std::shared_ptr<pool_src> m_shared_src;
};

// Implementation:
inline static_pool::static_pool(std::size_t sz, std::size_t capacity)
        : m_shared_src(std::make_shared<pool_src>(capacity)), size(sz)  {
    for (int i = 0; i < sz; ++i) {
        std::thread(
                [this](std::shared_ptr<pool_src> ptr) {
                    while (true) {
                        if (ptr->queue.empty()) {
                            if (ptr->shutdown.load(std::memory_order_acquire))
                                break;
                            std::this_thread::yield(); // Reduce CPU usage
                            continue;
                        }

                        task_container_base* task;
                        ptr->queue.pop(task);
                        (*task)();
                        ptr->to_finish.fetch_sub(1, std::memory_order_release);
                        
                        std::this_thread::yield(); // Reduce CPU usage
                    }
                },
                m_shared_src)
                .detach();
    }
}

inline bool static_pool::wait(const std::shared_ptr<pool_src> &ptr) {
    if (!ptr) {
        return false;
    }

    while (ptr->to_finish.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
    }
}

template<typename Func, typename... Args>
auto static_pool::enqueue(Func &&f, Args &&... args)
-> std::future<typename std::result_of<Func(Args...)>::type> {
    using return_type = typename std::result_of<Func(Args...)>::type;

    // std::packaged_task<return_type()> *task = nullptr;
    // try_allocate(task, std::forward<Func>(f), std::forward<Args>(args)...);
    auto task = new typename std::packaged_task<return_type()>(std::bind(std::forward<Func>(f), std::forward<Args>(args)...));
    auto result = task->get_future();
    auto container = new task_container<std::packaged_task<return_type()>>(task);

    m_shared_src->queue.push(container);
    m_shared_src->to_finish.fetch_add(1, std::memory_order_release);
    return result;
}

static_pool::~static_pool() {
    m_shared_src->shutdown = true;
}

} // namespace thread_pool