#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifdef CT_HAS_OPENMP
#include <omp.h>
#endif

namespace ct {

inline int get_default_thread_count() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) return 4;
    // On systems with high core counts or hybrid P/E core architectures (e.g. Apple Silicon M-series),
    // cap default threads to 8 (matching the physical performance cores) to prevent thread
    // oversubscription on efficiency cores during interactive or background operations.
    if (hw > 8) return 8;
    return static_cast<int>(hw);
}

inline int& global_thread_count() {
    static int threads = get_default_thread_count();
    return threads;
}

inline void set_num_threads(int n) {
    if (n > 0) {
        global_thread_count() = n;
#ifdef CT_HAS_OPENMP
        omp_set_num_threads(n);
#endif
    }
}

inline int get_num_threads() {
#ifdef CT_HAS_OPENMP
    return omp_get_max_threads();
#else
    return global_thread_count();
#endif
}

// Thread-local indicator to prevent nested parallel regions from spawning threads
inline thread_local bool t_in_parallel_region = false;

#ifndef CT_HAS_OPENMP

class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    ~ThreadPool() {
        shutdown();
    }

    template <typename Index, typename Func>
    void parallel_for(Index begin, Index end, Func&& fn, int64_t min_chunk = 1) {
        int num_threads = global_thread_count();
        int64_t total_items = static_cast<int64_t>(end - begin);

        if (total_items <= 0) return;
        if (num_threads <= 1 || total_items <= 1 || t_in_parallel_region) {
            for (Index i = begin; i < end; ++i) {
                fn(i);
            }
            return;
        }

        std::unique_lock<std::mutex> call_lock(pool_call_mutex_);

        ensure_threads(num_threads);

        int active_workers = std::min(num_threads, static_cast<int>(total_items)) - 1;
        if (active_workers <= 0) {
            for (Index i = begin; i < end; ++i) {
                fn(i);
            }
            return;
        }

        std::atomic<int64_t> current_index(static_cast<int64_t>(begin));
        int64_t end_val = static_cast<int64_t>(end);
        int total_threads = active_workers + 1;

        auto task = [&](int /*worker_id*/) {
            t_in_parallel_region = true;
            while (true) {
                int64_t curr = current_index.load(std::memory_order_relaxed);
                if (curr >= end_val) break;

                int64_t remaining = end_val - curr;
                // Guided scheduling chunk size for dynamic load balancing between P and E cores
                int64_t chunk = std::max(min_chunk, remaining / (2 * total_threads));
                if (chunk <= 0) chunk = 1;

                int64_t start = current_index.fetch_add(chunk, std::memory_order_relaxed);
                if (start >= end_val) break;
                int64_t stop = std::min(end_val, start + chunk);

                for (int64_t i = start; i < stop; ++i) {
                    fn(static_cast<Index>(i));
                }
            }
            t_in_parallel_region = false;
        };

        // Dispatch task to persistent worker threads
        {
            std::unique_lock<std::mutex> lock(mutex_);
            current_task_ = task;
            active_workers_ = active_workers;
            completed_workers_ = 0;
            ++job_id_;
            cv_work_.notify_all();
        }

        // Master thread participates in chunk execution
        task(0);

        // Wait for worker threads to complete their assigned chunks
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [this]() {
                return completed_workers_ == active_workers_;
            });
            current_task_ = nullptr;
        }
    }

private:
    ThreadPool() : stopping_(false), active_workers_(0), completed_workers_(0), job_id_(0) {}

    void ensure_threads(int count) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (workers_.size() < static_cast<size_t>(count - 1)) {
            int id = static_cast<int>(workers_.size()) + 1;
            workers_.emplace_back([this, id]() {
                worker_loop(id);
            });
        }
    }

    void worker_loop(int id) {
        uint64_t last_job_id = 0;
        while (true) {
            std::function<void(int)> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_work_.wait(lock, [this, id, last_job_id]() {
                    return stopping_ || (job_id_ > last_job_id && (id <= active_workers_));
                });

                if (stopping_) return;

                if (job_id_ > last_job_id && id <= active_workers_) {
                    last_job_id = job_id_;
                    task = current_task_;
                }
            }

            if (task) {
                task(id);
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    ++completed_workers_;
                    if (completed_workers_ == active_workers_) {
                        cv_done_.notify_one();
                    }
                }
            }
        }
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stopping_ = true;
            cv_work_.notify_all();
        }
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::mutex pool_call_mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::function<void(int)> current_task_;
    bool stopping_;
    int active_workers_;
    int completed_workers_;
    uint64_t job_id_;
};

#endif  // CT_HAS_OPENMP

template <typename Index, typename Func>
void parallel_for(Index begin, Index end, Func&& fn, int64_t work_per_item = 1024, int64_t min_total_work = 4096) {
    if (begin >= end) return;
    int64_t total_items = static_cast<int64_t>(end - begin);

    // If already inside a parallel region, or single thread requested, or work is below threshold
    if (t_in_parallel_region || global_thread_count() <= 1 || total_items <= 1 ||
        (total_items * work_per_item) < min_total_work) {
        for (Index i = begin; i < end; ++i) {
            fn(i);
        }
        return;
    }

#ifdef CT_HAS_OPENMP
    t_in_parallel_region = true;
    #pragma omp parallel for schedule(guided)
    for (Index i = begin; i < end; ++i) {
        fn(i);
    }
    t_in_parallel_region = false;
#else
    // Dynamic/guided block scheduling: minimum chunk size ensures cache block efficiency
    int64_t min_chunk = std::max(int64_t(1), int64_t(2048 / std::max(int64_t(1), work_per_item)));
    ThreadPool::instance().parallel_for(begin, end, std::forward<Func>(fn), min_chunk);
#endif
}

}  // namespace ct
