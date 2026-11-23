#include "ink2_embedded.h"

#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>

namespace ink2 {

struct embedded_server::impl {
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> input;
    std::deque<std::string> output;
    std::thread thread;
    bool input_closed = false;
    bool output_closed = false;
    int exit_code = -1;
};

static std::mutex active_mutex;
static std::weak_ptr<embedded_server::impl> active_channel;

static std::shared_ptr<embedded_server::impl> get_active_channel() {
    std::lock_guard<std::mutex> lock(active_mutex);
    return active_channel.lock();
}

embedded_server::embedded_server() = default;

embedded_server::~embedded_server() {
    stop();
}

bool embedded_server::start(const std::vector<std::string> & args) {
    if (p || args.empty()) {
        return false;
    }
    p = std::make_shared<impl>();
    {
        std::lock_guard<std::mutex> lock(active_mutex);
        if (!active_channel.expired()) {
            p.reset();
            return false;
        }
        active_channel = p;
    }

    const auto channel = p;
    channel->thread = std::thread([channel, args]() {
        std::vector<char *> argv;
        argv.reserve(args.size());
        for (const auto & arg : args) {
            argv.push_back(const_cast<char *>(arg.c_str()));
        }
        const int argc = (int) argv.size();
        argv.push_back(nullptr);
        int exit_code = 1;
        try {
            exit_code = engine_main(argc, argv.data());
        } catch (const std::exception & e) {
            std::cerr << "inkling2 engine: " << e.what() << '\n';
        } catch (...) {
            std::cerr << "inkling2 engine: unknown failure\n";
        }
        {
            std::lock_guard<std::mutex> lock(channel->mutex);
            channel->exit_code = exit_code;
            channel->output_closed = true;
        }
        channel->cv.notify_all();
    });
    return true;
}

bool embedded_server::write_line(const std::string & line) {
    if (!p) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(p->mutex);
        if (p->input_closed || p->output_closed) {
            return false;
        }
        std::string input = line;
        while (!input.empty() && (input.back() == '\n' || input.back() == '\r')) {
            input.pop_back();
        }
        p->input.push_back(std::move(input));
    }
    p->cv.notify_all();
    return true;
}

bool embedded_server::read_line(std::string & line) {
    if (!p) {
        return false;
    }
    std::unique_lock<std::mutex> lock(p->mutex);
    p->cv.wait(lock, [&]() { return p->output_closed || !p->output.empty(); });
    if (p->output.empty()) {
        return false;
    }
    line = std::move(p->output.front());
    p->output.pop_front();
    return true;
}

bool embedded_server::alive() const {
    if (!p) {
        return false;
    }
    std::lock_guard<std::mutex> lock(p->mutex);
    return !p->output_closed;
}

void embedded_server::stop() {
    if (!p) {
        return;
    }
    write_line("QUIT");
    {
        std::lock_guard<std::mutex> lock(p->mutex);
        p->input_closed = true;
    }
    p->cv.notify_all();
    if (p->thread.joinable()) {
        p->thread.join();
    }
    {
        std::lock_guard<std::mutex> lock(active_mutex);
        if (active_channel.lock() == p) {
            active_channel.reset();
        }
    }
    p.reset();
}

bool embedded_active() {
    return (bool) get_active_channel();
}

bool embedded_read_line(std::string & line) {
    const auto channel = get_active_channel();
    if (!channel) {
        return false;
    }
    std::unique_lock<std::mutex> lock(channel->mutex);
    channel->cv.wait(lock, [&]() { return channel->input_closed || !channel->input.empty(); });
    if (channel->input.empty()) {
        return false;
    }
    line = std::move(channel->input.front());
    channel->input.pop_front();
    return true;
}

void embedded_write_line(const std::string & line) {
    const auto channel = get_active_channel();
    if (!channel) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(channel->mutex);
        if (channel->output_closed) {
            return;
        }
        channel->output.push_back(line);
    }
    channel->cv.notify_all();
}

}
