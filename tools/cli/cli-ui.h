#pragma once

#include "common.h"
#include "console.h"

#include <array>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string_view>

// TODO?: Make this reusable, enums, docs
static const std::array<std::string_view, 8> cmds = {
    "/audio ",
    "/clear",
    "/exit",
    "/glob ",
    "/image ",
    "/read ",
    "/regen",
    "/video ",
};

static std::vector<std::pair<std::string, size_t>> auto_completion_callback(std::string_view line, size_t cursor_byte_pos) {
    std::vector<std::pair<std::string, size_t>> matches;
    std::string cmd;

    if (line.length() > 1 && line.front() == '/' && !std::any_of(cmds.begin(), cmds.end(), [line](std::string_view prefix) {
        return string_starts_with(line, prefix);
    })) {
        auto it = cmds.begin();

        while ((it = std::find_if(it, cmds.end(), [line](std::string_view cmd_line) {
            return string_starts_with(cmd_line, line);
        })) != cmds.end()) {
            matches.emplace_back(*it, it->length());
            ++it;
        }
    } else {
        auto it = std::find_if(cmds.begin(), cmds.end(), [line](std::string_view prefix) {
            return prefix.back() == ' ' && string_starts_with(line, prefix);
        });

        if (it != cmds.end()) {
            cmd = *it;
        }
    }

    if (!cmd.empty() && cmd != "/glob " && line.length() >= cmd.length() && cursor_byte_pos >= cmd.length()) {
        const std::string path_prefix  = std::string(line.substr(cmd.length(), cursor_byte_pos - cmd.length()));
        const std::string path_postfix = std::string(line.substr(cursor_byte_pos));
        auto cur_dir = std::filesystem::current_path();
        std::string cur_dir_str = cur_dir.string();
        std::string expanded_prefix = path_prefix;

#if !defined(_WIN32)
        if (string_starts_with(path_prefix, '~')) {
            const char * home = std::getenv("HOME");
            if (home && home[0]) {
                expanded_prefix = home + path_prefix.substr(1);
            }
        }
        if (string_starts_with(expanded_prefix, '/')) {
#else
        if (std::isalpha(static_cast<unsigned char>(expanded_prefix[0])) && expanded_prefix.find(':') == 1) {
#endif
            cur_dir = std::filesystem::path(expanded_prefix).parent_path();
            cur_dir_str.clear();
        } else if (!path_prefix.empty()) {
            cur_dir /= std::filesystem::path(path_prefix).parent_path();
        }

        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(cur_dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.exists(ec)) {
                ec.clear();
                continue;
            }

            const std::string path_full = entry.path().string();
            std::string path_entry = !cur_dir_str.empty() && string_starts_with(path_full, cur_dir_str) ? path_full.substr(cur_dir_str.length() + 1) : path_full;

            if (entry.is_directory(ec)) {
                path_entry.push_back(std::filesystem::path::preferred_separator);
            }

            if (expanded_prefix.empty() || string_starts_with(path_entry, expanded_prefix)) {
                const std::string updated_line = cmd + path_entry;
                matches.emplace_back(updated_line + path_postfix, updated_line.length());
            }

            if (ec) {
                ec.clear();
            }
        }

        if (matches.empty()) {
            const std::string updated_line = cmd + path_prefix;
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        // Add the longest common prefix
        if (!expanded_prefix.empty() && matches.size() > 1) {
            const std::string_view match0(matches[0].first);
            const std::string_view match1(matches[1].first);
            auto it = std::mismatch(match0.begin(), match0.end(), match1.begin(), match1.end());
            size_t len = it.first - match0.begin();

            for (size_t i = 2; i < matches.size(); ++i) {
                const std::string_view matchi(matches[i].first);
                auto cmp = std::mismatch(match0.begin(), match0.end(), matchi.begin(), matchi.end());
                len = std::min(len, static_cast<size_t>(cmp.first - match0.begin()));
            }

            const std::string updated_line = std::string(match0.substr(0, len));
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        std::sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) {
            return a.first.compare(0, a.second, b.first, 0, b.second) < 0;
        });
    }

    return matches;
}

namespace ui {
    static bool use_color = false;
    static bool simple_io = false;
    static constexpr const char * color_accent = "\033[1;38;2;152;30;50m";
    static constexpr const char * color_amber = "\033[38;2;237;170;67m";
    static constexpr const char * color_muted = "\033[2;38;2;170;150;141m";
    static constexpr const char * color_timing = "\033[38;2;170;99;48m";

    static void reset_display() {
        console::set_display(DISPLAY_TYPE_RESET);
        // The console cache does not track our private colors.
        if (use_color) {
            console::log("\033[0m");
        }
    }

    static void set_style(const char * color) {
        reset_display();
        if (use_color) {
            console::log("%s", color);
        }
    }

    static void init(const common_params & params) {
        simple_io = params.simple_io;
        use_color = params.use_color && !simple_io;
        // TODO: avoid using atexit() here by making `console` a singleton
        console::init(simple_io, use_color);
        atexit([]() { reset_display(); console::cleanup(); });

        console::set_completion_callback(auto_completion_callback);
    }

    struct spinner {
        spinner(std::string_view message) {
            if (!message.empty()) {
                set_style(color_amber);
                console::log("%.*s ", int(message.size()), message.data());
                reset_display();
                console::flush();
            }
            console::spinner::start();
        }
        ~spinner() {
            console::spinner::stop();
            reset_display();
        }
    };

    struct user_turn {
        user_turn() {
            reset_display();
        }
        ~user_turn() {
            reset_display();
            console::flush();
        }
        void print_prompt() {
            console::log("\n");
            set_style(color_amber);
            console::log("%s", simple_io ? "> " : "You > ");
            reset_display();
        }
        void echo(const std::string & buffer) {
            print_prompt();
            if (buffer.size() > 500) {
                console::log("%s ... (truncated)\n", buffer.substr(0, 500).c_str());
            } else {
                console::log("%s\n", buffer.c_str());
            }
        }
        std::string read_input(bool multiline_input, const char * prompt = nullptr) {
            if (prompt) {
                set_style(color_amber);
                console::log("%s", prompt);
                reset_display();
            } else {
                print_prompt();
            }
            console::flush();
            std::string buffer;
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, multiline_input);
                buffer += line;
            } while (another_line);
            return buffer;
        }
    };

    enum assistant_display_mode {
        ASSISTANT_DISPLAY_MODE_REASONING,
        ASSISTANT_DISPLAY_MODE_CONTENT,
    };
    struct assistant_turn {
        assistant_display_mode mode = ASSISTANT_DISPLAY_MODE_CONTENT;
        bool trailing_newline = true;
        assistant_turn() {
            set_style(color_accent);
            console::log("\nAssistant:\n");
            reset_display();
            console::flush();
        }
        ~assistant_turn() {
            add_newline_if_needed();
            reset_display();
            console::flush();
        }
        void push(assistant_display_mode m, const std::string & buffer) {
            if (buffer.empty()) {
                return;
            }
            if (m != mode) {
                add_newline_if_needed();
                switch (m) {
                    case ASSISTANT_DISPLAY_MODE_CONTENT:
                        set_style(color_amber);
                        console::log("Answer:\n");
                        reset_display();
                        break;
                    case ASSISTANT_DISPLAY_MODE_REASONING:
                        set_style(color_muted);
                        console::log("Reasoning:\n");
                        break;
                }
                mode = m;
            }
            trailing_newline = buffer.back() == '\n';
            console::log("%s", buffer.c_str());
            console::flush();
        }
        void add_newline_if_needed() {
            if (!trailing_newline) {
                console::log("\n");
                console::flush();
                trailing_newline = true;
            }
        }
    };

    static void show_error(const std::string & title, const std::string & message = "") {
        console::spinner::stop();
        reset_display();
        console::error("Error: %s\n", title.c_str());
        if (!message.empty()) {
            console::log("%s\n", message.c_str());
        }
        reset_display();
    }

    static void show_message(const std::string & message) {
        console::log("%s\n", message.c_str());
    }

    static void show_timings(double prompt_per_second, double predicted_per_second, double elapsed_seconds) {
        set_style(color_timing);
        console::log("  %.1f tok/s | %.1fs | prompt %.1f tok/s\n",
            predicted_per_second, elapsed_seconds, prompt_per_second);
        reset_display();
        console::flush();
    }
}
