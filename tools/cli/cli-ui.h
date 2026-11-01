#pragma once

#include "common.h"
#include "console.h"
#include "cli-markdown.h"
#include "betelgeuse-terminal-logo.h"

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

// Autocompletion commands
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
        const auto * it = std::find_if(cmds.begin(), cmds.end(), [line](std::string_view prefix) {
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

    // Palette: Betelgeuse deep red, amber, warm gold, muted rose-gray
    static constexpr const char * color_crimson = "\033[1;38;2;170;30;45m";
    static constexpr const char * color_amber   = "\033[1;38;2;237;170;67m";
    static constexpr const char * color_gold    = "\033[38;2;245;200;110m";
    static constexpr const char * color_muted   = "\033[38;2;160;140;135m";
    static constexpr const char * color_dark    = "\033[38;2;110;50;45m";
    static constexpr const char * color_subtle  = "\033[38;2;140;125;120m";

    static std::string g_model_name = "model";
    static std::string g_mode_name  = "local";
    static int64_t g_ctx_capacity   = 0;
    static int64_t g_ctx_used       = 0;

    static int get_terminal_width() {
#if defined(_WIN32)
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(hOut, &csbi)) {
            int w = csbi.srWindow.Right - csbi.srWindow.Left + 1;
            if (w > 20) return w;
        }
#else
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 20) {
            return ws.ws_col;
        }
#endif
        return 80;
    }

    static void set_session_info(const std::string & model, bool is_local, int64_t ctx_cap) {
        g_model_name = model;
        const std::string_view suffixes[] = {"-Dense.gguf", ".gguf", "-GSQ-RCO-IQ2_XS"};
        for (const auto & sfx : suffixes) {
            if (g_model_name.size() > sfx.size() && g_model_name.compare(g_model_name.size() - sfx.size(), sfx.size(), sfx) == 0) {
                g_model_name.erase(g_model_name.size() - sfx.size());
            }
        }
        g_mode_name  = is_local ? "local (sm_86)" : "remote";
        g_ctx_capacity = ctx_cap;
    }

    static void reset_display() {
        console::set_display(DISPLAY_TYPE_RESET);
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
        console::init(simple_io, use_color);
        atexit([]() { reset_display(); console::cleanup(); });
        console::set_completion_callback(auto_completion_callback);
    }


    static void update_context_usage(int64_t used_tokens) {
        if (used_tokens >= 0) {
            g_ctx_used = used_tokens;
        }
    }

    static void show_hero_banner(
        std::string_view model_name,
        bool is_local,
        int64_t n_ctx,
        std::string_view ftype,
        std::string_view modalities,
        std::string_view build_info,
        bool custom_system
    ) {
        if (!simple_io && use_color) {
            console::log("\n%s\n", betelgeuse_terminal_logo);
            console::log("  \033[1;38;2;170;30;45mBetelgeuse\033[0m \033[38;2;160;140;135m(Alpha-Orionis)\033[0m\033[38;2;100;75;70m · \033[0;38;2;237;170;67mv0.6.0-dev\033[0m\033[38;2;100;75;70m · \033[0;38;2;160;140;135mfrontier-scale MoE · 12GB VRAM\033[0m\n\n");
            console::log("  \033[2;38;2;160;140;135mTip: bel-cli runs directly with -m, or connects to bel-server via --server-base\033[0m\n");
            console::log("  \033[2;38;2;160;140;135m     /clear resets session · /exit quits · --color off for plain text\033[0m\n\n");
            console::log("  \033[1;38;2;75;180;95m✔\033[0m \033[1;38;2;250;240;235mSession initialized\033[0m\n");
            console::log("    \033[38;2;237;170;67mModel\033[0m      : %.*s (%s)\n", int(model_name.size()), model_name.data(), ftype.empty() ? "IQ2_XS" : ftype.data());
            console::log("    \033[38;2;237;170;67mRuntime\033[0m    : %s · %.*s · build %.*s\n", is_local ? "local direct-io (RTX 3060 12GB)" : "remote server", int(modalities.size()), modalities.data(), int(build_info.size()), build_info.data());
            if (n_ctx > 0) {
                console::log("    \033[38;2;237;170;67mContext\033[0m    : %lld tokens · 5000 slots\n", (long long)n_ctx);
            }
            if (custom_system) {
                console::log("    \033[38;2;170;99;48mSystem\033[0m     : custom system prompt active\n");
            }
            console::log("\n");
        } else if (!simple_io) {
            console::log("\n       ##\n     ######\n   ##########\n     ######\n       ##\n\n");
            console::log("Betelgeuse (Alpha-Orionis) | bel-cli\n");
            console::log("Model   : %.*s (%s)\n", int(model_name.size()), model_name.data(), ftype.empty() ? "IQ2_XS" : ftype.data());
            console::log("Runtime : %s (build %.*s)\n", is_local ? "local direct-io" : "remote server", int(build_info.size()), build_info.data());
            console::log("Context : %lld tokens\n\n", (long long)n_ctx);
        } else {
            console::log("Betelgeuse / bel-cli\n");
            console::log("Model   : %.*s\n", int(model_name.size()), model_name.data());
            console::log("Context : %lld tokens\n\n", (long long)n_ctx);
        }
        reset_display();
        console::flush();
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

        static void print_session_rail() {
            if (simple_io || !use_color) {
                console::log("\n> ");
                return;
            }

            int w = get_terminal_width();
            console::log("\n");

            // Segment 1: ★ Betelgeuse
            set_style(color_crimson);
            console::log(" ★ Betelgeuse");
            reset_display();

            set_style(color_muted);
            console::log(" › ");
            reset_display();

            // Segment 2: ◒ Model
            set_style(color_amber);
            console::log("◒ %s", g_model_name.c_str());
            reset_display();

            set_style(color_muted);
            console::log(" › ");
            reset_display();

            // Segment 3: ⚡ Mode
            set_style(color_gold);
            console::log("⚡ %s", g_mode_name.c_str());
            reset_display();

            // Segment 4: Context percentage bar
            double pct = 0.0;
            if (g_ctx_capacity > 0) {
                pct = std::min(100.0, 100.0 * double(g_ctx_used) / double(g_ctx_capacity));
            }
            char pct_buf[64];
            if (g_ctx_capacity > 0) {
                snprintf(pct_buf, sizeof(pct_buf), "─[ %.1f%% · %lld/%lld ]─", pct, (long long)g_ctx_used, (long long)g_ctx_capacity);
            } else {
                snprintf(pct_buf, sizeof(pct_buf), "─[ %lld tokens ]─", (long long)g_ctx_used);
            }

            set_style(color_amber);
            console::log(" ▶");
            reset_display();

            set_style(color_dark);
            console::log("%s", pct_buf);

            // Fill remainder with horizontal line
            int printed_len = 14 + 3 + (int)g_model_name.size() + 3 + 2 + (int)g_mode_name.size() + 2 + (int)strlen(pct_buf);
            int rem = std::max(4, w - printed_len - 2);
            for (int k = 0; k < rem; ++k) {
                console::log("─");
            }
            reset_display();
            console::log("\n");

            // Prompt line: ╰─ 
            set_style(color_amber);
            console::log("╰─ ");
            reset_display();
        }

        void echo(const std::string & buffer) {
            if (simple_io || !use_color) {
                console::log("\n> %s\n", buffer.c_str());
                return;
            }
            print_session_rail();
            console::log("%s\n", buffer.c_str());
        }

        std::string read_input(bool multiline_input, const char * prompt = nullptr) {
            if (prompt) {
                set_style(color_amber);
                console::log("%s", prompt);
                reset_display();
            } else {
                print_session_rail();
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
        bool in_reasoning_box = false;
        bool in_response_box  = false;
        cli_markdown_renderer md;

        assistant_turn() : md(use_color && !simple_io) {}

        ~assistant_turn() {
            close_boxes();
        }

        void close_boxes() {
            if (in_reasoning_box) {
                if (use_color && !simple_io) {
                    console::log("\n\033[38;2;120;60;50m╰────────────────────────────────────────────────────────\033[0m\n");
                } else {
                    console::log("\n+--------------------------------------------------------\n");
                }
                in_reasoning_box = false;
            }
            if (in_response_box) {
                std::string tail = md.process("", /*final_chunk=*/true);
                if (!tail.empty()) {
                    console::log("%s", tail.c_str());
                }
                console::log("\n");
                in_response_box = false;
            }
            reset_display();
            console::flush();
        }

        void push(assistant_display_mode m, const std::string & delta) {
            if (delta.empty()) {
                return;
            }

            if (m == ASSISTANT_DISPLAY_MODE_REASONING) {
                if (!in_reasoning_box) {
                    if (use_color && !simple_io) {
                        console::log("\n\033[38;2;152;30;50m╭── \033[1;38;2;237;170;67m💭 Thinking\033[0;38;2;152;30;50m ────────────────────────────────────────\033[0m\n");
                    } else {
                        console::log("\n+-- [Thinking] ------------------------------------------\n");
                    }
                    in_reasoning_box = true;
                }
                if (use_color && !simple_io) {
                    set_style(color_subtle);
                }
                console::log("%s", delta.c_str());
                if (use_color && !simple_io) {
                    reset_display();
                }
                console::flush();
                return;
            }

            if (m == ASSISTANT_DISPLAY_MODE_CONTENT) {
                if (in_reasoning_box) {
                    if (use_color && !simple_io) {
                        console::log("\n\033[38;2;152;30;50m╰────────────────────────────────────────────────────────\033[0m\n");
                    } else {
                        console::log("\n+--------------------------------------------------------\n");
                    }
                    in_reasoning_box = false;
                }
                if (!in_response_box) {
                    if (use_color && !simple_io) {
                        console::log("\n\033[1;38;2;170;30;45m╭── ★ Betelgeuse\033[0;38;2;152;30;50m ──────────────────────────────────────────\033[0m\n");
                    } else {
                        console::log("\nAssistant:\n");
                    }
                    in_response_box = true;
                }

                std::string formatted = md.process(delta, /*final_chunk=*/false);
                console::log("%s", formatted.c_str());
                console::flush();
            }
        }

        void finish() {
            close_boxes();
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
        if (simple_io || !use_color) {
            console::log("\n[ %.1f tok/s | %.1fs | prompt %.1f tok/s ]\n",
                predicted_per_second, elapsed_seconds, prompt_per_second);
            return;
        }
        int w = get_terminal_width();
        char buf[128];
        snprintf(buf, sizeof(buf), " ⚡ %.1f tok/s · %.1fs · prompt %.1f tok/s ",
                 predicted_per_second, elapsed_seconds, prompt_per_second);

        set_style(color_dark);
        console::log("╰──");
        set_style(color_amber);
        console::log("%s", buf);
        set_style(color_dark);
        int rem = std::max(4, w - 4 - (int)strlen(buf));
        for (int k = 0; k < rem; ++k) {
            console::log("─");
        }
        reset_display();
        console::log("\n");
        console::flush();
    }
}
