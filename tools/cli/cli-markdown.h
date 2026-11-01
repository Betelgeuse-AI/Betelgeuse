#pragma once

#include <string>
#include <string_view>

class cli_markdown_renderer {
public:
    explicit cli_markdown_renderer(bool use_color = true) : use_color_(use_color) {}

    void reset() {
        in_code_block_ = false;
        code_lang_.clear();
        in_inline_code_ = false;
        in_bold_ = false;
        in_italic_ = false;
        line_start_ = true;
        in_heading_ = 0;
        in_quote_ = false;
        pending_buf_.clear();
    }

    void set_color(bool enabled) {
        use_color_ = enabled;
    }

    // Process a chunk of incoming text from the assistant.
    // Returns formatted terminal text.
    std::string process(std::string_view delta, bool final_chunk = false) {
        std::string out;
        std::string text = pending_buf_;
        text.append(delta.data(), delta.size());
        pending_buf_.clear();

        size_t i = 0;
        const size_t n = text.size();

        while (i < n) {
            // If near the end of delta and not final, buffer potential multi-char tokens
            if (!final_chunk && (n - i) < 4) {
                // If this tail could be a delimiter start, keep in pending_buf_
                std::string_view tail(text.data() + i, n - i);
                if (line_start_ && (tail == "`" || tail == "``" || tail == "#" || tail == "##" || tail == "###" || tail == ">" || tail == "-" || tail == "*")) {
                    pending_buf_ = std::string(tail);
                    break;
                }
                if (!in_code_block_ && (tail == "*" || tail == "~" || tail == "`")) {
                    pending_buf_ = std::string(tail);
                    break;
                }
            }

            char ch = text[i];

            // 1. Fenced code block detection (at line start or anywhere)
            if (i + 2 < n && text[i] == '`' && text[i+1] == '`' && text[i+2] == '`') {
                if (!in_code_block_) {
                    in_code_block_ = true;
                    i += 3;
                    std::string lang;
                    while (i < n && text[i] != '\n') {
                        if (text[i] != '\r') {
                            lang.push_back(text[i]);
                        }
                        i++;
                    }
                    code_lang_ = lang;
                    while (!code_lang_.empty() && (code_lang_.front() == ' ' || code_lang_.front() == '\t')) {
                        code_lang_.erase(code_lang_.begin());
                    }
                    while (!code_lang_.empty() && (code_lang_.back() == ' ' || code_lang_.back() == '\t')) {
                        code_lang_.pop_back();
                    }

                    if (use_color_) {
                        out += "\n\033[38;2;152;30;50m╭── \033[1;38;2;237;170;67m";
                        out += (code_lang_.empty() ? "code" : code_lang_);
                        out += "\033[0;38;2;152;30;50m ──────────────────────────────────────────\033[0m\n";
                    } else {
                        out += "\n+-- [";
                        out += (code_lang_.empty() ? "code" : code_lang_);
                        out += "] ----------------------------------------\n";
                    }
                    if (i < n && text[i] == '\n') {
                        i++;
                    }
                    line_start_ = true;
                    continue;
                }

                in_code_block_ = false;
                code_lang_.clear();
                if (use_color_) {
                    out += "\033[38;2;152;30;50m╰─────────────────────────────────────────────\033[0m\n";
                } else {
                    out += "+---------------------------------------------\n";
                }
                i += 3;
                while (i < n && text[i] != '\n') {
                    i++;
                }
                if (i < n && text[i] == '\n') {
                    i++;
                }
                line_start_ = true;
                continue;
            }

            // Inside code block: verbatim text with bordered left rail
            if (in_code_block_) {
                if (line_start_) {
                    if (use_color_) {
                        out += "\033[38;2;152;30;50m│ \033[0;38;2;245;235;220m";
                    } else {
                        out += "| ";
                    }
                    line_start_ = false;
                }
                out.push_back(ch);
                if (ch == '\n') {
                    if (use_color_) {
                        out += "\033[0m";
                    }
                    line_start_ = true;
                }
                i++;
                continue;
            }

            // Outside code block: line-level Markdown at line start
            if (line_start_) {
                if (i + 3 < n && text.substr(i, 4) == "### ") {
                    in_heading_ = 3;
                    line_start_ = false;
                    out += use_color_ ? "\n\033[1;38;2;200;130;50m◆ " : "\n◆ ";
                    i += 4;
                    continue;
                }
                if (i + 2 < n && text.substr(i, 3) == "## ") {
                    in_heading_ = 2;
                    line_start_ = false;
                    out += use_color_ ? "\n\033[1;38;2;225;155;60m◇ " : "\n◇ ";
                    i += 3;
                    continue;
                }
                if (i + 1 < n && text.substr(i, 2) == "# ") {
                    in_heading_ = 1;
                    line_start_ = false;
                    out += use_color_ ? "\n\033[1;38;2;237;170;67m◈ " : "\n◈ ";
                    i += 2;
                    continue;
                }
                if (i + 1 < n && text.substr(i, 2) == "> ") {
                    in_quote_ = true;
                    line_start_ = false;
                    out += use_color_ ? "\033[38;2;170;99;48m│ \033[3m" : "| ";
                    i += 2;
                    continue;
                }
                if (i + 1 < n && (text.substr(i, 2) == "- " || text.substr(i, 2) == "* ")) {
                    line_start_ = false;
                    out += use_color_ ? "  \033[38;2;237;170;67m•\033[0m " : "  * ";
                    i += 2;
                    continue;
                }
                if (i + 3 < n && (text.substr(i, 4) == "---\n" || text.substr(i, 4) == "***\n")) {
                    out += use_color_ ? "\n\033[38;2;120;60;50m────────────────────────────────────────────────────────\033[0m\n" : "\n--------------------------------------------------------\n";
                    i += 4;
                    line_start_ = true;
                    continue;
                }
            }

            // Inline formatting
            if (ch == '\n') {
                if (in_heading_ > 0) {
                    if (use_color_) {
                        out += "\033[0m";
                    }
                    in_heading_ = 0;
                }
                if (in_quote_) {
                    if (use_color_) {
                        out += "\033[0m";
                    }
                    in_quote_ = false;
                }
                out.push_back('\n');
                line_start_ = true;
                i++;
                continue;
            }

            // Single backtick for inline code
            if (ch == '`') {
                in_inline_code_ = !in_inline_code_;
                if (use_color_) {
                    out += in_inline_code_ ? "\033[1;38;2;237;170;67m\033[48;2;45;30;25m " : " \033[0m";
                } else {
                    out.push_back('`');
                }
                line_start_ = false;
                i++;
                continue;
            }

            if (!in_inline_code_) {
                // Bold **
                if (i + 1 < n && text[i] == '*' && text[i+1] == '*') {
                    in_bold_ = !in_bold_;
                    if (use_color_) {
                        out += in_bold_ ? "\033[1;38;2;255;245;235m" : "\033[22m\033[0m";
                    } else {
                        out += "**";
                    }
                    line_start_ = false;
                    i += 2;
                    continue;
                }

                // Table column separator |
                if (ch == '|') {
                    out += use_color_ ? "\033[38;2;152;30;50m│\033[0m" : "|";
                    line_start_ = false;
                    i++;
                    continue;
                }
            }

            out.push_back(ch);
            line_start_ = false;
            i++;
        }

        if (final_chunk) {
            if (!pending_buf_.empty()) {
                out.append(pending_buf_);
                pending_buf_.clear();
            }
            if (in_code_block_) {
                out += use_color_ ? "\n\033[38;2;152;30;50m╰─────────────────────────────────────────────\033[0m\n" : "\n+---------------------------------------------\n";
                in_code_block_ = false;
            }
            if (use_color_) {
                out += "\033[0m";
            }
            reset();
        }

        return out;
    }

private:
    bool use_color_ = true;
    bool in_code_block_ = false;
    std::string code_lang_;
    bool in_inline_code_ = false;
    bool in_bold_ = false;
    bool in_italic_ = false;
    bool line_start_ = true;
    int in_heading_ = 0;
    bool in_quote_ = false;
    std::string pending_buf_;
};
