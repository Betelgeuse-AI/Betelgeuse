#undef NDEBUG
#include "../tools/cli/cli-markdown.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    // 1. Basic formatting test
    {
        cli_markdown_renderer md(/*use_color=*/false);
        std::string input = "# Heading\nHere is a list:\n- item 1\n- item 2\n\n```python\nprint(42)\n```\n";
        std::string output = md.process(input, /*final_chunk=*/true);
        assert(output.find("Heading") != std::string::npos);
        assert(output.find("python") != std::string::npos);
        assert(output.find("print(42)") != std::string::npos);
    }

    // 2. Chunk-split delimiter test
    {
        cli_markdown_renderer md(/*use_color=*/true);
        std::string out1 = md.process("This is *");
        std::string out2 = md.process("*bold** text");
        std::string out3 = md.process("", /*final_chunk=*/true);
        std::string combined = out1 + out2 + out3;
        assert(combined.find("bold") != std::string::npos);
    }

    // 3. Fenced code block across chunks
    {
        cli_markdown_renderer md(/*use_color=*/true);
        std::string out1 = md.process("```cpp\n");
        std::string out2 = md.process("int x = 1;\n");
        std::string out3 = md.process("```\n", /*final_chunk=*/true);
        std::string combined = out1 + out2 + out3;
        assert(combined.find("cpp") != std::string::npos);
        assert(combined.find("int x = 1;") != std::string::npos);
    }

    // 4. Color toggle off
    {
        cli_markdown_renderer md(/*use_color=*/false);
        std::string out = md.process("**bold text**", /*final_chunk=*/true);
        assert(out.find("\033[") == std::string::npos);
        assert(out.find("bold text") != std::string::npos);
    }

    std::cout << "test-cli-markdown: all assertions passed\n";
    return 0;
}
