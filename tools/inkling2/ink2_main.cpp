#include "ink2_embedded.h"

#include <cstdio>
#include <exception>

int main(int argc, char ** argv) {
    try {
        return ink2::engine_main(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "inkling2 engine: %s\n", e.what());
        return 1;
    }
}
