#include "../src/llama-mmap.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

int main() {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 1;
    }
    const size_t page = static_cast<size_t>(page_size);
    char path[] = "/tmp/llama-mmap-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        return 1;
    }
    std::vector<unsigned char> content(3 * page, 0x41);
    const bool written = write(fd, content.data(), content.size()) == static_cast<ssize_t>(content.size());
    close(fd);
    if (!written) {
        unlink(path);
        return 1;
    }
    bool preserved = false;
    {
        llama_file file(path, "rb");
        unlink(path);
        llama_mmap mapping(&file, 0);
        auto * base = static_cast<unsigned char *>(mapping.addr());
        mapping.unmap_fragment(page, 2 * page);
        void * reused = mmap(base + page, page, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        if (reused == MAP_FAILED) {
            return 1;
        }
        std::memset(reused, 0x5a, page);
        mapping.advise_random();
        mapping.drop_cache();
        preserved = base[0] == 0x41 && base[2 * page] == 0x41 &&
            std::all_of(base + page, base + 2 * page, [](unsigned char value) { return value == 0x5a; });
        munmap(reused, page);
    }
    if (!preserved) {
        std::fprintf(stderr, "cache discard cleared an anonymous mapping in a released file fragment\n");
        return 1;
    }
    return 0;
}
