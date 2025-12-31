#include <stdint.h>

void get_sample(uint32_t* m, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        m[i] = i;
    }
}