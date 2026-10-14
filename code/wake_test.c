/* Feed a 16 kHz mono 16-bit WAV through wake.c and print the score of every look. */
#include <stdio.h>
#include <stdint.h>

#include "wake.h"

int main(int argc, char **argv) {
    static wake_t wake;
    int16_t chunk[WAKE_HOP];
    size_t got;
    FILE *file;

    if (argc != 2 || (file = fopen(argv[1], "rb")) == NULL) {
        fprintf(stderr, "usage: wake_test FILE.wav\n");
        return 2;
    }
    fseek(file, 44, SEEK_SET);
    wake_init(&wake);
    while ((got = fread(chunk, sizeof(int16_t), WAKE_HOP, file)) > 0) {
        float score = wake_feed(&wake, chunk, (int)got);
        if (score >= 0.0f) {
            printf("%.6f\n", score);
        }
    }
    fclose(file);
    return 0;
}
