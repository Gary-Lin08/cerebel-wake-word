/*
 * "Hey Cerebel", heard on the glasses themselves.
 *
 * A small network (cerebel_wake/train.py) looks at the last 1.5 s of sound as 40
 * log-mel bands every 50 ms. No library, no NPU: about a million multiply-adds a
 * look, a few percent of the glasses' one core. Nothing leaves the glasses until
 * it fires.
 */
#ifndef CEREBEL_WAKE_H
#define CEREBEL_WAKE_H

#include <stdint.h>

#define WAKE_RATE 16000
#define WAKE_WIN 400   /* 25 ms */
#define WAKE_HOP 160   /* 10 ms */
#define WAKE_NFFT 512
#define WAKE_BANDS 40
#define WAKE_SPAN 150  /* frames looked at: 1.5 s */
#define WAKE_EVERY 5   /* frames between looks: 50 ms */

typedef struct {
    int16_t history[WAKE_WIN]; /* the newest WAKE_WIN samples, a ring */
    int write;                 /* where the next sample goes in history */
    uint32_t total;            /* samples so far */
    float bands[WAKE_SPAN][WAKE_BANDS]; /* log-mel, a ring */
    int newest;                /* index of the newest frame in bands */
    int frames;                /* frames so far (until the ring is full) */
    int since_look;
} wake_t;

void wake_init(wake_t *wake);

/* Feed sound; returns the highest score (0..1) of the looks taken, or -1 if none was. */
float wake_feed(wake_t *wake, const int16_t *samples, int count);

/* The score for exactly the last 1.5 s held (for tests). */
float wake_score(const wake_t *wake);

#endif
