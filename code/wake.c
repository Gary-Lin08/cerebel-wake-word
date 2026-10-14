/* "Hey Cerebel" detector: features and network exactly as cerebel_wake/train.py has them. */
#include "wake.h"

#include <math.h>
#include <string.h>

#include "wake_model.h"

#if WAKE_MELS != WAKE_BANDS || WAKE_FRAMES != WAKE_SPAN
#error "wake_model.h was trained for a different input"
#endif

#define BINS (WAKE_NFFT / 2 + 1)
#define PI 3.14159265358979f
#define T1 ((WAKE_SPAN - 1) / 2 + 1) /* after the first convolution (kernel 3, stride 2, padding 1) */

static float window[WAKE_WIN];
static float cos_table[WAKE_NFFT / 2];
static float sin_table[WAKE_NFFT / 2];
static int band_first[WAKE_BANDS];
static int band_count[WAKE_BANDS];
static float band_weight[WAKE_BANDS][BINS];
static int ready;

static float mel_of(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float hz_of(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

static void prepare(void) {
    float points[WAKE_BANDS + 2];
    int i;
    int m;

    for (i = 0; i < WAKE_WIN; i++) {
        window[i] = 0.5f - 0.5f * cosf(2.0f * PI * (float)i / WAKE_WIN);
    }
    for (i = 0; i < WAKE_NFFT / 2; i++) {
        cos_table[i] = cosf(2.0f * PI * (float)i / WAKE_NFFT);
        sin_table[i] = -sinf(2.0f * PI * (float)i / WAKE_NFFT);
    }
    for (i = 0; i < WAKE_BANDS + 2; i++) {
        float low = mel_of(60.0f);
        float high = mel_of(7600.0f);
        points[i] = hz_of(low + (high - low) * (float)i / (WAKE_BANDS + 1)) * WAKE_NFFT / WAKE_RATE;
    }
    for (m = 0; m < WAKE_BANDS; m++) {
        float left = points[m];
        float centre = points[m + 1];
        float right = points[m + 2];
        int k;
        band_first[m] = -1;
        band_count[m] = 0;
        for (k = 0; k < BINS; k++) {
            float weight = 0.0f;
            if ((float)k > left && (float)k <= centre) {
                weight = ((float)k - left) / (centre - left);
            } else if ((float)k > centre && (float)k < right) {
                weight = (right - (float)k) / (right - centre);
            }
            if (weight > 0.0f) {
                if (band_first[m] < 0) {
                    band_first[m] = k;
                }
                band_weight[m][band_count[m]++] = weight;
            }
        }
        if (band_first[m] < 0) {
            band_first[m] = 0;
        }
    }
    ready = 1;
}

/* In-place radix-2 FFT of WAKE_NFFT complex values. */
static void fft(float *re, float *im) {
    int i;
    int j = 0;
    int size;

    for (i = 1; i < WAKE_NFFT; i++) {
        int bit = WAKE_NFFT >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (size = 2; size <= WAKE_NFFT; size <<= 1) {
        int half = size >> 1;
        int step = WAKE_NFFT / size;
        int start;
        for (start = 0; start < WAKE_NFFT; start += size) {
            int k;
            for (k = 0; k < half; k++) {
                float wr = cos_table[k * step];
                float wi = sin_table[k * step];
                int a = start + k;
                int b = a + half;
                float tr = re[b] * wr - im[b] * wi;
                float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
    }
}

static void add_frame(wake_t *wake) {
    static float re[WAKE_NFFT];
    static float im[WAKE_NFFT];
    float power[BINS];
    float *out;
    int i;
    int m;

    for (i = 0; i < WAKE_NFFT; i++) {
        /* history is a ring: the oldest sample is the one about to be overwritten */
        re[i] = i < WAKE_WIN ? (float)wake->history[(wake->write + i) % WAKE_WIN] / 32768.0f * window[i] : 0.0f;
        im[i] = 0.0f;
    }
    fft(re, im);
    for (i = 0; i < BINS; i++) {
        power[i] = re[i] * re[i] + im[i] * im[i];
    }
    wake->newest = (wake->newest + 1) % WAKE_SPAN;
    out = wake->bands[wake->newest];
    for (m = 0; m < WAKE_BANDS; m++) {
        float sum = 0.0f;
        int k;
        for (k = 0; k < band_count[m]; k++) {
            sum += band_weight[m][k] * power[band_first[m] + k];
        }
        out[m] = logf(sum + 1e-6f);
    }
    if (wake->frames < WAKE_SPAN) {
        wake->frames++;
    }
}

/* One depthwise-separable block with a skip: x (WAKE_CH x length) -> out (WAKE_CH x new length). */
static int block(const float *x, int length, int stride, const float *dw, const float *db, const float *pw,
                 const float *pb, float *out) {
    static float depth[WAKE_CH * T1];
    int out_length = (length - 1) / stride + 1;
    int c;
    int t;

    for (c = 0; c < WAKE_CH; c++) {
        for (t = 0; t < out_length; t++) {
            float sum = db[c];
            int k;
            for (k = 0; k < 9; k++) {
                int at = t * stride + k - 4;
                if (at >= 0 && at < length) {
                    sum += dw[c * 9 + k] * x[c * length + at];
                }
            }
            depth[c * out_length + t] = sum > 0.0f ? sum : 0.0f;
        }
    }
    for (c = 0; c < WAKE_CH; c++) {
        for (t = 0; t < out_length; t++) {
            float sum = pb[c];
            int k;
            for (k = 0; k < WAKE_CH; k++) {
                sum += pw[c * WAKE_CH + k] * depth[k * out_length + t];
            }
            sum += x[c * length + t * stride];
            out[c * out_length + t] = sum > 0.0f ? sum : 0.0f;
        }
    }
    return out_length;
}

float wake_score(const wake_t *wake) {
    static float input[WAKE_BANDS * WAKE_SPAN];
    static float a[WAKE_CH * T1];
    static float b[WAKE_CH * T1];
    const float *dws[4] = {WAKE_B0_DW, WAKE_B1_DW, WAKE_B2_DW, WAKE_B3_DW};
    const float *dbs[4] = {WAKE_B0_DB, WAKE_B1_DB, WAKE_B2_DB, WAKE_B3_DB};
    const float *pws[4] = {WAKE_B0_PW, WAKE_B1_PW, WAKE_B2_PW, WAKE_B3_PW};
    const float *pbs[4] = {WAKE_B0_PB, WAKE_B1_PB, WAKE_B2_PB, WAKE_B3_PB};
    float *x = a;
    float *y = b;
    float sum;
    int length;
    int i;
    int c;
    int t;

    /* Oldest frame first, normalised as in training. */
    for (t = 0; t < WAKE_SPAN; t++) {
        const float *frame = wake->bands[(wake->newest + 1 + t) % WAKE_SPAN];
        for (c = 0; c < WAKE_BANDS; c++) {
            input[c * WAKE_SPAN + t] = (frame[c] - WAKE_MEAN[c]) / WAKE_STD[c];
        }
    }
    /* First convolution: kernel 3, stride 2, padding 1. */
    for (c = 0; c < WAKE_CH; c++) {
        for (t = 0; t < T1; t++) {
            float s = WAKE_FIRST_B[c];
            int k;
            for (i = 0; i < WAKE_BANDS; i++) {
                for (k = 0; k < 3; k++) {
                    int at = t * 2 + k - 1;
                    if (at >= 0 && at < WAKE_SPAN) {
                        s += WAKE_FIRST_W[(c * WAKE_BANDS + i) * 3 + k] * input[i * WAKE_SPAN + at];
                    }
                }
            }
            x[c * T1 + t] = s > 0.0f ? s : 0.0f;
        }
    }
    length = T1;
    for (i = 0; i < 4; i++) {
        float *swap;
        length = block(x, length, WAKE_STRIDES[i], dws[i], dbs[i], pws[i], pbs[i], y);
        swap = x; x = y; y = swap;
    }
    sum = WAKE_OUT_B[0];
    for (c = 0; c < WAKE_CH; c++) {
        float mean = 0.0f;
        for (t = 0; t < length; t++) {
            mean += x[c * length + t];
        }
        sum += WAKE_OUT_W[c] * mean / (float)length;
    }
    return 1.0f / (1.0f + expf(-sum));
}

void wake_init(wake_t *wake) {
    if (!ready) {
        prepare();
    }
    memset(wake, 0, sizeof(*wake));
    wake->newest = WAKE_SPAN - 1;
}

float wake_feed(wake_t *wake, const int16_t *samples, int count) {
    float best = -1.0f;
    int i;

    for (i = 0; i < count; i++) {
        wake->history[wake->write] = samples[i];
        wake->write = (wake->write + 1) % WAKE_WIN;
        wake->total++;
        /* Frames end at sample 400, 560, 720 ... as numpy's unfold has them. */
        if (wake->total < WAKE_WIN || (wake->total - WAKE_WIN) % WAKE_HOP != 0) {
            continue;
        }
        {
            /* The first look comes WAKE_EVERY frames after the span is full, as in training. */
            int was_full = wake->frames == WAKE_SPAN;
            add_frame(wake);
            if (!was_full || ++wake->since_look < WAKE_EVERY) {
                continue;
            }
        }
        {
            float score = wake_score(wake);
            wake->since_look = 0;
            if (score > best) {
                best = score;
            }
        }
    }
    return best;
}
