#!/usr/bin/env python3
"""
export_bitnet.py — Export trained BitNet model to C for ESP32 / embedded deployment
=====================================================================================

Loads the trained BitNet model from phase3-productize/models/bitnet_unified.pkl
and generates ANSI C source files that implement the same forward pass:

  bitnet_weights.h    — architecture constants + weight/bias/scale arrays
  bitnet_inference.c  — ternary matmul, ReLU, softmax, bitnet_forward()
  bitnet_main.c       — minimal test program (one-hot input → predicted class)
  Makefile            — builds bitnet_test binary via gcc
  README.md           — build/deploy instructions

Usage (from project root):
    python3 phase3-productize/deploy/esp32/export_bitnet.py

Then build and test:
    cd phase3-productize/deploy/esp32
    make
    ./bitnet_test
"""

import os
import sys
import pickle
import math
import importlib
import textwrap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'phase3-productize', 'models', 'bitnet_unified.pkl')
OUT_DIR = SCRIPT_DIR   # write generated files into this directory

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'phase1-from-scratch', 'level-a-abcs'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'phase1-from-scratch', 'level-d-unified'))

print(f"Loading model from: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    artifact = pickle.load(f)

model = artifact['model']
metadata = artifact['metadata']

INPUT_SIZE  = model.input_size
HIDDEN_SIZE = model.hidden_size
OUTPUT_SIZE = model.output_size
NUM_LAYERS  = model.num_layers

print(f"Architecture: [{INPUT_SIZE} → {HIDDEN_SIZE} → {OUTPUT_SIZE}]  num_layers={NUM_LAYERS}")
print(f"Scales:  {model.scales}")

# ---------------------------------------------------------------------------
# Run a Python reference forward pass on a test input (letter index 0 = 'A')
# ---------------------------------------------------------------------------
test_input = [0.0] * INPUT_SIZE
test_input[0] = 1.0   # one-hot: first letter

# Import helpers to call model.forward()
math_fn     = importlib.import_module('01_math_foundations')
neuron_mod  = importlib.import_module('02_single_neuron')

py_logits = model.forward(test_input)
py_pred   = py_logits.index(max(py_logits))

def softmax_py(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

py_probs = softmax_py(py_logits)
print(f"\nPython reference — input: one-hot[0]")
print(f"  Predicted class: {py_pred}")
print(f"  Top-5 probs: {sorted(enumerate(py_probs), key=lambda x: -x[1])[:5]}")

# ---------------------------------------------------------------------------
# Helpers to format arrays in C
# ---------------------------------------------------------------------------

def fmt_int8_array(name, flat_list, cols=16):
    """Format a flat list of int8 values as a static C array."""
    lines = [f"static const int8_t {name}[{len(flat_list)}] = {{"]
    for i in range(0, len(flat_list), cols):
        chunk = flat_list[i:i+cols]
        row = ', '.join(f"{int(v):2d}" for v in chunk)
        lines.append(f"    {row},")
    lines.append("};")
    return '\n'.join(lines)

def fmt_float_array(name, flat_list, cols=8):
    """Format a flat list of floats as a static C array."""
    lines = [f"static const float {name}[{len(flat_list)}] = {{"]
    for i in range(0, len(flat_list), cols):
        chunk = flat_list[i:i+cols]
        row = ', '.join(f"{float(v):.8f}f" for v in chunk)
        lines.append(f"    {row},")
    lines.append("};")
    return '\n'.join(lines)

# ---------------------------------------------------------------------------
# Build flat weight/bias arrays for each layer
# ---------------------------------------------------------------------------
# Layer 0: INPUT_SIZE → HIDDEN_SIZE   shape (HIDDEN_SIZE, INPUT_SIZE)
# Layer 1: HIDDEN_SIZE → OUTPUT_SIZE  shape (OUTPUT_SIZE, HIDDEN_SIZE)

layer_weights = []   # list of flat int8 arrays
layer_biases  = []   # list of flat float arrays
layer_scales  = model.scales   # one float per layer

for layer_idx in range(NUM_LAYERS):
    q_matrix = model.quantized_weights[layer_idx]
    flat_w = [int(w) for row in q_matrix for w in row]
    layer_weights.append(flat_w)

    flat_b = list(model.biases[layer_idx])
    layer_biases.append(flat_b)

# Compute memory footprint
weight_bytes = sum(len(w) for w in layer_weights)   # int8_t = 1 byte each
bias_bytes   = sum(len(b) * 4 for b in layer_biases) # float = 4 bytes
scale_bytes  = NUM_LAYERS * 4
total_bytes  = weight_bytes + bias_bytes + scale_bytes

print(f"\nMemory footprint (int8 weights):")
print(f"  Weights:  {weight_bytes} bytes")
print(f"  Biases:   {bias_bytes} bytes")
print(f"  Scales:   {scale_bytes} bytes")
print(f"  Total:    {total_bytes} bytes  ({total_bytes/1024:.2f} KB)")

# ---------------------------------------------------------------------------
# Generate bitnet_weights.h
# ---------------------------------------------------------------------------
weights_h = f"""\
/*
 * bitnet_weights.h — auto-generated by export_bitnet.py
 *
 * BitNet model exported from Python training.
 * Architecture: [{INPUT_SIZE} → {HIDDEN_SIZE} → {OUTPUT_SIZE}]
 * Ternary weights ({{-1, 0, +1}}) stored as int8_t.
 *
 * DO NOT EDIT — regenerate with export_bitnet.py
 */

#ifndef BITNET_WEIGHTS_H
#define BITNET_WEIGHTS_H

#include <stdint.h>

/* ---- Architecture ---- */
#define INPUT_SIZE    {INPUT_SIZE}
#define HIDDEN_SIZE   {HIDDEN_SIZE}
#define OUTPUT_SIZE   {OUTPUT_SIZE}
#define NUM_LAYERS    {NUM_LAYERS}

/* ---- Memory footprint (bytes) ---- */
#define WEIGHT_BYTES  {weight_bytes}
#define BIAS_BYTES    {bias_bytes}
#define SCALE_BYTES   {scale_bytes}
#define TOTAL_BYTES   {total_bytes}

/* ---- Scale factors (one per layer) ---- */
static const float scales[NUM_LAYERS] = {{
"""

for i, s in enumerate(layer_scales):
    weights_h += f"    {s:.8f}f"
    weights_h += ",\n" if i < NUM_LAYERS - 1 else "\n"
weights_h += "};\n\n"

# Layer 0 weights: shape [HIDDEN_SIZE][INPUT_SIZE] — stored row-major
weights_h += "/* ---- Layer 0: ternary weights [HIDDEN_SIZE x INPUT_SIZE] ---- */\n"
weights_h += fmt_int8_array("layer0_weights", layer_weights[0]) + "\n\n"
weights_h += "/* ---- Layer 0: biases [HIDDEN_SIZE] ---- */\n"
weights_h += fmt_float_array("layer0_biases", layer_biases[0]) + "\n\n"

# Layer 1 weights: shape [OUTPUT_SIZE][HIDDEN_SIZE] — stored row-major
weights_h += "/* ---- Layer 1: ternary weights [OUTPUT_SIZE x HIDDEN_SIZE] ---- */\n"
weights_h += fmt_int8_array("layer1_weights", layer_weights[1]) + "\n\n"
weights_h += "/* ---- Layer 1: biases [OUTPUT_SIZE] ---- */\n"
weights_h += fmt_float_array("layer1_biases", layer_biases[1]) + "\n\n"

weights_h += "#endif /* BITNET_WEIGHTS_H */\n"

# ---------------------------------------------------------------------------
# Generate bitnet_inference.c
# ---------------------------------------------------------------------------
inference_c = f"""\
/*
 * bitnet_inference.c — auto-generated by export_bitnet.py
 *
 * Ternary BitNet forward pass in ANSI C.
 * No dynamic allocation; works on bare-metal microcontrollers (ESP32, etc.).
 *
 * Compile with:
 *   gcc -Wall -O2 -std=c99 -c bitnet_inference.c -lm
 */

#include "bitnet_inference.h"
#include "bitnet_weights.h"
#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * ternary_matmul_add_bias
 *   Computes:  out[i] = scale * sum(weights[i*in_size + j] * in[j]) + bias[i]
 *   weights values are in {{-1, 0, +1}}, so multiply is add/subtract.
 * --------------------------------------------------------------------------- */
static void ternary_matmul_add_bias(
    const int8_t * restrict weights,  /* [out_size * in_size] row-major */
    float                   scale,
    const float  * restrict bias,     /* [out_size] */
    const float  * restrict in,       /* [in_size]  */
    float        * restrict out,      /* [out_size] */
    int in_size,
    int out_size)
{{
    for (int i = 0; i < out_size; ++i) {{
        float acc = 0.0f;
        const int8_t *row = weights + (size_t)i * in_size;
        for (int j = 0; j < in_size; ++j) {{
            int8_t w = row[j];
            if (w == 1)       acc += in[j];
            else if (w == -1) acc -= in[j];
            /* w == 0: skip */
        }}
        out[i] = acc * scale + bias[i];
    }}
}}

/* ---------------------------------------------------------------------------
 * relu_inplace — apply ReLU in-place: x = max(0, x)
 * --------------------------------------------------------------------------- */
static void relu_inplace(float *x, int n)
{{
    for (int i = 0; i < n; ++i)
        if (x[i] < 0.0f) x[i] = 0.0f;
}}

/* ---------------------------------------------------------------------------
 * softmax_inplace — numerically stable softmax in-place
 * --------------------------------------------------------------------------- */
void softmax_inplace(float *x, int n)
{{
    /* find max for numerical stability */
    float mx = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] > mx) mx = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {{
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }}
    for (int i = 0; i < n; ++i)
        x[i] /= sum;
}}

/* ---------------------------------------------------------------------------
 * bitnet_forward
 *
 *   Runs the full [{INPUT_SIZE} → {HIDDEN_SIZE} → {OUTPUT_SIZE}] BitNet forward pass.
 *
 *   input  : float array of length INPUT_SIZE  (caller provides)
 *   output : float array of length OUTPUT_SIZE (caller provides, filled on return)
 *
 *   After return, output[i] holds the probability for class i.
 *   argmax(output) gives the predicted class index.
 * --------------------------------------------------------------------------- */
void bitnet_forward(const float *input, float *output)
{{
    /* Hidden layer activation buffer */
    float hidden[HIDDEN_SIZE];

    /* Layer 0: input → hidden */
    ternary_matmul_add_bias(
        layer0_weights, scales[0], layer0_biases,
        input, hidden,
        INPUT_SIZE, HIDDEN_SIZE);
    relu_inplace(hidden, HIDDEN_SIZE);

    /* Layer 1: hidden → output logits */
    ternary_matmul_add_bias(
        layer1_weights, scales[1], layer1_biases,
        hidden, output,
        HIDDEN_SIZE, OUTPUT_SIZE);

    /* Apply softmax to get probabilities */
    softmax_inplace(output, OUTPUT_SIZE);
}}

/* ---------------------------------------------------------------------------
 * bitnet_argmax — returns index of maximum value in array of length n
 * --------------------------------------------------------------------------- */
int bitnet_argmax(const float *x, int n)
{{
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    return best;
}}
"""

# ---------------------------------------------------------------------------
# Generate bitnet_inference.h
# ---------------------------------------------------------------------------
inference_h = f"""\
/*
 * bitnet_inference.h — BitNet inference API
 * auto-generated by export_bitnet.py
 */

#ifndef BITNET_INFERENCE_H
#define BITNET_INFERENCE_H

#ifdef __cplusplus
extern "C" {{
#endif

/**
 * bitnet_forward — run the full [{INPUT_SIZE}→{HIDDEN_SIZE}→{OUTPUT_SIZE}] BitNet forward pass.
 *
 * @param input   float array of length INPUT_SIZE
 * @param output  float array of length OUTPUT_SIZE (filled with probabilities)
 */
void bitnet_forward(const float *input, float *output);

/**
 * softmax_inplace — apply softmax to array x of length n, in-place.
 */
void softmax_inplace(float *x, int n);

/**
 * bitnet_argmax — return index of maximum value in array x of length n.
 */
int bitnet_argmax(const float *x, int n);

#ifdef __cplusplus
}}
#endif

#endif /* BITNET_INFERENCE_H */
"""

# ---------------------------------------------------------------------------
# Generate bitnet_main.c  — test program
# ---------------------------------------------------------------------------

# Build a formatted one-hot input initializer
onehot_vals = ', '.join('1.0f' if i == 0 else '0.0f' for i in range(INPUT_SIZE))

# Provide the Python reference prediction so the tester can see agreement
py_top5 = sorted(enumerate(py_probs), key=lambda x: -x[1])[:5]
py_comment = "Python reference top-5: " + ", ".join(f"class {i}={p:.4f}" for i, p in py_top5)

main_c = f"""\
/*
 * bitnet_main.c — minimal test program for the exported BitNet model
 * auto-generated by export_bitnet.py
 *
 * Build:  make
 * Run:    ./bitnet_test
 *
 * Expected behaviour: matches the Python model's prediction for the same input.
 * {py_comment}
 */

#include <stdio.h>
#include <string.h>
#include "bitnet_inference.h"
#include "bitnet_weights.h"

int main(void)
{{
    printf("=== BitNet ESP32 Inference Test ===\\n");
    printf("Architecture: [%d -> %d -> %d]  num_layers=%d\\n",
           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS);
    printf("Total ROM bytes: %d  (%.2f KB)\\n\\n",
           TOTAL_BYTES, (float)TOTAL_BYTES / 1024.0f);

    /* ---- Test input: one-hot encoding of class 0 (first letter / feature) ---- */
    float input[INPUT_SIZE] = {{ {onehot_vals} }};
    float output[OUTPUT_SIZE];

    printf("Input: one-hot[0] (first feature active, all others zero)\\n");

    /* ---- Run inference ---- */
    bitnet_forward(input, output);

    /* ---- Report results ---- */
    int predicted = bitnet_argmax(output, OUTPUT_SIZE);
    printf("Predicted class: %d\\n", predicted);
    printf("\\nOutput probabilities:\\n");
    for (int i = 0; i < OUTPUT_SIZE; ++i) {{
        printf("  class %2d: %.6f%s\\n",
               i, output[i], (i == predicted) ? "  <-- predicted" : "");
    }}

    /* ---- Python reference (embed for easy comparison) ---- */
    printf("\\nPython reference prediction: {py_pred}\\n");
    printf("%s: %s\\n",
           (predicted == {py_pred}) ? "PASS" : "FAIL",
           (predicted == {py_pred})
               ? "C and Python models agree!"
               : "MISMATCH — C and Python predictions differ");

    return (predicted == {py_pred}) ? 0 : 1;
}}
"""

# ---------------------------------------------------------------------------
# Generate Makefile
# ---------------------------------------------------------------------------
makefile = """\
# Makefile — build BitNet test binary
# Usage: make && ./bitnet_test

CC      = gcc
CFLAGS  = -Wall -O2 -std=c99
LDFLAGS = -lm

SRCS = bitnet_inference.c bitnet_main.c
OBJS = $(SRCS:.c=.o)
BIN  = bitnet_test

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJS)
\t$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
\t$(CC) $(CFLAGS) -c -o $@ $<

bitnet_inference.o: bitnet_inference.c bitnet_inference.h bitnet_weights.h
bitnet_main.o:      bitnet_main.c      bitnet_inference.h bitnet_weights.h

clean:
\trm -f $(OBJS) $(BIN)
"""

# ---------------------------------------------------------------------------
# Generate README.md
# ---------------------------------------------------------------------------
readme_md = f"""\
# BitNet Model — Exported to C for ESP32

This directory contains the BitNet phonics model exported from Python training
to ANSI C. The same model that runs in Python now runs as a ~{total_bytes}-byte
firmware image — no Python, no NumPy, no dynamic allocation required.

## Architecture

```
Input [{INPUT_SIZE}] → Hidden [{HIDDEN_SIZE}] → Output [{OUTPUT_SIZE}]
Ternary weights {{-1, 0, +1}}   ·   {NUM_LAYERS} layers
```

## Files

| File | Description |
|------|-------------|
| `bitnet_weights.h` | All weights, biases, scales as C arrays (`int8_t` / `float`) |
| `bitnet_inference.h` | Public API header |
| `bitnet_inference.c` | Ternary matmul, ReLU, softmax, `bitnet_forward()` |
| `bitnet_main.c` | Minimal test program — hardcoded one-hot input, prints predicted class |
| `Makefile` | Build the test binary via `gcc` |
| `export_bitnet.py` | Python script that regenerates all the above from the `.pkl` model |

## Memory Footprint

| Component | Size |
|-----------|------|
| Ternary weights (int8_t) | {weight_bytes} bytes |
| Biases (float) | {bias_bytes} bytes |
| Scale factors (float) | {scale_bytes} bytes |
| **Total** | **{total_bytes} bytes ({total_bytes/1024:.2f} KB)** |

This fits comfortably in the ESP32's 400 KB DRAM or 4 MB flash.

## Build and Run (macOS / Linux)

```bash
make
./bitnet_test
```

Expected output — the C model should predict the same class as the Python model
for the same one-hot input.

## Deploy to ESP32 (ESP-IDF)

1. Copy `bitnet_weights.h`, `bitnet_inference.h`, `bitnet_inference.c` into your
   ESP-IDF component directory (e.g. `components/bitnet/`).
2. Add a `CMakeLists.txt`:
   ```cmake
   idf_component_register(
       SRCS "bitnet_inference.c"
       INCLUDE_DIRS ".")
   ```
3. In your `main.c`, include `bitnet_inference.h` and call `bitnet_forward()`.
4. Build with `idf.py build` and flash with `idf.py flash`.

The model uses no heap allocation — all buffers are stack-local inside
`bitnet_forward()` — making it safe for FreeRTOS tasks with small stacks.

## Regenerate

```bash
cd <project-root>
python3 phase3-productize/deploy/esp32/export_bitnet.py
```
"""

# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------
def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    size = os.path.getsize(path)
    print(f"  Wrote: {os.path.basename(path)}  ({size} bytes)")

print("\nGenerating C files in:", OUT_DIR)
write_file(os.path.join(OUT_DIR, 'bitnet_weights.h'),    weights_h)
write_file(os.path.join(OUT_DIR, 'bitnet_inference.h'),  inference_h)
write_file(os.path.join(OUT_DIR, 'bitnet_inference.c'),  inference_c)
write_file(os.path.join(OUT_DIR, 'bitnet_main.c'),       main_c)
write_file(os.path.join(OUT_DIR, 'Makefile'),            makefile)
write_file(os.path.join(OUT_DIR, 'README.md'),           readme_md)

print(f"\nDone. Python reference prediction for one-hot[0]: class {py_pred}")
print(f"Memory footprint: {total_bytes} bytes ({total_bytes/1024:.2f} KB)")
