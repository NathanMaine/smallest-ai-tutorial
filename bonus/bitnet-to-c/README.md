# Bonus: Exporting BitNet to C

**Take your trained Python BitNet model and run it in 4.5 KB of C.**

---

## What This Is

After training a BitNet model in Python, you can export it to C code with no external dependencies. The C version:
- Has no dynamic allocation (all buffers are stack-local)
- Fits in ~4,508 bytes of firmware
- Runs on ESP32-class microcontrollers or any ANSI C environment
- Makes the same predictions as the Python model

This is the end of the "build from scratch" pipeline: Python prototype → exported weights → C inference engine → runs anywhere.

---

## Architecture

```
Input [27] → Hidden [64] → Output [37]
Ternary weights {-1, 0, +1}  ·  2 layers
```

## Files

| File | Description |
|------|-------------|
| `export_bitnet.py` | Python script: loads trained model, generates C header |
| `bitnet_inference.h` | Public API header |
| `bitnet_inference.c` | Ternary matmul, ReLU, softmax, `bitnet_forward()` |
| `bitnet_main.c` | Minimal test program — one-hot input, prints predicted class |
| `Makefile` | Build test binary via `gcc` |

## Memory Footprint

| Component | Size |
|-----------|------|
| Ternary weights (int8_t) | 4,096 bytes |
| Biases (float) | 404 bytes |
| Scale factors (float) | 8 bytes |
| **Total** | **4,508 bytes (4.40 KB)** |

---

## How the Export Works

The `export_bitnet.py` script:

1. Loads the trained BitNet model from `tutorials/04-comparison-study/solution/`
2. Extracts ternary weight matrices, biases, and scale factors
3. Generates a C header file (`bitnet_weights.h`) with all weights as `int8_t` arrays
4. The inference code in `bitnet_inference.c` uses `bitnet_forward()` which:
   - Multiplies by ternary weights (add/subtract, no actual multiply)
   - Scales the result by the learned magnitude factor
   - Applies ReLU for hidden layers, softmax for the output

---

## Build and Run on macOS/Linux

```bash
# First: train the model and export weights
cd tutorials/04-comparison-study/solution
python3 export_bitnet.py    # generates bitnet_weights.h in bonus/bitnet-to-c/

# Then: build the C test binary
cd ../../../bonus/bitnet-to-c
make
./bitnet_test
```

Expected output: predicted class index and confidence.

---

## Deploy to ESP32 (ESP-IDF)

1. Copy `bitnet_weights.h`, `bitnet_inference.h`, `bitnet_inference.c` into your ESP-IDF project
2. Add a `CMakeLists.txt`:
   ```cmake
   idf_component_register(
       SRCS "bitnet_inference.c"
       INCLUDE_DIRS ".")
   ```
3. Include `bitnet_inference.h` in your `main.c` and call `bitnet_forward()`
4. Build with `idf.py build`, flash with `idf.py flash`

No heap allocation required — safe for FreeRTOS tasks with small stacks.

---

## Why Ternary Weights Are 20x Smaller

Float32 uses 32 bits per weight. Ternary weights {-1, 0, +1} only need ~1.58 bits (log₂(3)). In this implementation we store them as `int8_t` (8 bits each) for simplicity, giving 4x compression. Production systems can pack multiple ternary values per byte for the full ~20x compression.

The key insight: multiplication by {-1, 0, +1} is just a conditional add/subtract. No multiplier circuit needed. On microcontrollers without hardware FPUs, this is a significant advantage.
