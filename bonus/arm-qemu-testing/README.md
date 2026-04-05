# Bonus: Testing on ARM via QEMU

**Verify your C code runs correctly on ARM without owning ARM hardware.**

---

## Why Do This

The BitNet C code targets embedded systems (ESP32-class ARM Cortex-M). You can test it on any machine using QEMU to emulate ARM.

This also works for testing any C code that will eventually run on ARM — CI/CD pipelines, for example.

---

## What You Need

- **Docker** (install from [docker.com](https://docker.com))
- The built binary from `bonus/bitnet-to-c/`

---

## Dockerfile

Save this as `bonus/arm-qemu-testing/Dockerfile`:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    gcc-arm-linux-gnueabihf \
    qemu-user \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /test
COPY bitnet_test_arm /test/
RUN chmod +x /test/bitnet_test_arm

CMD ["qemu-arm", "-L", "/usr/arm-linux-gnueabihf", "/test/bitnet_test_arm"]
```

---

## Step-by-Step

### 1. Cross-compile for ARM

```bash
cd bonus/bitnet-to-c

# Compile for 32-bit ARM
arm-linux-gnueabihf-gcc \
    -O2 \
    -o bitnet_test_arm \
    bitnet_main.c \
    bitnet_inference.c

echo "Cross-compilation done"
```

### 2. Build the Docker image

```bash
cd bonus/arm-qemu-testing
cp ../bitnet-to-c/bitnet_test_arm .
docker build -t bitnet-arm-test .
```

### 3. Run under QEMU

```bash
docker run --rm bitnet-arm-test
```

You should see the same output as the native `./bitnet_test` binary.

---

## What QEMU Is Doing

QEMU-user mode intercepts every ARM instruction and translates it to your host's native instruction set. System calls are passed through to the host kernel. The result: the binary runs as if it's on real ARM hardware, but on your MacBook or Linux box.

For testing correctness (does the model produce the right predictions?) this is sufficient. For testing timing (how fast does inference run?) you'd want real hardware.

---

## Using This in CI

This approach works in GitHub Actions:

```yaml
jobs:
  arm-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install cross-compiler and QEMU
        run: sudo apt-get install -y gcc-arm-linux-gnueabihf qemu-user
      - name: Cross-compile
        run: |
          arm-linux-gnueabihf-gcc -O2 \
            -o bitnet_test_arm \
            bonus/bitnet-to-c/bitnet_main.c \
            bonus/bitnet-to-c/bitnet_inference.c
      - name: Run under QEMU
        run: qemu-arm -L /usr/arm-linux-gnueabihf ./bitnet_test_arm
```

---

## Troubleshooting

**`qemu-arm: Could not open '/lib/ld-linux-armhf.so.3'`**
→ Add `-L /usr/arm-linux-gnueabihf` to the qemu-arm command.

**`arm-linux-gnueabihf-gcc: command not found`**
→ `sudo apt-get install gcc-arm-linux-gnueabihf` (Debian/Ubuntu) or `brew install arm-linux-gnueabihf-binutils` (macOS).

**Segfault in QEMU**
→ Check that all arrays in `bitnet_weights.h` have the right dimensions. Regenerate with `export_bitnet.py`.
