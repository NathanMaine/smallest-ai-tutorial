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
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

bitnet_inference.o: bitnet_inference.c bitnet_inference.h bitnet_weights.h
bitnet_main.o:      bitnet_main.c      bitnet_inference.h bitnet_weights.h

clean:
	rm -f $(OBJS) $(BIN)
