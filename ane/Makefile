CC = xcrun clang
CFLAGS = -O2 -Wall -DACCELERATE_NEW_LAPACK -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface -framework Accelerate
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS = experiment_config.h stories_config.h stories_io.h stories_mil_dynamic.h stories_cpu_ops.h

train_ane: train_ane.m $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f train_ane

.PHONY: clean
