# Nsight Compute Notes

When you profile a kernel, do not try to understand every metric at once.

## Start with these questions

1. Is this kernel mostly compute limited or memory limited?
2. Did occupancy improve after the change?
3. Did global memory traffic or achieved bandwidth change?
4. Did the kernel launch shape look sensible?

## Suggested first metrics

- achieved occupancy
- SM throughput
- memory throughput
- launch statistics

## Good habit

Before looking at the profiler, write down what you expect to see. Then compare
the report to your hypothesis.

