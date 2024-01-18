# shared_buffer_rng

This crate implements a buffer that's intended mainly to reduce lock contention associated with having 
multiple threads call [OsRng] at around the same time (insofar as they generate pseudorandom numbers at around the same
rate) to reseed their [ThreadRng] instances, by providing a shared buffer for those threads to read from and a dedicated 
thread that will fill that buffer as long as it's reachable either through a static variable or by at least one
consuming thread.

# Examples

## Simple

Suppose you've tried [rand::thread_rng()](https://rust-random.github.io/rand/rand/fn.thread_rng.html) and your only
complaint is one of the following:

* In a single-threaded program, too much time is spent context-switching into kernel mode to read from /dev/urandom.
* In a multithreaded program, performance slows to a crawl when the [ThreadRng] instances decide to reseed themselves
  at the same time, because they've been consuming pseudorandom numbers at about the same rate. 

For these situations, you can use [shared_buffer_rng::thread_rng()]() as a drop-in replacement for 
`rand::thread_rng()`. The thread-local instances will use the same algorithm to generate pseudorandom numbers, and
they'll generate the same amount before reseeding themselves, but they'll reseed themselves faster because most of their
seed requests will use buffered seed bytes rather than making a syscall to fetch more seed bytes.

```
let mut floats = vec![0.0, 1024];
```