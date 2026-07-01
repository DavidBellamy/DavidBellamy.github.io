---
layout: post
title: "RL weight sync takes minutes for large models but it can be 100x faster"
date: 2026-06-30
description: "In RL, the weight-sync transfer over the wire is only a couple of seconds, even for a trillion-parameter model, and barely grows with model size. The reported minutes come from weight conversion and loading around the transfer, not the network."
permalink: /blog/weight-sync/
og_image: https://davidbellamy.github.io/assets/img/weight-sync-transfer-time.png
related_posts: false
---

In RL training, the throughput metric that matters is the number of model updates per second times the number of effective samples (in the sense of Alex Smola's [post](https://alex.smola.org/posts/40-effective-sample-size/)) per update. Increase this quantity and two good things happen: RL training runs get cheaper, and you can run more experiments on the same budget.

Ask around and you will hear that weight sync can take minutes for large models, and that it is a major tax on RL training throughput.

But, as I will show, the actual transfer of weights over the wire is only a couple of seconds, and almost independent of model size. When weight sync is slow, the delay is almost never the network. It is the two steps on either side of the transfer. Here is how to tell which regime you are in, and what to do about it.

## What weight sync actually is

Weight sync is the transfer of updated weights from the trainer to the inference engine, in any setup that runs them as separate systems. That is the norm in RL training, and increasingly in other training regimes too.

Most people's mental model of weight sync looks like this:

`forward → backward → optimizer.step() → send weights → continue inference`

The real loop, in most setups, has several more steps:

`forward → backward → optimizer.step() → gather and convert weights → pause inference → flush KV cache → send weights → load into resident params → resume inference → prefill → ...`

Each middle step is a place that time can hide. We will account for all of them.

Note: you do not have to flush the KV cache. If you persist the KV cache, you skip the re-prefill when resuming inference, which saves time. The cost of doing this, though, is correctness: fresh weights attending to a stale KV cache produces off-distribution generations, which can hurt training stability and reduce the effective samples delivered to the trainer per second. To what extent this KV staleness is tolerable is, to my knowledge, an open research question. If it is tolerable to some extent, we can flush and re-prefill only every $$K$$ weight syncs and amortize the cost of prefill.

## How often does weight sync happen?

Before asking what weight sync costs, let's express how often it happens, because that bounds how much it can possibly matter to RL training throughput.

RL training is the loop: inference -> environment -> reward -> trainer -> wt sync -> repeat

Whichever step of the loop is the slowest governs the step time. Call the time between trainer updates (treating micro-steps as one big step) $$T_{\text{step}}$$. Weight sync happens once per step, so $$T_{\text{step}}$$ is also the time between weight syncs, and weight sync happens at rate $$1/T_{\text{step}}$$. Call the time to do weight sync $$T_{\text{wtsync}}$$.

Because the slowest step sets $$T_{\text{step}}$$ and inference is one of those steps, in a fully async setup $$T_{\text{step}} \ge T_{\text{inf}}$$ (both as averages). So the contribution of weight sync to the step time is the fraction $$T_{\text{wtsync}} / T_{\text{step}}$$.

### An upper bound on $$T_{\text{wtsync}} / T_{\text{step}}$$

The trouble is that $$T_{\text{step}}$$ is hard to estimate across setups because a lot of design choices feed into it. But $$T_{\text{inf}}$$ is easier to estimate: take the number of output tokens per task on a representative workload and divide by your inference speed on that workload, both of which we can borrow from [Artificial Analysis](https://artificialanalysis.ai/) for almost any model.

That gives an upper bound on $$T_{\text{wtsync}} / T_{\text{step}}$$ since $$T_{\text{step}} \ge T_{\text{inf}}$$:

$$\frac{T_{\text{wtsync}}}{T_{\text{step}}} \le \frac{T_{\text{wtsync}}}{T_{\text{inf}}}$$

The significance of weight sync to RL throughput is at most $$T_{\text{wtsync}} / T_{\text{inf}}$$, which you can compute from public numbers across a variety of settings. If this upper bound is small, we are done: optimizing weight sync will not matter much. This bound is doubly conservative in synchronous RL setups, where $$T_{\text{inf}}$$ is the *max* rollout time rather than the *mean*, making $$T_{\text{step}}$$ larger and the weight-sync fraction smaller still.

### Formalizing the impact of optimizing weight sync

Amdahl's law makes the idea of "not mattering much" precise. The speedup from optimizing a step that is fraction $$p$$ of total time, parallelized $$n$$ ways, is:

$$\frac{1}{(1 - p) + p/n}$$

If $$p \le 0.1$$, then even as $$n \to \infty$$ (perfect pipelining, the entire cost hidden), the most you can win is $$1 / 0.9 \approx 1.11$$. An 11% ceiling on system throughput, for potentially unbounded engineering effort. If your measured weight-sync fraction is under 10%, that ceiling is the whole prize, and it might not be worth chasing.

## Decomposing the cost of weight sync

Suppose you measure it and weight sync is worth optimizing. Take these step-times as fixed: forward/backward pass, `optimizer.step()`, inference prefill & decode. What is actually left to optimize?

Pausing and resuming inference is control flow, sub-millisecond. Flushing the KV cache is also sub-millisecond - it only requires marking every KV block available plus dropping any data structures that organize pointers into the KV pool (like a radix tree for shared prefixes). No data moves.

That leaves three real steps to study:

1. Gathering and converting weights on the trainer.
2. Sending the weights over the wire.
3. Loading the weights into the inference engine's resident parameters.

Let's take these in order of how much they surprise people, starting with the one most people assume is the bottleneck: the send.

## The send is (very) cheap

The fastest way to move weights is NCCL over GPUDirect RDMA on InfiniBand. Most Hopper-generation clusters have NDR NICs, which run at 400 Gb/s (50 GB/s), one per GPU. So an 8-GPU node has 400 GB/s of aggregate egress, and about 90% of that is realizable in practice. The transfer time is set by the slower side, and the slower side is always the receiver: the trainer (i.e. sender) carries gradients and optimizer state on top of weights, so one trainer replica spans far more GPUs, and far more NICs, than one inference replica that holds only weights and a KV pool.

That sender-receiver asymmetry is supposed to be benign. The standard argument is that the inference GPU count scales with the model size (the payload), so aggregate receive bandwidth scales with the payload, and payload over bandwidth is constant. The send should be a roughly fixed ~1.4 seconds at *any model size*, just the time to drain one GPU's worth of weights through one NIC.

I tested that argument directly on the real production shape: many trainer ranks fanning into the few inference ranks that hold one replica. On an Azure H200 cluster (NDR, 8 NICs per node) I transferred one replica of each of seven real models from a trainer-width sender pool into an inference-width receiver pool. Senders $$S$$ are the ranks needed to hold one model replica's ~16 bytes/param of training state (roughly $$\text{params} / 6.25\text{B}$$ GPUs). Receivers $$R$$ are the documented serving layout for these seven models. Each receiver (i.e. inference rank) pulls its shard, $$\text{weight bytes} / R$$, from all $$S$$ senders, which is the exact reshard fan-in a real sync performs.

<div class="text-center my-4">
  <img src="/assets/img/weight-sync-transfer-time.png" class="img-fluid rounded z-depth-1" data-zoomable alt="Transfer time for one replica of seven models, measured on an NDR H200 cluster at about 48 GB/s per NIC; transfer time tracks bytes per receiver GPU, not model size." />
</div>

Time to transfer one model replica from the trainer's ranks to its inference ranks, measured on an NDR H200 cluster at ~48 GB/s per NIC (near line rate). Bars are ordered by total model size, but transfer time does not follow size: it is set by the bytes each receiver GPU pulls, i.e. serving width and precision. The same DeepSeek-V3 takes 1.83 s on 8 GPUs but 0.57 s spread over 32, and Kimi-K2 (1T, INT4) beats GLM-5.2 (744B, fp8).

The measurement confirms the standard argument. Per-NIC ingress is essentially flat at ~48 GB/s across the single- and two-node configs, and even the widest expert-parallel layout only dips to ~40, where the thin 21 GB/GPU shard amortizes the per-transfer overhead less well. Even with 160 senders fanning into 8 receiver NICs, each NIC still runs at line speed, so the fan-in imposes no penalty. The transfer time is just $$C = \text{weight bytes} / R$$ over that rate, and since $$C$$ is limited by the amount of available inference HBM, it cannot grow with the model. Kimi-K2, a trillion parameters at 20:1, syncs in 1.4 seconds.

The one thing the simple story misses is that the transfer is not a single number, it is a band, because $$C$$ is a design choice. It runs from **about half a second to a bit over two seconds**, set by how thinly you shard inference and at what precision. Note that model size does not predict the transfer time: Kimi-K2 at 1T (INT4) syncs faster than GLM-5.2 at 744B (fp8), because lower precision and its serving layout put fewer bytes behind each receiver NIC. The lever is $$\text{weight bytes} / R$$. The same DeepSeek-V3 that takes 1.83 seconds packed onto 8 GPUs syncs in 0.57 seconds spread across 32. You buy a faster sync by serving wider or quantizing lower.

Scaling to $$N$$ replicas does not change the per-replica number to first order. By growing the inference fleet, you grow the trainer fleet with it, so the senders keep their egress and the receivers keep their ingress. Per-replica sync stays in the same sub-second-to-two-second band.

The one caveat is a cluster's bisection bandwidth, the network's narrowest cross-section. A single replica's receive rate is fixed by its own NICs at line speed, but sync many replicas at once and their aggregate can exceed the cluster's cross-section bandwidth, at which point bisection bandwidth, not NIC bandwidth, becomes the limit. On clusters today that is rarely the binding constraint.

## So why do people report minutes for weight sync?

If the wire transfer is a couple of seconds, where do multi-minute syncs come from? Here is a real one. In a production 400B RL run at my lab, the end-to-end weight update took about **140 seconds** per sync.

<div class="text-center my-4">
  <img src="/assets/img/weight-sync-breakdown.png" class="img-fluid rounded z-depth-1" data-zoomable alt="A 140-second production weight update: about 2 seconds of wire transfer and about 138 seconds of weight conversion and loading." />
</div>

That timer brackets the entire lifecycle, not just the send: gather each tensor out of the trainer's parallel layout, convert its layout and dtype, broadcast it, load it into the inference engine, and quantize it there, all while generation is paused. By the rate from the previous section, the wire transfer inside those 140 seconds is about **2** of them. The other **~138** seconds (99%) are the two steps on either side, and both are usually unoptimized and are easy to hide with pipelining.

Therefore, I propose renaming this problem from "weight sync" to "model transfer" as the former does not embody any reference to the trainer-side or inference-side model conversion steps, which are the actual time cost.

### What "conversion" actually involves

The trainer's weights are not in the form the inference engine wants. Converting them is a sequence of small but non-negligible transformations, including:

- **Weight conversion**: naming, tensor layout, fused vs unfused tensors, dtype and quantization, sharding and parallelism (trainer layout to inference layout), runtime packing.
- **Config conversion**: architecture metadata, RoPE and position settings, attention/MLP/MoE settings, generation defaults.
- **Tokenizer and artifact conversion**: tokenizer files, special token IDs, chat template, vocab and embedding alignment.

None of this is hard but all of it takes time, and in naive implementations it happens on the critical path while inference sits paused.

### Prepare before you pause

A faster approach is to prepare the model weights while inference is still serving, and pause inference only once the prepared weights are ready to land:

`forward → backward → optimizer.step() → full conversion on trainer → pause inference → flush KV cache → NCCL + RDMA send → inference load → resume → prefill → ...`

There is an honest cost to delaying when we pause inference. Every step of model transfer that we move ahead of pausing inference makes the inference weights that much more stale, because while we prepare the weights inference continues generating tokens using stale ones. In practice, this is fine because it only introduces 1-step async weights for around a minute per update, but it is not a completely free lunch.

## The regimes that change the optimum

There is no single right design for weight sync, because the regime influences the solution:

- Model size, and whether you send full weights or deltas.
- Interconnect speeds (InfiniBand, NIC generation, NVLink, cluster bisection). NVLink in particular changes the cost of the gather and the format conversion.
- The rollout time distribution.
- Sync vs k-step async RL.
- Flushing KV cache every sync or every $$K$$ syncs.
- Colocated vs disaggregated trainer and inference.
- The trainer's distributed optimizer (ZeRO-1, FSDP, ZeRO-3). Depending on which, you may need a DP-gather (i.e. inter-node gather) to assemble one full copy of the weights before sending.

## Can we go even faster?

There are two ways to shrink the inference 'pause bubble' caused by model transfer to near-zero.

### The shadow buffer (and why it loses)

One clean way to eliminate the bubble is a shadow buffer on the inference engines: stage the new weights in a second set of buffers while serving from the live ones, then pause only long enough to swap a pointer. The pause bubble goes to near zero.

But this costs 2x the weight memory on inference. That memory comes straight out of the KV pool, which shrinks concurrency, which lowers aggregate inference throughput, which lowers your RL update rate. This defeats our purpose.

### A staging tier

Use a separate pool of GPUs to prepare weights off the critical path, then pause the inference engines only for the final transfer.

A design sketch is a cordon of staging nodes holding one fully prepared, inference-ready copy of the model. The trainer feeds the stagers one model replica. The stagers, once inference is ready, deliver to the serving engines. If you register the inference engine's resident parameter buffers for RDMA, the stagers can write straight into them (not into a temp buffer that then gets copied). The only unavoidable delay is the staging-to-inference transfer - which is just a couple of seconds.

Whether even that delay can be hidden is an open question: can generation continue while the resident parameters are being overwritten? If yes, the bubble goes to zero. If no, we must accept to pay the tiny transfer time.

### Feeding the stagers without stalling the trainer

The trainer-to-stager transfer wants two properties. First, have the stagers **pull** rather than the trainer push, so they pull at their own NIC limit and you avoid backpressure on the fabric. This needs the trainer's weights registered for RDMA so the stagers can read them.

Second, pipeline the trainer-to-stager send with the trainer's own work. The send only reads the trainer's weights, and the next forward and backward also only read the weights, so they can run concurrently. The `optimizer.step()` is the only writer, so it is the only thing that must wait:

`forward → backward → optimizer.step() → (start send to stagers) + next forward → backward → wait for send → optimizer.step() → ...`

The trainer barely stalls. It resumes immediately and only blocks the next `optimizer.step()` on the send completing, which usually finishes inside the forward-and-backward window.

Two more optimizations:

- **Pull per-rank.** If the stagers pull directly from each trainer rank's shard, the trainer never has to gather. And pull only enough ranks to assemble one replica, not all of them. The other data-parallel replicas are redundant copies. The limit on this pull speed is therefore the cordon's aggregate ingress.
- **Pick NICs by topology.** Have the stagers pull through the NICs closest to the trainer's source memory and the inference target memory, a min-distance assignment over the fabric.

One caveat: pipelining the trainer-to-stager transfer against the trainer's next forward and backward puts two bandwidth-heavy flows on the same fabric at once (the transfer and the gradient all-reduce) - these can collide. 

## The takeaway

Model transfer (aka weight sync) is worth exactly as much attention as your measurement says. Compute $$T_{\text{wtsync}} / T_{\text{inf}}$$ before you write a line of optimization code. If it is under ~10%, Amdahl caps your entire upside at ~11%, and you might prioritize optimizing something else depending on your team's bandwidth (pun intended).

When model transfer time does matter, *the network is not the main problem*. The over-the-wire transfer is $$O(1)$$ seconds and stays there across model sizes and replica counts, because receive bandwidth scales with the payload. The minutes of model transfer that people report stem from the two steps around the transfer, model conversion and model loading, and both can move off the critical path: stage on a separate pool of GPUs, register inference rank buffers for direct RDMA, pull instead of push, and pipeline the trainer-to-stager send against the trainer's own forward and backward.

None of this makes model transfer intrinsically faster. It removes the transfer from the critical path. That is the actual game in weight sync, and in a lot of systems work: the expensive-looking number is cheap once you measure it, and the real work is somewhere else.