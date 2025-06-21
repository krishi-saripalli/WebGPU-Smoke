# WebGPU-Smoke

<!-- ![](public/media/smoke-trimmed.mp4) -->

A grid-based fluid solver mostly based on the paper (Visual Simulation of Smoke (Fedkiw et. al 2001))[https://web.stanford.edu/class/cs237d/smoke.pdf] and (Simulating Fluids, Fire, and Smoke in Real-Time (Chan 2023))[https://andrewkchan.dev/posts/fire.html]

## Running

To run the app, you'll need a WebGPU compatible browser. If you're not sure whether you have one, check (here)[https://caniuse.com/webgpu]. You'll also need a device that supports the `float32-filterable` which allows filtered sampling of texture values for linear interpolation. Unfortunately, that means that most mobile devices cannot run the app for now.

First, install dependencies

```
pnpm i
```

Then, run

```
pnpm run dev
```

## TODO
- Get `shader-f16` working for better memory bandwidth
- Pre-baked radiance texture for more efficient lighting
