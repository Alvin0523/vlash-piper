#!/bin/sh
# Use the Pixi environment libstdc++ first so pyarrow/datasets do not load an older system libstdc++ on Jetson.
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# Force Jetson-specific CUDA libraries to load before PyTorch's bundled generic ones.
# PyTorch 2.11 bundles cuBLAS/cuDNN built for generic aarch64, but Jetson's nvgpu requires
# the JetPack-provided CUDA libraries from /usr/local/cuda-12.6/lib64/.
# LD_PRELOAD ensures these are resolved before PyTorch's RPATH kicks in.
_CUDA_LIB=/usr/local/cuda-12.6/lib64
_CUDNN_LIB=/lib/aarch64-linux-gnu
_CUBLAS_PRELOAD=""
for _lib in \
  "$_CUDA_LIB/libcublas.so.12" \
  "$_CUDA_LIB/libcublasLt.so.12" \
  "$_CUDNN_LIB/libcudnn.so.9" \
  "$_CUDNN_LIB/libcudnn_ops.so.9" \
  "$_CUDNN_LIB/libcudnn_graph.so.9" \
  "$_CUDNN_LIB/libcudnn_heuristic.so.9" \
  "$_CUDNN_LIB/libcudnn_engines_precompiled.so.9" \
  "$_CUDNN_LIB/libcudnn_engines_runtime_compiled.so.9"; do
  if [ -f "$_lib" ]; then
    _CUBLAS_PRELOAD="${_CUBLAS_PRELOAD:+$_CUBLAS_PRELOAD:}$_lib"
  fi
done
if [ -n "$_CUBLAS_PRELOAD" ]; then
  export LD_PRELOAD="${_CUBLAS_PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
unset _CUDA_LIB _CUDNN_LIB _CUBLAS_PRELOAD _lib

# Expose CUDA toolkit so Triton (used by torch.compile) can find cuda.h and ptxas
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=/usr/local/cuda-12.6
export CPATH="/usr/local/cuda-12.6/include${CPATH:+:$CPATH}"
export PATH="/usr/local/cuda-12.6/bin${PATH:+:$PATH}"
# Triton 3.6 does NOT search PATH for ptxas — it requires this explicit env var
export TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxas

# X11 display for pynput keyboard support and rerun viewer (Jetson runs X server on :1)

# export DISPLAY=:1
# export XAUTHORITY=/run/user/1000/gdm/Xauthority
