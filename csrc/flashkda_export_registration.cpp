// FlashKDA's forward built with the kernels in csrc/flashkda_export/, for
// prefill calls that export checkpoint states (mamba_prefill_export).
//
// The three files there replace FlashKDA 17a037d's csrc/smxx/fwd_kernel1.cuh,
// fwd_kernel2.cuh and fwd_launch.cu (MIT, csrc/flashkda_export/LICENSE). The
// recurrence kernel stores the checkpoint and the final state from the
// registers that hold them, and the two kernels are chained with programmatic
// dependent launch. Output, final state and checkpoint states are bit-identical
// to FlashKDA's; only calls with checkpoint offsets get faster. The host code,
// op signatures and workspace size are FlashKDA's.

#include "core/registration.h"
#include "flash_kda.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(_flashkda_export_C, m) {
  m.def("get_workspace_size(int T_total, int H, int N=1) -> int");
  m.def(
      "fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, "
      "Tensor(a!) out, Tensor(c!) workspace, Tensor A_log, Tensor dt_bias, "
      "float lower_bound, "
      "Tensor? initial_state=None, Tensor(b!)? final_state=None, "
      "Tensor? cu_seqlens=None, Tensor(d!)? checkpoint_state=None, "
      "Tensor? checkpoint_offsets=None) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_flashkda_export_C, CompositeExplicitAutograd, m) {
  m.impl("get_workspace_size", TORCH_BOX(&get_workspace_size));
}

STABLE_TORCH_LIBRARY_IMPL(_flashkda_export_C, CUDA, m) {
  m.impl("fwd", TORCH_BOX(&fwd));
}

REGISTER_EXTENSION(_flashkda_export_C)
