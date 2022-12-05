//===- GdbDebugExecutionContextHook.cpp - GDB for Debugger Support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/GdbDebugExecutionContextHook.h"
#include "mlir/Support/BreakpointManagers/SimpleBreakpointManager.h"
#include <signal.h>

mlir::DebugExecutionControl GDB_RETURN = mlir::DebugExecutionControl::Apply;

extern "C" {

void mlirDebuggerSetControl(int controlOption) {
  GDB_RETURN = static_cast<mlir::DebugExecutionControl>(controlOption);
}

void mlirDebuggerAddBreakpoint(const char *test) {
  auto &sbm = mlir::SimpleBreakpointManager::getGlobalInstance();
  sbm.addBreakpoint(mlir::StringRef(test));
}
}

namespace mlir {

static void *volatile sink;

LLVM_ATTRIBUTE_USED DebugExecutionControl
GdbOnBreakpoint(DebugExecutionContext *dbg) {
  static bool initialized = [&]() {
    sink = (void *)mlirDebuggerSetControl;
    sink = (void *)mlirDebuggerAddBreakpoint;
    return true;
  }();
  (void)initialized;
  raise(SIGTRAP);
  return GDB_RETURN;
}

} // namespace mlir
