//===- GdbDebugExecutionContextHook.h - GDB for Debugger Support *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: Write a proper description for the service
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H
#define MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H

#include "mlir/Support/DebugExecutionContext.h"
#include <signal.h>

static mlir::DebugExecutionControl GDB_RETURN =
    mlir::DebugExecutionControl::Apply;

extern "C" {
void mlirDebuggerSetControl(int controlOption) {
  GDB_RETURN = static_cast<mlir::DebugExecutionControl>(controlOption);
}

void mlirDebuggerAddBreakpoint(mlir::DebugExecutionContext *dbg);
}

namespace mlir {
static void *volatile sink;

// TODO: Just expose this in the header
DebugExecutionControl GdbOnBreakpoint(DebugExecutionContext *dbg) {
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

#endif // MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H
