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

namespace mlir {

static DebugExecutionControl GDB_RETURN = DebugExecutionControl::Apply;

DebugExecutionControl GdbOnBreakpoint() {
  raise(SIGTRAP);
  return GDB_RETURN;
}

extern "C" {
void mlirDebuggerSetControlApply() {
  GDB_RETURN = DebugExecutionControl::Apply;
}
}

} // namespace mlir

#endif // MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H
