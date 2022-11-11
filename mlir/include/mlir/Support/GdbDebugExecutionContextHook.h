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

#include "llvm/Support/Compiler.h"
#include "mlir/Support/DebugExecutionContext.h"

extern mlir::DebugExecutionControl GDB_RETURN;

extern "C" {
void mlirDebuggerSetControl(int controlOption);

void mlirDebuggerAddBreakpoint(const char *test);
}

namespace mlir {

LLVM_ATTRIBUTE_USED DebugExecutionControl GdbOnBreakpoint(DebugExecutionContext *dbg);

} // namespace mlir

#endif // MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H
