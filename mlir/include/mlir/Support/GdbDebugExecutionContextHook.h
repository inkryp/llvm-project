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
#include "llvm/Support/Compiler.h"

extern "C" {
void mlirDebuggerSetControl(int controlOption);

void mlirDebuggerAddSimpleBreakpoint(const char *tag);

void mlirDebuggerAddRewritePatternBreakpoint(const char *patternNameInfo);

void mlirDebuggerAddFileLineColLocBreakpoint(const char *file, unsigned line,
                                             unsigned col);
}

namespace mlir {

DebugExecutionControl
GdbCallBackFunction(ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
                    StringRef something, StringRef other, const int &depth,
                    const DebugActionInformation *daiHead);

} // namespace mlir

#endif // MLIR_SUPPORT_GDBDEBUGEXECUTIONCONTEXTHOOK_H
