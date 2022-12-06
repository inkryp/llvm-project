//===- GdbDebugExecutionContextHook.cpp - GDB for Debugger Support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/GdbDebugExecutionContextHook.h"
#include "mlir/Support/BreakpointManagers/RewritePatternBreakpointManager.h"
#include "mlir/Support/BreakpointManagers/SimpleBreakpointManager.h"
#include <signal.h>

mlir::DebugExecutionControl GDB_RETURN = mlir::DebugExecutionControl::Apply;

extern "C" {
void mlirDebuggerSetControl(int controlOption) {
  GDB_RETURN = static_cast<mlir::DebugExecutionControl>(controlOption);
}

void mlirDebuggerAddSimpleBreakpoint(const char *test) {
  auto &sbm = mlir::SimpleBreakpointManager::getGlobalInstance();
  sbm.addBreakpoint(mlir::StringRef(test));
}

void mlirDebuggerAddRewritePatternBreakpoint(const char *test) {
  auto &breakpointManager =
      mlir::RewritePatternBreakpointManager::getGlobalInstance();
  breakpointManager.addBreakpoint(mlir::StringRef(test));
}
}

namespace mlir {

DebugExecutionControl
GdbCallBackFunction(ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
                    StringRef tag, StringRef desc, const int &depth,
                    const DebugActionInformation *daiHead) {
  raise(SIGTRAP);
  return GDB_RETURN;
}

static void *volatile sink;

LLVM_ATTRIBUTE_USED void GdbOnBreakpoint() {
  static bool initialized = [&]() {
    sink = (void *)mlirDebuggerSetControl;
    sink = (void *)mlirDebuggerAddSimpleBreakpoint;
    sink = (void *)mlirDebuggerAddRewritePatternBreakpoint;
    return true;
  }();
  (void)initialized;
}

} // namespace mlir
