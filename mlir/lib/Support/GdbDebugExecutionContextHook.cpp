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

void mlirDebuggerAddSimpleBreakpoint(const char *tag) {
  auto &simpleBreakpointManager =
      mlir::SimpleBreakpointManager::getGlobalInstance();
  simpleBreakpointManager.addBreakpoint(mlir::StringRef(tag, strlen(tag)));
}

void mlirDebuggerAddRewritePatternBreakpoint(const char *patternNameInfo) {
  auto &rewritePatternBreakpointManager =
      mlir::RewritePatternBreakpointManager::getGlobalInstance();
  rewritePatternBreakpointManager.addBreakpoint(
      mlir::StringRef(patternNameInfo, strlen(patternNameInfo)));
}
}

namespace mlir {

static void *volatile sink;

DebugExecutionControl
GdbCallBackFunction(ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
                    StringRef tag, StringRef desc, const int &depth,
                    const DebugActionInformation *daiHead) {
  static bool initialized = [&]() {
    sink = (void *)mlirDebuggerSetControl;
    sink = (void *)mlirDebuggerAddSimpleBreakpoint;
    sink = (void *)mlirDebuggerAddRewritePatternBreakpoint;
    return true;
  }();
  (void)initialized;
  raise(SIGTRAP);
  return GDB_RETURN;
}

} // namespace mlir
