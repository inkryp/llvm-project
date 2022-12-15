//===- GdbDebugExecutionContextHook.cpp - GDB for Debugger Support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/GdbDebugExecutionContextHook.h"
#include "mlir/Support/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/Support/BreakpointManagers/RewritePatternBreakpointManager.h"
#include "mlir/Support/BreakpointManagers/SimpleBreakpointManager.h"
#include "mlir/Support/DebugExecutionContext.h"
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

void mlirDebuggerAddFileLineColLocBreakpoint(const char *file, unsigned line,
                                             unsigned col) {
  auto &fileLineColLocBreakpointManager =
      mlir::FileLineColLocBreakpointManager::getGlobalInstance();
  fileLineColLocBreakpointManager.addBreakpoint(
      mlir::StringRef(file, strlen(file)), line, col);
}

bool mlirDebuggerDeleteBreakpoint(unsigned breakpointID) {
  auto &mp = getGlobalInstanceOfBreakpoindIdsMap();
  auto it = mp.find(breakpointID);
  if (it != mp.end()) {
    auto &breakpointInstanceById = it->second;
    auto *breakpoint = std::get<0>(breakpointInstanceById);
    auto &breakpointManager = std::get<1>(breakpointInstanceById);
    return breakpointManager.deleteBreakpoint(breakpoint);
  }
  return false;
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
    sink = (void *)mlirDebuggerAddFileLineColLocBreakpoint;
    sink = (void *)mlirDebuggerDeleteBreakpoint;
    return true;
  }();
  (void)initialized;
  raise(SIGTRAP);
  return GDB_RETURN;
}

} // namespace mlir
