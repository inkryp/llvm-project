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
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
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

bool mlirDebuggerChangeStatusOfBreakpoint(unsigned breakpointID, bool status) {
  auto &mp = getGlobalInstanceOfBreakpoindIdsMap();
  auto it = mp.find(breakpointID);
  if (it != mp.end()) {
    auto *breakpoint = std::get<0>(it->second);
    if (status) {
      breakpoint->setEnableStatusTrue();
    } else {
      breakpoint->setEnableStatusFalse();
    }
    return true;
  }
  return false;
}

bool mlirDebuggerListBreakpoints() {
  auto &mp = getGlobalInstanceOfBreakpoindIdsMap();
  if (mp.empty()) {
    return false;
  }
  llvm::dbgs() << llvm::formatv("{0,-8}", "ID")
               << llvm::formatv("{0,-15}", "Type")
               << llvm::formatv("{0,-4}", "Enb") << "Info\n";
  for (auto &[id, tuple] : mp) {
    auto *breakpoint = std::get<0>(tuple);
    std::string type;
    if (llvm::isa<mlir::SimpleBreakpoint>(*breakpoint)) {
      type = "Tag";
    } else if (llvm::isa<mlir::RewritePatternBreakpoint>(*breakpoint)) {
      type = "Pattern";
    } else if (llvm::isa<mlir::FileLineColLocBreakpoint>(*breakpoint)) {
      type = "Location";
    } else {
      type = "Unknown";
    }
    llvm::dbgs() << llvm::formatv("{0,-8}", id)
                 << llvm::formatv("{0,-15}", type)
                 << llvm::formatv("{0,-4}",
                                  breakpoint->getEnableStatus() ? 'y' : 'n')
                 << *breakpoint << '\n';
  }
  return true;
}

bool mlirDebuggerPrintAction() {
  if (auto *daiHead =
          mlir::GdbDebugExecutionContextInformation::getGlobalInstance()
              .getDebugActionInformation()) {
    daiHead->action.print(llvm::dbgs());
    return true;
  }
  return false;
}
}

namespace mlir {

GdbDebugExecutionContextInformation &
GdbDebugExecutionContextInformation::getGlobalInstance() {
  static GdbDebugExecutionContextInformation *ctx =
      new GdbDebugExecutionContextInformation();
  return *ctx;
}

void GdbDebugExecutionContextInformation::updateContents(
    ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags, StringRef tag,
    StringRef desc, const int &depth, const DebugActionInformation *daiHead) {
  this->units = units;
  this->instanceTags = instanceTags;
  this->tag = tag;
  this->desc = desc;
  this->depth = depth;
  this->daiHead = daiHead;
  idxActiveUnit = 0;
  activeUnit = this->units[idxActiveUnit];
}

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
    sink = (void *)mlirDebuggerChangeStatusOfBreakpoint;
    sink = (void *)mlirDebuggerListBreakpoints;
    sink = (void *)mlirDebuggerPrintAction;
    return true;
  }();
  (void)initialized;
  auto &ctx = GdbDebugExecutionContextInformation::getGlobalInstance();
  ctx.updateContents(units, instanceTags, tag, desc, depth, daiHead);
  raise(SIGTRAP);
  return GDB_RETURN;
}

} // namespace mlir
